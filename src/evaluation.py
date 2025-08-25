import copy
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.distance import squareform

import processing
from embeddingstate import EmbeddingState


def compute_kruskal_stress(
    dists_highdim: npt.NDArray[np.float32], dists_lowdim: npt.NDArray[np.float32]
) -> float:
    if not dists_highdim.any():
        print("WARNING: Highdim distances are all 0. Returning *absolute* stress.")
        stress_numerator = np.sum((dists_highdim - dists_lowdim) ** 2)
        return np.sqrt(stress_numerator)

    scaling_factor = processing.compute_distance_scaling(dists_highdim, dists_lowdim)

    dists_lowdim_scaled = dists_lowdim * scaling_factor

    stress_numerator = np.sum((dists_highdim - dists_lowdim_scaled) ** 2)
    stress_denominator = np.sum(dists_highdim**2)

    return np.sqrt(stress_numerator / stress_denominator)


def compute_kruskal_stress_partition(
    dists_highdim: npt.NDArray[np.float32],
    dists_lowdim: npt.NDArray[np.float32],
    partition: dict[int, npt.NDArray[np.int32]],
) -> float:
    # Holds sum of Kruskal stress for each community
    kruskal_com = 0.0
    # Counts number of used communities (communities with at least 2 nodes)
    community_count = 0

    for community_nodes in list(partition.values()):
        # skip communities with less than 2 nodes
        if len(community_nodes) < 2:
            continue

        # extract relevant distances for community nodes
        dists_highdim_com = np.take(dists_highdim, community_nodes, axis=0)
        dists_highdim_com = np.take(dists_highdim_com, community_nodes, axis=1)

        dists_lowdim_com = np.take(dists_lowdim, community_nodes, axis=0)
        dists_lowdim_com = np.take(dists_lowdim_com, community_nodes, axis=1)

        # convert to condensed form
        dists_highdim_com = squareform(dists_highdim_com)
        dists_lowdim_com = squareform(dists_lowdim_com)

        kruskal_com += compute_kruskal_stress(dists_highdim_com, dists_lowdim_com)
        community_count += 1

    # normalize by number of communities
    return kruskal_com / community_count


def coranking_matrix(
    r1: npt.NDArray[np.int32], r2: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    assert r1.shape == r2.shape
    crm = np.zeros(r1.shape)
    m = len(crm)

    m = max(r1.max(), r2.max()) + 1

    crm, _, _ = np.histogram2d(
        r1.ravel(), r2.ravel(), bins=(m, m), range=[[0, m], [0, m]]
    )

    return crm


def compute_metrics(
    highdim_df: pd.DataFrame,
    embeddings: list[EmbeddingState],
    target_features: list[str],
    fixed_k: int | None = None,
    ranking_metrics: bool = True,
    distance_metrics: bool = True,
    inplace: bool = False,
    verbose: bool = False,
) -> list[EmbeddingState]:
    if not inplace:
        for i, emb in enumerate(embeddings):
            embeddings[i] = copy.deepcopy(emb)

    # compute pairwise distances + ranking matrix for highdim data
    dists_highdim = processing.compute_pairwise_dists(highdim_df)

    if ranking_metrics:
        rank_highdim = [np.argsort(np.argsort(row)) for row in dists_highdim]

    # compute pairwise distances for reference data to compute differences in community-stress.
    # Assuming that the reference data is the same for all embeddings and is given by the first embedding.
    if distance_metrics:
        reference_df = pd.DataFrame(embeddings[0].embedding.values(), index=None)
        reference_lowdim = processing.compute_pairwise_dists(reference_df)

    for emb in embeddings:
        if verbose:
            print("------------------------------------------------------------")
            print(f"Computing global metrics for embedding: `{emb.title}'.")
            start_time = time.time()

        # compute pairwise distances + ranking matrix for lowdim data
        lowdim_df = pd.DataFrame(emb.embedding.values(), index=None)
        dists_lowdim = processing.compute_pairwise_dists(lowdim_df)
        rank_lowdim = [np.argsort(np.argsort(row)) for row in dists_lowdim]

        # The computation of the co-ranking matrix and its associated metrics is adapted from the
        # pyDRMetrics package (https://github.com/zhangys11/pyDRMetrics) by Yinsheng Zhang (oo@zju.edu.cn / zhangys@illinois.edu)
        # Original code licensed under Creative Commons Attribution 4.0 International (CC BY 4.0)
        # See: https://creativecommons.org/licenses/by/4.0/
        # Changes: Only compute the AUC of the metrics if needed,
        # not automatically while computing the co-ranking matrix (for runtime efficiency).
        if ranking_metrics:
            cr_matrix = coranking_matrix(
                np.asarray(rank_highdim, dtype=int),
                np.asarray(rank_lowdim, dtype=int),
            )

            cr_matrix = cr_matrix[1:, 1:]
            n = len(cr_matrix)

            trustworthiness = np.zeros(n - 1)  # trustworthiness
            continuity = np.zeros(n - 1)  # continuity
            r_quality = np.zeros(n - 1)  # R-Quality (Rnx)
            qnn = np.zeros(n)  # Co-k-nearest neighbor size

            cr_matrix_cumsum = np.cumsum(np.cumsum(cr_matrix, axis=0), axis=1)
            diag_idxs = np.arange(n)
            qnn = cr_matrix_cumsum[diag_idxs, diag_idxs] / ((diag_idxs + 1) * n)

            # TODO: add further validation for fixed_k
            if fixed_k is None:
                for k in range(n - 1):
                    qs = cr_matrix[k:, :k]
                    w = np.arange(
                        qs.shape[0]
                    ).reshape(
                        -1, 1
                    )  # a column vector of weights. weight = rank error = actual_rank - k
                    trustworthiness[k] = (
                        1 - np.sum(qs * w) / (k + 1) / n / (n - 1 - k)
                    )  # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
                    w = np.arange(qs.shape[1]).reshape(
                        1, -1
                    )  # a row vector of weights. weight = rank error = actual_rank - k
                    continuity[k] = 1 - np.sum(qs * w) / (k + 1) / n / (
                        n - 1 - k
                    )  # 1 - normalized hard-k-extrusions. upper-right region

                    r_quality[k] = (n * qnn[k - 1] - k) / (n - k)

                emb.metrics["trustworthiness"] = np.mean(trustworthiness)
                emb.metrics["continuity"] = np.mean(continuity)
                emb.metrics["rnx"] = np.mean(r_quality)

            else:
                k = fixed_k
                qs = cr_matrix[k:, :k]
                w = np.arange(qs.shape[0]).reshape(
                    -1, 1
                )  # a column vector of weights. weight = rank error = actual_rank - k
                trustworthiness = (
                    1 - np.sum(qs * w) / (k + 1) / n / (n - 1 - k)
                )  # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
                w = np.arange(qs.shape[1]).reshape(
                    1, -1
                )  # a row vector of weights. weight = rank error = actual_rank - k
                continuity = 1 - np.sum(qs * w) / (k + 1) / n / (
                    n - 1 - k
                )  # 1 - normalized hard-k-extrusions. upper-right region
                # R = ((m - 1) * QNN[k-1] - k) / (m - 1 - k)
                r_quality = (n * qnn[k - 1] - k) / (n - k)

                emb.metrics["trustworthiness"] = trustworthiness
                emb.metrics["continuity"] = continuity
                emb.metrics["rnx"] = r_quality

            rank_score_list = [
                emb.metrics["trustworthiness"],
                emb.metrics["continuity"],
                emb.metrics["rnx"],
            ]
            rank_score_nominator = np.sum(rank_score_list)

            # avoid division by zero if all scores are zero (not computed)
            if rank_score_nominator > 0.0:
                # compute global rank score as the average of the non-zero (i. e. computed) scores
                rank_score_denominator = len([x for x in rank_score_list if x > 0.0])
                emb.metrics["rank_score"] = (
                    rank_score_nominator / rank_score_denominator
                )

            emb.metrics["coranking_matrix"] = cr_matrix

        if distance_metrics:
            if emb.com_partition is None:
                # if no communities are defined, use the whole embedding as one community
                emb.com_partition = {0: np.arange(len(emb.embedding))}

            dists_highdim_feat = processing.compute_pairwise_dists(
                highdim_df, sim_features=target_features, invert=False
            )

            emb.metrics["sim_stress_com"] = compute_kruskal_stress_partition(
                dists_highdim_feat, dists_lowdim, emb.com_partition
            )

            # compute differences in community-stress
            reference_com_stress = compute_kruskal_stress_partition(
                dists_highdim_feat, reference_lowdim, emb.com_partition
            )
            emb.metrics["sim_stress_com_diff"] = (
                emb.metrics["sim_stress_com"] - reference_com_stress
            )

            dists_highdim_feat = squareform(dists_highdim_feat)
            dists_lowdim = squareform(dists_lowdim)

            emb.metrics["sim_stress"] = compute_kruskal_stress(
                dists_highdim_feat, dists_lowdim
            )

            stress_com_diff_norm = (emb.metrics["sim_stress_com_diff"] + 1) / 2

            distance_score_list = [
                emb.metrics["sim_stress"],
                stress_com_diff_norm,
            ]
            distance_score_nominator = np.sum(distance_score_list)

            # avoid division by zero if all scores are zero (not computed)
            if distance_score_nominator > 0.0:
                # compute rank score as the average of the non-zero (i. e. computed) scores
                distance_score_denominator = len(
                    [x for x in distance_score_list if x > 0.0]
                )
                emb.metrics["distance_score"] = 1 - (
                    distance_score_nominator / distance_score_denominator
                )

        emb.metrics["total_score"] = metric_total_score(emb)

        if verbose:
            end_time = time.time()
            print(f"Computation finished after {end_time - start_time:.2f} seconds")
            print("------------------------------------------------------------")

    return embeddings


def metric_total_score(
    embedding_obj: EmbeddingState,
    balance: float | None = 0.5,
) -> float:
    return (
        balance * embedding_obj.metrics["rank_score"]
        + (1 - balance) * embedding_obj.metrics["distance_score"]
    )


def create_report(
    embeddings: list[EmbeddingState], metadata: bool = True, metrics: bool = True
) -> pd.DataFrame:
    if not metadata and not metrics:
        raise ValueError("At least one of `metadata` or `metrics` must be True.")

    if not embeddings:
        return pd.DataFrame()

    if metadata and not metrics:
        df = pd.DataFrame(
            [
                {
                    "obj_id": e.obj_id,
                    **e.metadata,
                }
                for e in embeddings
            ]
        )
        return df

    if not metadata and metrics:
        df = pd.DataFrame(
            [
                {
                    "obj_id": e.obj_id,
                    **e.metrics,
                }
                for e in embeddings
            ]
        )

    # both metadata and metrics are True
    df = pd.DataFrame(
        [
            {
                "obj_id": e.obj_id,
                **e.metadata,
                **e.metrics,
            }
            for e in embeddings
        ]
    )
