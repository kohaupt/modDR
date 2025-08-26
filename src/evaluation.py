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
    if dists_highdim.shape != dists_lowdim.shape:
        raise ValueError(
            f"Shape mismatch (dists_highdim.shape={dists_highdim.shape}, dists_lowdim.shape={dists_lowdim.shape}): "  # noqa: E501
            f"Both arrays must have the same shape."
        )

    # convert to vector form if necessary as distances must not be used more than once
    if dists_highdim.ndim != 1:
        dists_highdim = squareform(dists_highdim)

    if dists_lowdim.ndim != 1:
        dists_lowdim = squareform(dists_lowdim)

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
    if dists_highdim.shape != dists_lowdim.shape:
        raise ValueError(
            f"Shape mismatch (dists_highdim.shape={dists_highdim.shape}, dists_lowdim.shape={dists_lowdim.shape}): "  # noqa: E501
            f"Both arrays must have the same shape."
        )

    # convert to square form if necessary, as extraction of distances requires 2D arrays
    if dists_highdim.ndim != 2:
        dists_highdim = squareform(dists_highdim)
    if dists_lowdim.ndim != 2:
        dists_lowdim = squareform(dists_lowdim)

    # accumulated sum of Kruskal stress for each community
    kruskal_com = 0.0
    # counts number of used communities (communities with at least 2 nodes)
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


# The computation of the co-ranking matrix and its associated metrics is adapted from the
# pyDRMetrics package (https://github.com/zhangys11/pyDRMetrics) by Yinsheng Zhang (oo@zju.edu.cn / zhangys@illinois.edu)
# Original code licensed under Creative Commons Attribution 4.0 International (CC BY 4.0)
# See: https://creativecommons.org/licenses/by/4.0/
# Changes: Only compute the AUC of the metrics if needed,
# not automatically while computing the co-ranking matrix (for runtime efficiency).
def coranking_matrix(
    r1: npt.NDArray[np.int32], r2: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    if r1.shape != r2.shape:
        raise ValueError(
            f"Shape mismatch (r1.shape={r1.shape}, r2.shape={r2.shape}): "
            f"Both arrays must have the same shape."
        )
    crm = np.zeros(r1.shape)
    m = len(crm)

    m = max(r1.max(), r2.max()) + 1

    crm, _, _ = np.histogram2d(
        r1.ravel(), r2.ravel(), bins=(m, m), range=[[0, m], [0, m]]
    )

    return crm


def compute_trustworthiness(cr_matrix: npt.NDArray[np.float32], k: int) -> float:
    n = len(cr_matrix)
    if k < 1 or k >= n:
        raise ValueError(f"Invalid k: {k}. It must be in the range [1, {n - 1}].")

    qs = cr_matrix[k:, :k]
    w = np.arange(qs.shape[0]).reshape(-1, 1)
    return 1 - np.sum(qs * w) / k / n / (n - 1 - k)


def compute_continuity(cr_matrix: npt.NDArray[np.float32], k: int) -> float:
    n = len(cr_matrix)
    if k < 1 or k >= n:
        raise ValueError(f"Invalid k: {k}. It must be in the range [1, {n - 1}].")

    qs = cr_matrix[:k, k:]
    w = np.arange(qs.shape[1]).reshape(1, -1)
    return 1 - np.sum(qs * w) / k / n / (n - 1 - k)


def compute_rnx(qnn: npt.NDArray[np.float32], k: int) -> float:
    n = len(qnn)
    if k < 1 or k >= n:
        raise ValueError(f"Invalid k: {k}. It must be in the range [1, {n - 1}].")

    return (n * qnn[k - 1] - k) / (n - k)


def compute_rank_score(embedding: EmbeddingState) -> float:
    rank_score_list = [
        embedding.metrics.get("trustworthiness", None),
        embedding.metrics.get("continuity", None),
        embedding.metrics.get("rnx", None),
    ]

    if any(x is None for x in rank_score_list):
        raise ValueError(
            "All of the following metrics must be computed to compute the rank score:"
            "`trustworthiness`, `continuity`, `rnx`."
        )

    rank_score_nominator = np.sum(rank_score_list)
    return rank_score_nominator / len(rank_score_list)


def compute_dist_score(embedding: EmbeddingState) -> float:
    required_keys = ["sim_stress", "sim_stress_com_diff"]
    if not all(metric in embedding.metrics for metric in required_keys):
        raise ValueError(
            "All of the following metrics must be computed to compute the distance score:"  # noqa: E501
            "`sim_stress`, `sim_stress_com_diff`."
        )

    # normalize to [0, 1]
    stress_com_diff_norm = (embedding.metrics["sim_stress_com_diff"] + 1) / 2

    distance_score_list = [
        embedding.metrics["sim_stress"],
        stress_com_diff_norm,
    ]

    distance_score_nominator = np.sum(distance_score_list)
    return 1 - (distance_score_nominator / len(distance_score_list))


def compute_total_score(
    embedding: EmbeddingState,
    balance: float = 0.5,
) -> float:
    required_keys = ["rank_score", "distance_score"]
    if not all(metric in embedding.metrics for metric in required_keys):
        raise ValueError(
            "All of the following metrics must be computed to compute the total score:"
            "`rank_score`, `distance_score`."
        )

    return (
        balance * embedding.metrics["rank_score"]
        + (1 - balance) * embedding.metrics["distance_score"]
    )


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
    # assuming that the reference data is the same for all embeddings and is given by the first embedding.
    if distance_metrics:
        reference_df = pd.DataFrame(embeddings[0].embedding.values(), index=None)
        reference_lowdim = processing.compute_pairwise_dists(reference_df)

    for emb in embeddings:
        if verbose:
            print("------------------------------------------------------------")
            print(f"Computing global metrics for embedding: `{emb.title}'.")
            start_time = time.time()

        # The computation of the co-ranking matrix and its associated metrics is adapted from the
        # pyDRMetrics package (https://github.com/zhangys11/pyDRMetrics) by Yinsheng Zhang (oo@zju.edu.cn / zhangys@illinois.edu)
        # Original code licensed under Creative Commons Attribution 4.0 International (CC BY 4.0)
        # See: https://creativecommons.org/licenses/by/4.0/
        # Changes: Only compute the AUC of the metrics if needed,
        # not automatically while computing the co-ranking matrix (for runtime efficiency).

        # compute pairwise distances + ranking matrix for lowdim data
        lowdim_df = pd.DataFrame(emb.embedding.values(), index=None)
        dists_lowdim = processing.compute_pairwise_dists(lowdim_df)
        rank_lowdim = [np.argsort(np.argsort(row)) for row in dists_lowdim]

        if ranking_metrics:
            cr_matrix = coranking_matrix(
                np.asarray(rank_highdim, dtype=int),
                np.asarray(rank_lowdim, dtype=int),
            )

            cr_matrix = cr_matrix[1:, 1:]
            n = len(cr_matrix)

            trustworthiness = np.zeros(n - 1)
            continuity = np.zeros(n - 1)
            r_quality = np.zeros(n - 1)
            qnn = np.zeros(n)

            # compute cumulative sums for QNN
            cr_matrix_cumsum = np.cumsum(np.cumsum(cr_matrix, axis=0), axis=1)
            diag_k = np.arange(n)
            qnn = cr_matrix_cumsum[diag_k, diag_k] / ((diag_k + 1) * n)

            # compute metrics for single k if fixed_k is set
            if fixed_k is not None:
                if fixed_k < 1 or fixed_k >= n:
                    raise ValueError(
                        f"Invalid fixed_k: {fixed_k}. "
                        f"It must be in the range [1, {n - 1}]."
                    )

                emb.metrics["trustworthiness"] = compute_trustworthiness(
                    cr_matrix, fixed_k
                )
                emb.metrics["continuity"] = compute_continuity(cr_matrix, fixed_k)
                emb.metrics["rnx"] = compute_rnx(qnn, fixed_k)
            # compute AUC values (mean over all ks) if fixed_k is not set
            else:
                for k in range(1, n):
                    trustworthiness[k - 1] = compute_trustworthiness(cr_matrix, k)
                    continuity[k - 1] = compute_continuity(cr_matrix, k)
                    r_quality[k - 1] = compute_rnx(qnn, k)

                emb.metrics["trustworthiness"] = np.mean(trustworthiness)
                emb.metrics["continuity"] = np.mean(continuity)
                emb.metrics["rnx"] = np.mean(r_quality)

            emb.metrics["rank_score"] = compute_rank_score(emb)
            emb.metrics["coranking_matrix"] = cr_matrix

        if distance_metrics:
            if emb.com_partition is None:
                # if no communities are defined, use the whole embedding as one community  # noqa: E501
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

            emb.metrics["distance_score"] = compute_dist_score(emb)

        emb.metrics["total_score"] = compute_total_score(emb)

        if verbose:
            end_time = time.time()
            print(f"Computation finished after {end_time - start_time:.2f} seconds")
            print("------------------------------------------------------------")

    return embeddings


def create_report(
    embeddings: list[EmbeddingState], metadata: bool = True, metrics: bool = True
) -> pd.DataFrame:
    if not metadata and not metrics:
        raise ValueError("At least one of `metadata` or `metrics` must be True.")

    if not embeddings:
        return pd.DataFrame()

    if metadata and not metrics:
        emb_df = pd.DataFrame(
            [
                {
                    "obj_id": e.obj_id,
                    **e.metadata,
                }
                for e in embeddings
            ]
        )
        return emb_df

    if not metadata and metrics:
        emb_df = pd.DataFrame(
            [
                {
                    "obj_id": e.obj_id,
                    **e.metrics,
                }
                for e in embeddings
            ]
        )
        return emb_df.drop(columns=["coranking_matrix"])

    # both metadata and metrics are True
    emb_df = pd.DataFrame(
        [
            {
                "obj_id": e.obj_id,
                **e.metadata,
                **e.metrics,
            }
            for e in embeddings
        ]
    )

    return emb_df.drop(columns=["coranking_matrix"])
