import copy
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.neighbors import KDTree

import processing
from embeddingstate import EmbeddingState


def compute_kruskal_stress(
    highdim_dists: npt.NDArray[np.float32], lowdim_dists: npt.NDArray[np.float32]
) -> float:
    if not highdim_dists.any():
        print("WARNING: Highdim distances are all 0. Returning *absolute* stress.")
        stress_numerator = np.sum((highdim_dists - lowdim_dists) ** 2)
        return np.sqrt(stress_numerator)

    scaling_factor = processing.compute_distance_scaling(highdim_dists, lowdim_dists)

    lowdim_dists_scaled = lowdim_dists * scaling_factor

    stress_numerator = np.sum((highdim_dists - lowdim_dists_scaled) ** 2)
    stress_denominator = np.sum(highdim_dists**2)

    return np.sqrt(stress_numerator / stress_denominator)


def compute_kruskal_stress_community(
    highdim_dists: npt.NDArray[np.float32],
    lowdim_dists: npt.NDArray[np.float32],
    com_partition: dict[int, npt.NDArray[np.int32]],
) -> float:
    kruskal_com = 0.0
    community_count = 0
    for part_nodes in list(com_partition.values()):
        if len(part_nodes) < 2:
            # skip communities with less than 2 nodes
            continue

        highdim_com_dists = np.take(highdim_dists, part_nodes, axis=0)
        highdim_com_dists = np.take(highdim_com_dists, part_nodes, axis=1)

        lowdim_com_dists = np.take(lowdim_dists, part_nodes, axis=0)
        lowdim_com_dists = np.take(lowdim_com_dists, part_nodes, axis=1)

        highdim_com_dists = squareform(highdim_com_dists)
        lowdim_com_dists = squareform(lowdim_com_dists)

        kruskal_com += compute_kruskal_stress(highdim_com_dists, lowdim_com_dists)
        community_count += 1

    # normalize by number of communities
    return kruskal_com / community_count


def metric_spearman(
    highdim_dists: npt.NDArray[np.float32], lowdim_dists: npt.NDArray[np.float32]
) -> float:
    return spearmanr(highdim_dists.flatten(), lowdim_dists.flatten())[0]


# -----------------------------------------------------------------------------
# From https://github.com/lmcinnes/umap/blob/master/umap/plot.py
def submatrix(dmat: ArrayLike, indices_col: ArrayLike, n_neighbors: int) -> ArrayLike:
    """Return a submatrix given an orginal matrix and the indices to keep.

    Parameters
    ----------
    dmat: array, shape (n_samples, n_samples)
        Original matrix.

    indices_col: array, shape (n_samples, n_neighbors)
        Indices to keep. Each row consists of the indices of the columns.

    n_neighbors: int
        Number of neighbors.

    Returns
    -------
    submat: array, shape (n_samples, n_neighbors)
        The corresponding submatrix.
    """
    n_samples_transform, n_samples_fit = dmat.shape
    submat = np.zeros((n_samples_transform, n_neighbors), dtype=dmat.dtype)
    for i in range(n_samples_transform):
        for j in range(n_neighbors):
            submat[i, j] = dmat[i, indices_col[i, j]]
    return submat


def _nhood_compare(
    indices_left: ArrayLike, indices_right: ArrayLike
) -> npt.NDArray[np.float32]:
    """Compute Jaccard index of two neighborhoods"""
    result = np.empty(indices_left.shape[0], dtype=np.float32)

    for i in range(indices_left.shape[0]):
        # with numba.objmode(intersection_size="intp"):
        #     intersection_size = np.intersect1d(
        #         indices_left[i], indices_right[i], assume_unique=True
        #     ).shape[0]
        intersection_size = np.intersect1d(
            indices_left[i], indices_right[i], assume_unique=True
        ).shape[0]
        union_size = np.unique(np.hstack((indices_left[i], indices_right[i]))).shape[0]
        result[i] = float(intersection_size) / float(union_size)

    return result


def _nhood_search(
    highd_data: ArrayLike, nhood_size: int
) -> tuple[npt.NDArray[np.float32], ArrayLike]:
    dmat = processing.compute_pairwise_dists(highd_data)
    indices = np.argpartition(dmat, nhood_size)[:, :nhood_size]
    dmat_shortened = submatrix(dmat, indices, nhood_size)

    indices_sorted = np.argsort(dmat_shortened)
    indices = submatrix(indices, indices_sorted, nhood_size)
    dists = submatrix(dmat_shortened, indices_sorted, nhood_size)

    return indices, dists


def compute_jaccard_distances(
    highd_data: ArrayLike, lowd_points: ArrayLike, nhood_size: int = 15
):
    highd_indices, highd_dists = _nhood_search(highd_data, nhood_size)
    tree = KDTree(lowd_points)
    lowd_dists, lowd_indices = tree.query(lowd_points, k=nhood_size)
    accuracy = _nhood_compare(
        highd_indices.astype(np.int32), lowd_indices.astype(np.int32)
    )

    return accuracy


def _nhood_search_unlimited(data: ArrayLike) -> npt.NDArray[np.float32]:
    dmat = processing.compute_pairwise_dists(data)
    indices_sorted = np.argsort(dmat)
    # dists = dmat[indices_sorted]

    # return only the indices of the neighbors without distance from a point to itself
    indices_sorted_cleaned = np.empty((0, indices_sorted.shape[1] - 1), dtype=np.int32)

    for i in range(indices_sorted.shape[0]):
        indices_sorted_cleaned = np.vstack(
            [indices_sorted_cleaned, indices_sorted[i][indices_sorted[i] != i]]
        )

    return indices_sorted_cleaned


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Based on pyDRMetrics (https://github.com/zhangys11/pyDRMetrics)
# Reference
# .. [1] Zhang, Y., Shang, Q., & Zhang, G. (2021). pyDRMetrics-A Python toolkit for dimensionality reduction quality assessment. Heliyon, 7(2).


def coranking_matrix(R1: npt.NDArray[int], R2: npt.NDArray[int]) -> npt.NDArray[int]:
    assert R1.shape == R2.shape
    Q = np.zeros(R1.shape)
    m = len(Q)

    m = max(R1.max(), R2.max()) + 1

    Q, _, _ = np.histogram2d(
        R1.ravel(), R2.ravel(), bins=(m, m), range=[[0, m], [0, m]]
    )

    return Q


def compute_global_metrics(
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
    D_highdim = processing.compute_pairwise_dists(highdim_df)

    if ranking_metrics:
        D_highdim_rank = [np.argsort(np.argsort(row)) for row in D_highdim]

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
        D_lowdim = processing.compute_pairwise_dists(lowdim_df)
        D_lowdim_rank = [np.argsort(np.argsort(row)) for row in D_lowdim]

        # The computation of the co-ranking matrix and its associated metrics is adapted from the
        # pyDRMetrics package (https://github.com/zhangys11/pyDRMetrics) by Yinsheng Zhang (oo@zju.edu.cn / zhangys@illinois.edu)
        # Original code licensed under Creative Commons Attribution 4.0 International (CC BY 4.0)
        # See: https://creativecommons.org/licenses/by/4.0/
        # Changes: Only compute the AUC of the metrics if needed,
        # not automatically while computing the co-ranking matrix (for runtime efficiency).
        if ranking_metrics:
            cr_matrix = coranking_matrix(
                np.asarray(D_highdim_rank, dtype=int),
                np.asarray(D_lowdim_rank, dtype=int),
            )

            Q = cr_matrix[1:, 1:]
            m = len(Q)

            T = np.zeros(m - 1)  # trustworthiness
            C = np.zeros(m - 1)  # continuity
            R = np.zeros(m - 1)  # R-Quality (Rnx)
            QNN = np.zeros(m)  # Co-k-nearest neighbor size

            Q_cumsum = np.cumsum(np.cumsum(Q, axis=0), axis=1)
            diag_idxs = np.arange(m)
            QNN = Q_cumsum[diag_idxs, diag_idxs] / ((diag_idxs + 1) * m)

            # TODO: add further validation for fixed_k
            if fixed_k is None:
                for k in range(m - 1):
                    Qs = Q[k:, :k]
                    W = np.arange(
                        Qs.shape[0]
                    ).reshape(
                        -1, 1
                    )  # a column vector of weights. weight = rank error = actual_rank - k
                    T[k] = (
                        1 - np.sum(Qs * W) / (k + 1) / m / (m - 1 - k)
                    )  # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
                    W = np.arange(Qs.shape[1]).reshape(
                        1, -1
                    )  # a row vector of weights. weight = rank error = actual_rank - k
                    C[k] = 1 - np.sum(Qs * W) / (k + 1) / m / (
                        m - 1 - k
                    )  # 1 - normalized hard-k-extrusions. upper-right region

                    R[k] = (m * QNN[k - 1] - k) / (m - k)

                emb.metrics["trustworthiness"] = np.mean(T)
                emb.metrics["continuity"] = np.mean(C)
                emb.metrics["rnx"] = np.mean(R)

            else:
                k = fixed_k
                Qs = Q[k:, :k]
                W = np.arange(Qs.shape[0]).reshape(
                    -1, 1
                )  # a column vector of weights. weight = rank error = actual_rank - k
                T = (
                    1 - np.sum(Qs * W) / (k + 1) / m / (m - 1 - k)
                )  # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
                W = np.arange(Qs.shape[1]).reshape(
                    1, -1
                )  # a row vector of weights. weight = rank error = actual_rank - k
                C = 1 - np.sum(Qs * W) / (k + 1) / m / (
                    m - 1 - k
                )  # 1 - normalized hard-k-extrusions. upper-right region
                # R = ((m - 1) * QNN[k-1] - k) / (m - 1 - k)
                R = (m * QNN[k - 1] - k) / (m - k)

                emb.metrics["trustworthiness"] = T
                emb.metrics["continuity"] = C
                emb.metrics["rnx"] = R

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

            D_highdim_feat = processing.compute_pairwise_dists(
                highdim_df, sim_features=target_features, invert=False
            )

            emb.metrics["sim_stress_com"] = compute_kruskal_stress_community(
                D_highdim_feat, D_lowdim, emb.com_partition
            )

            # compute differences in community-stress
            reference_com_stress = compute_kruskal_stress_community(
                D_highdim_feat, reference_lowdim, emb.com_partition
            )
            emb.metrics["sim_stress_com_diff"] = (
                emb.metrics["sim_stress_com"] - reference_com_stress
            )

            D_highdim_feat = squareform(D_highdim_feat)
            D_lowdim = squareform(D_lowdim)

            emb.metrics["sim_stress"] = compute_kruskal_stress(D_highdim_feat, D_lowdim)

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


# -----------------------------------------------------------------------------


def compute_pairwise_metrics(
    highdim_data: npt.NDArray[np.float32],
    embeddings: list[EmbeddingState],
    inplace: bool = False,
    verbose: bool = False,
) -> list[EmbeddingState]:
    if not inplace:
        for i, emb in enumerate(embeddings):
            embeddings[i] = copy.deepcopy(emb)

    for emb in embeddings:
        if verbose:
            print("------------------------------------------------------------")
            print(f"Computing global metrics for embedding: `{emb.title}'.")
            start_time = time.time()

        emb.metrics["jaccard"] = compute_jaccard_distances(
            highdim_data,
            np.array(list(emb.embedding.values())),
            nhood_size=emb.metadata["k_neighbors"],
        )

        if verbose:
            end_time = time.time()
            print(f"Computation finished after {end_time - start_time:.2f} seconds")
            print("------------------------------------------------------------")

    return embeddings


def compute_sequence_diff(
    reference_points: ArrayLike,
    embedding_points: ArrayLike,
    nhood_size: int = 15,
) -> npt.NDArray[np.float32]:
    highd_indices = _nhood_search_unlimited(reference_points)
    lowd_indices = _nhood_search_unlimited(embedding_points)

    seq_diffs = np.zeros((embedding_points.shape[0]), dtype=np.float32)

    # Iterate over all points in the low-dimensional space
    for i in range(embedding_points.shape[0]):
        sum_highd = 0.0
        sum_lowd = 0.0

        # Iterate over indices of nearest neighbors of the current point
        # (in the low-dimensional space)
        for pos_index in range(nhood_size):
            # Calculate the high-dimensional position of the 'pos_index'-nearest-neighbor
            # in the low-dimensional space
            p_i2 = pos_index
            dim_nn = np.where(highd_indices[i] == lowd_indices[i][pos_index])
            assert len(dim_nn[0]) == 1, (
                f"Point {i}: The {lowd_indices[i][pos_index]}-nearest neighbor "
                f"was not assigned to exactly one match (index) "
                f"in high-dimensional space."
            )

            p_in = dim_nn[0][0]
            sum_lowd += (nhood_size - p_i2) * np.abs(p_i2 - p_in)

            # Calculate the low-dimensional position of the 'pos_index'-nearest-neighbor
            # in the high-dimensional space
            p_in = pos_index
            dim_nn = np.where(highd_indices[i][pos_index] == lowd_indices[i])
            assert len(dim_nn[0]) == 1, (
                f"Point {i}: The {highd_indices[i][pos_index]}-nearest neighbor "
                f"was not assigned to exactly one match (index) "
                f"in low-dimensional space."
            )

            p_i2 = dim_nn[0][0]
            sum_highd += (nhood_size - p_in) * np.abs(p_i2 - p_in)

        seq_diffs[i] = 0.5 * sum_highd + 0.5 * sum_lowd

    return seq_diffs


def compute_sequence_change(
    sequence_diff_start: ArrayLike,
    sequence_diff_mod: ArrayLike,
    allow_neg: bool = True,
) -> list[float]:
    sequence_diff_change = sequence_diff_mod - sequence_diff_start

    if allow_neg:
        return sequence_diff_change

    return [x if x > 0 else 0 for x in sequence_diff_change]


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
        return df.drop(columns=["jaccard", "coranking_matrix"])

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
    # remove columns with pairwise metrics
    return df.drop(columns=["jaccard", "coranking_matrix"])
