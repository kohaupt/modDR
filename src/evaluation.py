import copy
import time
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.neighbors import KDTree  # type: ignore

import processing
from embedding_obj import EmbeddingObj  # type: ignore


def compute_kruskal_stress(
    highdim_dists: npt.NDArray[np.float32], lowdim_dists: npt.NDArray[np.float32]
) -> float:
    stress_numerator = np.sum((highdim_dists - lowdim_dists) ** 2)
    stress_denominator = np.sum(highdim_dists**2)
    return np.sqrt(stress_numerator / stress_denominator)


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
    dmat = processing.compute_pairwise_dists(highd_data, normalize=False)
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
    dmat = processing.compute_pairwise_dists(data, normalize=False)
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
    embeddings: list[EmbeddingObj],
    target_features: list[str],
    fixed_k: Optional[int] = None,
    ranking_metrics: bool = True,
    distance_metrics: bool = True,
    inplace: bool = False,
    verbose: bool = False,
) -> list[EmbeddingObj]:
    if not inplace:
        for i, emb in enumerate(embeddings):
            embeddings[i] = copy.deepcopy(emb)

    # compute pairwise distances + ranking matrix for highdim data
    D_highdim = processing.compute_pairwise_dists(highdim_df)
    D_highdim_rank = [np.argsort(np.argsort(row)) for row in D_highdim]

    for emb in embeddings:
        if verbose:
            print("------------------------------------------------------------")
            print(f"Computing global metrics for embedding: `{emb.title}'.")
            start_time = time.time()

        # compute pairwise distances + ranking matrix for lowdim data
        lowdim_df = pd.DataFrame(emb.embedding.values(), index=None)
        D_lowdim = processing.compute_pairwise_dists(lowdim_df)
        D_lowdim_rank = [np.argsort(np.argsort(row)) for row in D_lowdim]

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
            LCMC = np.zeros(m)  # Local Continuity Meta Criterion

            Q_cumsum = np.cumsum(np.cumsum(Q, axis=0), axis=1)
            diag_idxs = np.arange(m)
            QNN = Q_cumsum[diag_idxs, diag_idxs] / ((diag_idxs + 1) * m)
            LCMC = QNN - (diag_idxs + 1) / (m - 1)

            kmax = np.argmax(LCMC)
            Qlocal = np.sum(QNN[: kmax + 1]) / (kmax + 1)

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

                    C[k] = 1 - np.sum(Qs * W) / (k + 1) / m / (
                        m - 1 - k
                    )  # 1 - normalized hard-k-extrusions. upper-right region

                    R[k] = (m * QNN[k - 1] - k) / (m - k)

                emb.m_trustworthiness = np.mean(T)
                emb.m_continuity = np.mean(C)
                emb.m_rnx = np.mean(R)

            else:
                k = fixed_k
                Qs = Q[k:, :k]
                W = np.arange(Qs.shape[0]).reshape(
                    -1, 1
                )  # a column vector of weights. weight = rank error = actual_rank - k
                T = (
                    1 - np.sum(Qs * W) / (k + 1) / m / (m - 1 - k)
                )  # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)

                C = 1 - np.sum(Qs * W) / (k + 1) / m / (
                    m - 1 - k
                )  # 1 - normalized hard-k-extrusions. upper-right region
                # R = ((m - 1) * QNN[k-1] - k) / (m - 1 - k)
                R = (m * QNN[k - 1] - k) / (m - k)

                emb.m_trustworthiness = T
                emb.m_continuity = C
                emb.m_rnx = R

            global_rank_score_list = [
                emb.m_trustworthiness,
                emb.m_continuity,
                emb.m_rnx,
            ]
            global_rank_score_nominator = np.sum(global_rank_score_list)

            # avoid division by zero if all scores are zero (not computed)
            if global_rank_score_nominator > 0.0:
                # compute global rank score as the average of the non-zero (i. e. computed) scores
                global_rank_score_denominator = len(
                    [x for x in global_rank_score_list if x > 0.0]
                )
                emb.m_global_rank_score = (
                    global_rank_score_nominator / global_rank_score_denominator
                )

            emb.coranking_matrix = cr_matrix
            emb.m_q_local = Qlocal

        if distance_metrics:
            D_highdim_feat = processing.compute_pairwise_dists(
                highdim_df, sim_features=target_features
            )

            # kruskal_com = 0.0
            # for part in set(emb.com_partition.values()):
            #     part_keys = [k for k, v in emb.com_partition.items() if v == part]

            #     highdim_com_dists = np.take(D_highdim_feat, part_keys, axis=0)
            #     highdim_com_dists = np.take(highdim_com_dists, part_keys, axis=1)

            #     lowdim_com_dists = np.take(D_lowdim, part_keys, axis=0)
            #     lowdim_com_dists = np.take(lowdim_com_dists, part_keys, axis=1)

            #     highdim_com_dists = squareform(highdim_com_dists)
            #     lowdim_com_dists = squareform(lowdim_com_dists)

            #     kruskal_com += compute_kruskal_stress(
            #         highdim_com_dists, lowdim_com_dists
            #     )

            # # normalize by number of communities
            # kruskal_com = kruskal_com / len(set(emb.com_partition.values()))
            # emb.m_kruskal_stress_community = kruskal_com

            D_highdim_feat = squareform(D_highdim_feat)
            D_lowdim = squareform(D_lowdim)

            emb.m_kruskal_stress = compute_kruskal_stress(D_highdim_feat, D_lowdim)
            emb.m_shepard_spearman = metric_spearman(D_highdim_feat, D_lowdim)

        emb.m_total_score = metric_total_score(emb)

        if verbose:
            end_time = time.time()
            print(f"Computation finished after {end_time - start_time:.2f} seconds")
            print("------------------------------------------------------------")

    return embeddings


# -----------------------------------------------------------------------------


def compute_pairwise_metrics(
    highdim_data: npt.NDArray[np.float32],
    embeddings: list[EmbeddingObj],
    inplace: bool = False,
    verbose: bool = False,
) -> list[EmbeddingObj]:
    if not inplace:
        for i in enumerate(embeddings):
            embeddings[i] = copy.deepcopy(i)

    for emb in embeddings:
        if verbose:
            print("------------------------------------------------------------")
            print(f"Computing global metrics for embedding: `{emb.title}'.")
            start_time = time.time()

        emb.m_jaccard = compute_jaccard_distances(
            highdim_data,
            np.array(list(emb.embedding.values())),
            nhood_size=emb.k_neighbors,
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
    embedding_obj: EmbeddingObj,
    weights: Optional[ArrayLike] = None,
) -> float:
    n_metrics = 5

    if weights is None:
        weights = [1, 1, 1, 1, 1]

    assert len(weights) == n_metrics, (
        "The number of weights must match the number of metrics."
    )

    m_ql = embedding_obj.m_q_local
    m_t = embedding_obj.m_trustworthiness
    m_c = embedding_obj.m_continuity
    m_ks = 1 - embedding_obj.m_kruskal_stress
    m_s = embedding_obj.m_shepard_spearman

    metrics_list = [m_ql, m_t, m_c, m_ks, m_s]
    m_total = 0
    for i in range(n_metrics):
        m_total += weights[i] * metrics_list[i]

    return m_total / n_metrics


def metrics_report(embeddings: list[EmbeddingObj]) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "marker",
            "m_total_score",
            # "metric_jaccard (size)",
            "m_q_local",
            "m_trustworthiness",
            "m_continuity",
            "m_shepard_spearman",
            "m_kruskal_stress",
            "m_kruskal_stress_community",
            "m_rnx",
            "m_global_rank_score",
        ]
    )

    for i, emb in enumerate(embeddings):
        df.loc[i] = [
            emb.obj_id,
            emb.m_total_score,
            # emb.m_jaccard.size,
            emb.m_q_local,
            emb.m_trustworthiness,
            emb.m_continuity,
            emb.m_shepard_spearman,
            emb.m_kruskal_stress,
            emb.m_kruskal_stress_community,
            emb.m_rnx,
            emb.m_global_rank_score,
        ]

    return df
