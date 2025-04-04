import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sb
from numpy.typing import ArrayLike
from pyDRMetrics.pyDRMetrics import DRMetrics  # type: ignore
from sklearn.isotonic import IsotonicRegression  # type: ignore
from sklearn.metrics import euclidean_distances, pairwise_distances  # type: ignore
from sklearn.neighbors import KDTree  # type: ignore

from embedding_obj import EmbeddingObj  # type: ignore


def compute_shepard_curve(
    original_similarities: ArrayLike, transformed_similarities: ArrayLike
) -> tuple[list[float], npt.NDArray[np.float32]]:
    # Sort values based on transformed similarities
    sorted_indices = np.argsort(transformed_similarities)
    sorted_transformed = transformed_similarities[sorted_indices]
    sorted_original = original_similarities[sorted_indices]

    # Apply isotonic regression to enforce monotonicity
    iso_reg = IsotonicRegression(increasing=True)
    shepard_curve = iso_reg.fit_transform(sorted_transformed, sorted_original)

    return sorted_transformed.tolist(), shepard_curve


def compute_kruskal_stress(
    original_similarities: ArrayLike, transformed_similarities: ArrayLike
) -> float:
    # TODO: Replace transformed_similarities in stress_numerator with
    #  "target distances", i. e. isometric regression values(?)
    stress_numerator = np.sum((original_similarities - transformed_similarities) ** 2)
    stress_denominator = np.sum(original_similarities**2)
    return np.sqrt(stress_numerator / stress_denominator)


def compute_distance_matrix(input_array: ArrayLike) -> npt.NDArray[np.float32]:
    distances = euclidean_distances(input_array, input_array)
    return distances[np.triu_indices(distances.shape[0], k=1)]


def plot_shepard_diagram(
    x_data: ArrayLike,
    y_data: ArrayLike,
    feature_name: str,
    show_stress: bool = True,
) -> None:
    df_data = pd.DataFrame(
        {
            "Transformed Similarity": x_data,
            "Original Similarity": y_data,
        }
    )

    # Plot the Shepard diagram using Seaborn
    sb.jointplot(
        x="Transformed Similarity",
        y="Original Similarity",
        data=df_data,
        kind="hist",
    )
    sb.lineplot(x=[0, 1], y=[0, 1], color="red", linestyle="--")

    if show_stress:
        shepard_x, shepard_y = compute_shepard_curve(
            df_data["Original Similarity"], df_data["Transformed Similarity"]
        )
        sb.lineplot(x=shepard_x, y=shepard_y, color="blue")

        stress = compute_kruskal_stress(
            df_data["Original Similarity"], df_data["Transformed Similarity"]
        )
        print(f"Kruskal Stress: {stress}")

    plt.suptitle(f"Shepard Diagram: Similarity for '{feature_name}'", y=1.02)
    plt.show()


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
    dmat = pairwise_distances(highd_data)
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


# -----------------------------------------------------------------------------


def _nhood_search_unlimited(data: ArrayLike) -> npt.NDArray[np.float32]:
    dmat = pairwise_distances(data)
    indices_sorted = np.argsort(dmat)
    # dists = dmat[indices_sorted]

    # return only the indices of the neighbors without distance from a point to itself
    indices_sorted_cleaned = np.empty((0, indices_sorted.shape[1] - 1), dtype=np.int32)

    for i in range(indices_sorted.shape[0]):
        indices_sorted_cleaned = np.vstack(
            [indices_sorted_cleaned, indices_sorted[i][indices_sorted[i] != i]]
        )

    return indices_sorted_cleaned


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


def pydrmetrics_plot_coranking_matrix(
    highdim_data: ArrayLike, lowdim_data: ArrayLike
) -> None:
    drm = DRMetrics(highdim_data, lowdim_data)
    drm.plot_coranking_matrix()


def metric_q_local(
    reference_points: ArrayLike,
    embedding_points: ArrayLike,
    metrics_obj: Optional[type[DRMetrics]] = None,
) -> tuple[float, type[DRMetrics]]:
    if metrics_obj is None:
        metrics_obj = DRMetrics(reference_points, embedding_points)

    return metrics_obj.Qlocal, metrics_obj


def metric_trustworthiness(
    reference_points: ArrayLike,
    embedding_points: ArrayLike,
    metrics_obj: Optional[type[DRMetrics]] = None,
) -> tuple[float, type[DRMetrics]]:
    if metrics_obj is None:
        metrics_obj = DRMetrics(reference_points, embedding_points)

    return metrics_obj.AUC_T, metrics_obj


def metric_continuity(
    reference_points: ArrayLike,
    embedding_points: ArrayLike,
    metrics_obj: Optional[type[DRMetrics]] = None,
) -> tuple[float, type[DRMetrics]]:
    if metrics_obj is None:
        metrics_obj = DRMetrics(reference_points, embedding_points)

    return metrics_obj.AUC_T, metrics_obj


# TODO: Implement metric_norm_stress
def metric_norm_stress(
    reference_points: ArrayLike,
    embedding_points: ArrayLike,
    metrics_obj: Optional[type[DRMetrics]] = None,
) -> tuple[float, type[DRMetrics]]:
    if metrics_obj is None:
        metrics_obj = DRMetrics(reference_points, embedding_points)

    return 0, metrics_obj


def metric_spearman(
    reference_points: ArrayLike,
    embedding_points: ArrayLike,
    metrics_obj: Optional[type[DRMetrics]] = None,
) -> tuple[float, type[DRMetrics]]:
    if metrics_obj is None:
        metrics_obj = DRMetrics(reference_points, embedding_points)

    return metrics_obj.Vrs, metrics_obj


def metric_total_score(
    reference_points: ArrayLike,
    embedding_points: ArrayLike,
    weights: Optional[ArrayLike] = None,
    metrics_obj: Optional[type[DRMetrics]] = None,
) -> float:
    n_metrics = 5

    if weights is None:
        weights = [1, 1, 1, 1, 1]

    assert len(weights) == n_metrics, (
        "The number of weights must match the number of metrics."
    )

    m_ql, metrics_obj_computed = metric_q_local(
        reference_points, embedding_points, metrics_obj
    )

    if metrics_obj is None:
        metrics_obj = metrics_obj_computed

    m_t = metric_trustworthiness(reference_points, embedding_points, metrics_obj)[0]
    m_c = metric_continuity(reference_points, embedding_points, metrics_obj)
    m_ns = 1 - metric_norm_stress(reference_points, embedding_points, metrics_obj)[0]
    m_s = metric_spearman(reference_points, embedding_points, metrics_obj)[0]

    metrics_list = [m_ql, m_t, m_c, m_ns, m_s]
    m_total = 0
    for i in range(n_metrics):
        m_total += weights[i] * metrics_list[i]

    return m_total / n_metrics


def metric_total_score_emb(
    embedding_obj: type[EmbeddingObj],
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
    m_ns = 1 - embedding_obj.m_normalized_stress
    m_s = embedding_obj.m_spearman

    metrics_list = [m_ql, m_t, m_c, m_ns, m_s]
    m_total = 0
    for i in range(n_metrics):
        m_total += weights[i] * metrics_list[i]

    return m_total / n_metrics


def add_local_metrics(
    highdim_data: npt.NDArray[np.float32], embeddings: list[EmbeddingObj]
) -> list[EmbeddingObj]:
    for emb in embeddings:
        print("------------------------------------------------------------")
        print("Computing metrics for embedding with marker: ", emb.marker)
        start_time = time.time()

        emb.m_jaccard = compute_jaccard_distances(
            highdim_data, emb.embedding, nhood_size=7
        )
        end_time = time.time()
        print("Computation time: ", end_time - start_time)
        print("------------------------------------------------------------")

    return embeddings


def add_global_metrics(
    highdim_data: npt.NDArray[np.float32], embeddings: list[EmbeddingObj]
) -> list[EmbeddingObj]:
    for emb in embeddings:
        print("------------------------------------------------------------")
        print("Computing metrics for embedding with marker: ", emb.marker)
        start_time = time.time()

        emb.m_q_local, metrics_obj = metric_q_local(highdim_data, emb.embedding)
        emb.m_trustworthiness, _ = metric_trustworthiness(
            highdim_data, emb.embedding, metrics_obj=metrics_obj
        )
        emb.m_continuity, _ = metric_continuity(
            highdim_data, emb.embedding, metrics_obj=metrics_obj
        )
        emb.m_spearman, _ = metric_spearman(
            highdim_data, emb.embedding, metrics_obj=metrics_obj
        )
        emb.m_normalized_stress, _ = metric_norm_stress(
            highdim_data, emb.embedding, metrics_obj=metrics_obj
        )

        emb.m_total_score = metric_total_score_emb(emb)

        end_time = time.time()
        print("Computation time: ", end_time - start_time)
        print("------------------------------------------------------------")
    return embeddings


def metrics_report(embeddings: list[EmbeddingObj]) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "marker",
            "m_total_score",
            "metric_jaccard (size)",
            "m_q_local",
            "m_trustworthiness",
            "m_continuity",
            "m_spearman",
            "m_normalized_stress",
        ]
    )

    for i, emb in enumerate(embeddings):
        df.loc[i] = [emb.marker,
                     emb.m_total_score,
                     emb.m_jaccard.size,
                     emb.m_q_local,
                     emb.m_trustworthiness,
                     emb.m_continuity,
                     emb.m_spearman,
                     emb.m_normalized_stress]

    return df