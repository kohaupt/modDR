import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from pyDRMetrics.pyDRMetrics import *
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.neighbors import KDTree


def compute_shepard_curve(original_similarities, transformed_similarities):
    """
    Compute the Shepard curve using monotonic regression.

    Parameters:
    original_similarities (array-like): The original dissimilarity matrix.
    transformed_similarities (array-like): The dissimilarity matrix in the transformed space.

    Returns:
    array-like: The computed Shepard curve (monotonic regression values).
    """
    # Sort values based on transformed similarities
    sorted_indices = np.argsort(transformed_similarities)
    sorted_transformed = transformed_similarities[sorted_indices]
    sorted_original = original_similarities[sorted_indices]

    # Apply isotonic regression to enforce monotonicity
    iso_reg = IsotonicRegression(increasing=True)
    shepard_curve = iso_reg.fit_transform(sorted_transformed, sorted_original)

    return sorted_transformed.tolist(), shepard_curve


def compute_kruskal_stress(original_similarities, transformed_similarities):
    """
    Compute the Kruskal stress for given similarity arrays.

    Parameters:
        original_similarities (array-like): The original dissimilarity matrix.
        transformed_similarities (array-like): The dissimilarity matrix in the transformed space.

    Returns:
        float: The Kruskal stress value.
    """
    # TODO: Replace transformed_similarities in stress_numerator with "target distances", i. e. isometric regression values(?)
    stress_numerator = np.sum((original_similarities - transformed_similarities) ** 2)
    stress_denominator = np.sum(original_similarities**2)
    return np.sqrt(stress_numerator / stress_denominator)


def compute_distance_matrix(input_array):
    """
    Compute the Euclidean distance matrix between all pairs of points in the input array.

    Parameters:
        input_array (array-like): A matrix where each row represents a data point.

    Returns:
        array: The upper triangular portion of the distance matrix, excluding the diagonal.
    """
    distances = euclidean_distances(input_array, input_array)
    return distances[np.triu_indices(distances.shape[0], k=1)]


def plot_shepard_diagram(x_data, y_data, feature_name, show_stress=True):
    """
    Plot a Shepard diagram comparing original and transformed similarities.

    Parameters:
        x_data (array-like): The transformed similarity values.
        y_data (array-like): The original similarity values.
        show_stress (bool, optional): If True, computes and displays the Kruskal stress
                                     value and Shepard curve. Default is True.
    """
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
def submatrix(dmat, indices_col, n_neighbors):
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


def _nhood_compare(indices_left, indices_right):
    """Compute Jaccard index of two neighborhoods"""
    result = np.empty(indices_left.shape[0])

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


def _nhood_search(umap_object, nhood_size):
    dmat = pairwise_distances(umap_object._raw_data)
    indices = np.argpartition(dmat, nhood_size)[:, :nhood_size]
    dmat_shortened = submatrix(dmat, indices, nhood_size)

    indices_sorted = np.argsort(dmat_shortened)
    indices = submatrix(indices, indices_sorted, nhood_size)
    dists = submatrix(dmat_shortened, indices_sorted, nhood_size)

    return indices, dists


def compute_jaccard_distances(umap_object, embedding_points, nhood_size=15):
    highd_indices, highd_dists = _nhood_search(umap_object, nhood_size)
    tree = KDTree(embedding_points)
    lowd_dists, lowd_indices = tree.query(embedding_points, k=nhood_size)
    accuracy = _nhood_compare(
        highd_indices.astype(np.int32), lowd_indices.astype(np.int32)
    )

    return accuracy


# -----------------------------------------------------------------------------


def _nhood_search_unlimited(umap_object, data):
    dmat = pairwise_distances(data)
    indices_sorted = np.argsort(dmat)
    # dists = dmat[indices_sorted]

    # return only the indices of the neighbors without distance from a point to itself
    return indices_sorted[:, 1:]


def compute_sequence_diff(reference_points, embedding_points, nhood_size=15):
    highd_indices = _nhood_search_unlimited(reference_points, reference_points)
    lowd_indices = _nhood_search_unlimited(reference_points, embedding_points)

    seq_diffs = np.zeros((embedding_points.shape[0]), dtype=np.float32)
    for i in range(embedding_points.shape[0]):
        sum_highd = 0.0
        sum_lowd = 0.0
        for pos_index in range(nhood_size):
            p_i2 = pos_index

            p_in = np.where(highd_indices[i] == lowd_indices[i][pos_index])[0]
            sum_lowd += (nhood_size - p_i2) * np.abs(p_i2 - p_in)

            p_i2 = np.where(lowd_indices[i] == highd_indices[i][pos_index])[0]
            p_in = pos_index
            sum_highd += (nhood_size - p_in) * np.abs(p_i2 - p_in)

        seq_diffs[i] = 0.5 * sum_highd + 0.5 * sum_lowd

    # tree = KDTree(embedding_points)
    # lowd_dists, lowd_indices = tree.query(embedding_points, k=nhood_size)
    # accuracy = _nhood_compare(
    #     highd_indices.astype(np.int32), lowd_indices.astype(np.int32)
    # )

    return seq_diffs


def compute_sequence_change(sequence_diff_start, sequence_diff_mod):
    sequence_diff_change = sequence_diff_mod - sequence_diff_start
    return [x if x > 0 else 0 for x in sequence_diff_change]


def pydrmetrics_report(highdim_data, lowdim_data):
    drm = DRMetrics(highdim_data, lowdim_data)
    drm.report()


def pydrmetrics_plot_coranking_matrix(highdim_data, lowdim_data):
    drm = DRMetrics(highdim_data, lowdim_data)
    drm.plot_coranking_matrix()
