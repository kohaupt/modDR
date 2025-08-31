"""Evaluation metrics for dimensionality reduction quality assessment.

This module provides evaluation functions for assessing the quality
of dimensionality reduction embeddings. It includes both ranking-based and
distance-based metrics to evaluate how well the low-dimensional representation
preserves the structure and relationships from the high-dimensional space.

The implementation includes adaptations from the pyDRMetrics package by
Yinsheng Zhang, with optimizations for runtime efficiency and integration
with the moddr pipeline.

References:
    - Zhang, Y. pyDRMetrics: A Python package for dimensionality reduction
      quality metrics. https://github.com/zhangys11/pyDRMetrics
    - Lee, J. A., et al. "Type 1 and 2 mixtures of Kullback–Leibler divergences
      as cost functions in dimensionality reduction based on similarity
      preservation." Neurocomputing 112 (2013): 92-108.
"""

import copy
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.distance import squareform

from .. import processing
from ..embedding_state import EmbeddingState


def compute_kruskal_stress(
    dists_highdim: npt.NDArray[np.float32], dists_lowdim: npt.NDArray[np.float32]
) -> float:
    """Compute the Kruskal stress metric between high-dimensional and low-dimensional distances.

    Kruskal stress measures how well the low-dimensional embedding preserves the
    pairwise distances from the high-dimensional space. The stress is normalized
    by the sum of squared high-dimensional distances.

    Args:
        dists_highdim (npt.NDArray[np.float32]): Array of pairwise distances in high-dimensional space.
            Can be either a square distance matrix or condensed distance vector.
        dists_lowdim (npt.NDArray[np.float32]): Array of pairwise distances in low-dimensional space.
            Can be either a square distance matrix or condensed distance vector.
            Must have the same shape as dists_highdim.

    Returns:
        float: The normalized Kruskal stress value (0 = perfect preservation,
        higher = worse).

    Raises:
        ValueError: If the input arrays have different shapes or have non-zero off-diagonal elements.
    """
    if dists_highdim.shape != dists_lowdim.shape:
        raise ValueError(
            f"Shape mismatch (dists_highdim.shape={dists_highdim.shape}, dists_lowdim.shape={dists_lowdim.shape}): "  # noqa: E501
            f"Both arrays must have the same shape."
        )

    # convert to vector form if necessary as distances must not be used more than once
    if dists_highdim.ndim == 2:
        if not np.allclose(np.diag(dists_highdim), 0):
            raise ValueError(
                "Input must be a square distance matrix with zeros on the diagonal."
            )

        dists_highdim = squareform(dists_highdim)

    if dists_lowdim.ndim == 2:
        if not np.allclose(np.diag(dists_lowdim), 0):
            raise ValueError(
                "Input must be a square distance matrix with zeros on the diagonal."
            )
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
    """Compute the average Kruskal stress for each community in a partition.

    This function computes the Kruskal stress separately for each community
    defined in the partition and returns the average stress across all communities
    with at least 2 nodes.

    Args:
        dists_highdim (npt.NDArray[np.float32]): Array of pairwise distances in high-dimensional space.
            Must be a square distance matrix or condensed distance vector.
        dists_lowdim (npt.NDArray[np.float32]): Array of pairwise distances in low-dimensional space.
            Must be a square distance matrix or condensed distance vector.
            Must have the same shape as dists_highdim.
        partition (dict[int, npt.NDArray[np.int32]]): Dictionary mapping community IDs to arrays of node indices
            belonging to each community.

    Returns:
        float: The average Kruskal stress across all communities with at least 2 nodes.

    Raises:
        ValueError: If the input arrays have different shapes or have non-zero off-diagonal elements.
    """
    if dists_highdim.shape != dists_lowdim.shape:
        raise ValueError(
            f"Shape mismatch (dists_highdim.shape={dists_highdim.shape}, dists_lowdim.shape={dists_lowdim.shape}): "  # noqa: E501
            f"Both arrays must have the same shape."
        )

    # convert to square form if necessary, as extraction of distances requires 2D arrays
    if dists_highdim.ndim != 2:
        dists_highdim = squareform(dists_highdim)
    else:
        if not np.allclose(np.diag(dists_highdim), 0):
            raise ValueError(
                "Input must be a square distance matrix with zeros on the diagonal."
            )
    if dists_lowdim.ndim != 2:
        dists_lowdim = squareform(dists_lowdim)
    else:
        if not np.allclose(np.diag(dists_lowdim), 0):
            raise ValueError(
                "Input must be a square distance matrix with zeros on the diagonal."
            )

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
        # dists_highdim_com = squareform(dists_highdim_com)
        # dists_lowdim_com = squareform(dists_lowdim_com)

        kruskal_com += compute_kruskal_stress(dists_highdim_com, dists_lowdim_com)
        community_count += 1

    # normalize by number of communities
    return kruskal_com / community_count


def compute_coranking_matrix(
    r1: npt.NDArray[np.int32], r2: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """Compute the co-ranking matrix between two ranking arrays.

    This implementation is adapted from the pyDRMetrics package by Yinsheng Zhang [1].
    Modifications include computing AUC of metrics only when needed for runtime
    efficiency, rather than automatically during co-ranking matrix computation.

    Args:
        r1 (npt.NDArray[np.int32]): Ranking matrix for the original high-dimensional space.
            Each row contains the ranks of distances for one data point.
        r2 (npt.NDArray[np.int32]): Ranking matrix for the reduced low-dimensional space.
            Each row contains the ranks of distances for one data point.
            Must have the same shape as r1.

    Returns:
        npt.NDArray[np.int32]: The co-ranking matrix as a 2D array where entry (i,j) represents
        the count of points that are among the i nearest neighbors in the
        original space and j nearest neighbors in the reduced space.

    Raises:
        ValueError: If the input ranking arrays have different shapes.

    References:
        [1] Zhang, Y. pyDRMetrics: A Python package for dimensionality reduction
        quality metrics. https://github.com/zhangys11/pyDRMetrics
        Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
        https://creativecommons.org/licenses/by/4.0/
    """
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
    """Compute the trustworthiness metric from a co-ranking matrix.

    Trustworthiness measures how well the k-nearest neighbors in the
    low-dimensional space correspond to the k-nearest neighbors in the
    high-dimensional space. A value of 1 indicates perfect trustworthiness.

    This implementation is adapted from the pyDRMetrics package by Yinsheng Zhang [1].

    Args:
        cr_matrix (npt.NDArray[np.float32]): The co-ranking matrix computed from ranking data.
        k (int): The neighborhood size parameter. Must be in range [1, n-1] where
            n is the size of the co-ranking matrix.

    Returns:
        float: The trustworthiness value between 0 and 1, where 1 is perfect.

    Raises:
        ValueError: If k is not in the valid range [1, n-1].

    References:
        [1] Zhang, Y. pyDRMetrics: A Python package for dimensionality reduction
        quality metrics. https://github.com/zhangys11/pyDRMetrics
        Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
        https://creativecommons.org/licenses/by/4.0/
    """
    n = len(cr_matrix)
    if k < 1 or k >= n:
        raise ValueError(f"Invalid k: {k}. It must be in the range [1, {n - 1}].")

    qs = cr_matrix[k:, :k]
    w = np.arange(qs.shape[0]).reshape(-1, 1)
    return 1 - np.sum(qs * w) / k / n / (n - 1 - k)


def compute_continuity(cr_matrix: npt.NDArray[np.float32], k: int) -> float:
    """Compute the continuity metric from a co-ranking matrix.

    Continuity measures how well the k-nearest neighbors in the
    high-dimensional space correspond to the k-nearest neighbors in the
    low-dimensional space. A value of 1 indicates perfect continuity.

    This implementation is adapted from the pyDRMetrics package by Yinsheng Zhang [1].

    Args:
        cr_matrix (npt.NDArray[np.float32]): The co-ranking matrix computed from ranking data.
        k (int): The neighborhood size parameter. Must be in range [1, n-1] where
            n is the size of the co-ranking matrix.

    Returns:
        float: The continuity value between 0 and 1, where 1 is perfect.

    Raises:
        ValueError: If k is not in the valid range [1, n-1].

    References:
        [1] Zhang, Y. pyDRMetrics: A Python package for dimensionality reduction
        quality metrics. https://github.com/zhangys11/pyDRMetrics
        Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
        https://creativecommons.org/licenses/by/4.0/
    """
    n = len(cr_matrix)
    if k < 1 or k >= n:
        raise ValueError(f"Invalid k: {k}. It must be in the range [1, {n - 1}].")

    qs = cr_matrix[:k, k:]
    w = np.arange(qs.shape[1]).reshape(1, -1)
    return 1 - np.sum(qs * w) / k / n / (n - 1 - k)


def compute_rnx(qnn: npt.NDArray[np.float32], k: int) -> float:
    """Compute the RNX (R_NX) metric from QNN values (see [1]).

    RNX (Rank-based relative neighborhood preservation) measures the degree
    to which the neighborhood structure is preserved in the dimensionality
    reduction. A value of 1 indicates perfect preservation.

    Args:
        qnn (npt.NDArray[np.float32]): Array of QNN (Quality of Nearest Neighbors) values computed
            from the co-ranking matrix.
        k (int): The neighborhood size parameter. Must be in range [1, n-1] where
            n is the length of the qnn array.

    Returns:
        float: The RNX value, where 1 indicates perfect neighborhood preservation.

    Raises:
        ValueError: If k is not in the valid range [1, n-1].

    Reference:
        [1]: Lee, J. A., et al. "Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation." Neurocomputing 112 (2013): 92-108. https://doi.org/10.1016/j.neucom.2012.12.036
    """
    n = len(qnn)
    if k < 1 or k >= n:
        raise ValueError(f"Invalid k: {k}. It must be in the range [1, {n - 1}].")

    return (n * qnn[k - 1] - k) / (n - k)


def compute_rank_score(embedding: EmbeddingState) -> float:
    """Compute the overall rank score from ranking-based metrics.

    The rank score is the average of trustworthiness, continuity, and RNX metrics.
    All three metrics must have been computed and stored in the embedding's
    metrics dictionary.

    Args:
        embedding (EmbeddingState): An EmbeddingState object containing the computed metrics
            'trustworthiness', 'continuity', and 'rnx'.

    Returns:
        float: The average rank score as a float between 0 and 1, where 1 is perfect.

    Raises:
        ValueError: If any of the required metrics are missing from the
            embedding's metrics dictionary.
    """
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
    """Compute the overall distance score from distance-based metrics.

    The distance score combines similarity stress and community stress difference
    metrics to provide an overall assessment of distance preservation quality.
    The score is normalized to the range [0, 1] where 1 is perfect.

    Args:
        embedding (EmbeddingState): An EmbeddingState object containing the computed metrics
            'sim_stress' and 'sim_stress_com_diff'.

    Returns:
        float: The distance score as a float between 0 and 1, where 1 is perfect.

    Raises:
        ValueError: If any of the required metrics ('sim_stress',
            'sim_stress_com_diff') are missing from the embedding's
            metrics dictionary.
    """
    required_keys = ["sim_stress", "sim_stress_com_diff"]
    if any(embedding.metrics.get(x) is None for x in required_keys):
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
    """Compute the overall total score by combining rank and distance scores.

    The total score is a weighted average of the rank score and distance score,
    allowing for different emphasis on ranking-based vs distance-based metrics.

    Args:
        embedding (EmbeddingState): An EmbeddingState object containing the computed metrics
            'rank_score' and 'distance_score'.
        balance (float): Weight for the rank score in the range [0, 1]. The distance
            score gets weight (1 - balance). Default is 0.5 for equal weighting.

    Returns:
        float: The total score as a float between 0 and 1, where 1 is perfect.

    Raises:
        ValueError: If any of the required metrics ('rank_score',
            'distance_score') are missing from the embedding's metrics dictionary.
    """
    required_keys = ["rank_score", "distance_score"]
    if any(embedding.metrics.get(x) is None for x in required_keys):
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
    """Compute comprehensive evaluation metrics for dimensionality reduction embeddings.

    This function computes various quality metrics for dimensionality reduction
    including ranking-based metrics (trustworthiness, continuity, RNX) and
    distance-based metrics (stress measures). The metrics are computed for each
    embedding and stored in their respective metrics dictionaries.

    The co-ranking matrix computation and associated metrics are adapted from
    the pyDRMetrics package by Yinsheng Zhang [1], with modifications to compute
    AUC metrics only when needed for improved runtime efficiency.

    Args:
        highdim_df (pd.DataFrame): DataFrame containing the original high-dimensional data.
        embeddings (list[EmbeddingState]): List of EmbeddingState objects to evaluate.
        target_features (list[str]): List of feature names in highdim_df to use
            for similarity calculations.
        fixed_k (int | None): If specified, compute ranking metrics only for this neighborhood
            size. If None, compute AUC (average) over all possible k values.
        ranking_metrics (bool): Whether to compute ranking-based metrics (trustworthiness,
            continuity, RNX). Default is True.
        distance_metrics (bool): Whether to compute distance-based metrics (stress).
            Default is True.
        inplace (bool): Whether to modify embeddings in-place or create deep copies.
            Default is False (creates copies).
        verbose (bool): Whether to print progress information. Default is False.

    Returns:
        list[EmbeddingState]: List of EmbeddingState objects with computed metrics. If inplace=False,
        these are deep copies of the input embeddings.

    Notes:
        - Community-based stress metrics require partitions to be defined in
          the embeddings.
        - The reference embedding for community stress differences is taken
          from the first embedding in the list.

    References:
        Zhang, Y. pyDRMetrics: A Python package for dimensionality reduction
        quality metrics. https://github.com/zhangys11/pyDRMetrics
        Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
        https://creativecommons.org/licenses/by/4.0/
    """
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

        # compute pairwise distances + ranking matrix for lowdim data
        lowdim_df = pd.DataFrame(emb.embedding.values(), index=None)
        dists_lowdim = processing.compute_pairwise_dists(lowdim_df)
        rank_lowdim = [np.argsort(np.argsort(row)) for row in dists_lowdim]

        if ranking_metrics:
            cr_matrix = compute_coranking_matrix(
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
            if emb.partition is None:
                # if no communities are defined, use the whole embedding as one community  # noqa: E501
                emb.com_partition = {0: np.arange(len(emb.embedding))}

            dists_highdim_feat = processing.compute_pairwise_dists(
                highdim_df, sim_features=target_features, invert=False
            )

            emb.metrics["sim_stress_com"] = compute_kruskal_stress_partition(
                dists_highdim_feat, dists_lowdim, emb.partition
            )

            # compute differences in community-stress
            reference_com_stress = compute_kruskal_stress_partition(
                dists_highdim_feat, reference_lowdim, emb.partition
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
    """Create a DataFrame report from a list of EmbeddingState objects.

    This function extracts metadata and/or metrics from EmbeddingState objects
    and creates a structured DataFrame for analysis and comparison.

    Args:
        embeddings (list[EmbeddingState]): List of EmbeddingState objects to include in the report.
        metadata (bool): Whether to include metadata columns in the report.
            Default is True.
        metrics (bool): Whether to include metrics columns in the report.
            Default is True.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the requested information from all
        embeddings. The 'coranking_matrix' metric is excluded from the
        output as it is not suitable for tabular representation.

    Raises:
        ValueError: If both metadata and metrics are False, or if the
            embeddings list is empty.

    Notes:
        - Each row in the DataFrame represents one embedding.
        - The 'obj_id' column is always included to identify embeddings.
        - The co-ranking matrix is excluded from metrics output due to size.
    """
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
