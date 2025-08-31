"""Processing pipeline for modified dimensionality reduction (moddr).

This module provides the core processing functionality for the moddr package,
implementing a pipeline for modified dimensionality reduction with
community detection and position refinement.

Pipeline Stages:
1. Dimensionality Reduction: UMAP-based reduction to 2D space
2. Feature Similarity: Computation of pairwise similarities based on target features
3. Graph Construction: Creation of neighborhood graphs from DR results or KNN
4. Community Detection: Leiden algorithm for community identification
5. Position Refinement: Layout algorithms (MDS, Kamada-Kawai, Fruchterman-Reingold)
6. Metrics Computation: Quality assessment of the final embeddings

The module supports both individual function usage and complete pipeline
execution through the `run_pipeline` function.
"""

import copy
import time
import warnings
from typing import Any

import leidenalg as la
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import umap
from igraph import Graph
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

from .. import evaluation
from ..embedding_state import EmbeddingState


def run_pipeline(
    data: pd.DataFrame,
    sim_features: list[str],
    dr_method: str = "UMAP",
    dr_param_n_neighbors: int = 15,
    graph_method: str = "DR",
    community_resolutions: list[float] = None,
    community_resolution_amount: int = 3,
    layout_method: str = "MDS",
    boundary_neighbors: bool = False,
    layout_params: list[int] | None = None,
    compute_metrics: bool = True,
    verbose: bool = False,
) -> list[EmbeddingState]:
    """Run the complete moddr (modified Dimensionality Reduction) pipeline.

    This function orchestrates the entire pipeline including dimensionality reduction,
    feature similarity computation, graph construction, community detection, and
    position refinement using various layout algorithms.

    Args:
        data (pd.DataFrame): Input DataFrame containing the high-dimensional data.
        sim_features (list[str]): List of feature column names to use for similarity computation.
        dr_method (str): Dimensionality reduction method. Currently only "UMAP" is supported.
            Default is "UMAP".
        dr_param_n_neigbors (int): Number of neighbors parameter for the DR method.
            Default is 15.
        graph_method (str): Graph construction method. Either "DR" (use DR graph) or "KNN" (use KNN graph based on feature similarity).
            Default is "DR".
        community_resolutions (list[float] | None): List of resolution parameters for community detection.
            If None, automatic resolutions are computed based on the minimum and maximum edge weights. Default is None.
        community_resolution_amount (int): Number of resolution values to use if
            community_resolutions is None. Has no effect if community_resolutions is set. Default is 3.
        layout_method (str): Layout method for position refinement.
            Options: "MDS" (Multidimensional Scaling), "KK" (Kamada-Kawai), "FR" (Fruchterman-Reingold). Default is "MDS".
        boundary_neighbors (bool): Whether to include boundary edges in layout computation.
            Default is False.
        layout_params (list[int] | None): Parameter values for the layout method. Iterations for FR,
            balance factors for MDS/KK. If None, defaults are used ([1, 10, 100, 1000] for FR,
            [0.2, 0.4, 0.6, 0.8, 1.0] for KK/MDS). Default is None.
        compute_metrics (bool): Whether to compute evaluation metrics for embeddings.
            Default is True.
        verbose (bool): Whether to print detailed progress information. Default is False.

    Returns:
        list[EmbeddingState]: List of EmbeddingState objects containing the processed embeddings with
        different community resolutions and layout parameters.

    Raises:
        ValueError: If unsupported methods are specified or invalid parameters
            are provided.
    """
    if verbose:
        print("------------------------------------------------------------")
        print(
            "Start moddr pipeline with the following parameters:\n"
            f"Similarity Features: {sim_features if sim_features else 'all features'}\n"
            f"Dimensionality Reduction Method: {dr_method} with {dr_param_n_neighbors} neighbors\n"  # noqa: E501
            f"Graph Construction Method: {graph_method}\n"
            f"Community Detection Resolutions: {community_resolutions if community_resolutions else 'automatic'}\n"  # noqa: E501
            f"Layout Method: {layout_method}\n"
            f"Boundary Neighbors: {boundary_neighbors}\n"
            f"Layout Parameters: {layout_params if layout_params else 'default'}\n"
            f"Compute Metrics: {compute_metrics}\n"
        )
        start_time = time.time()

    # 1. Step: dimensionality reduction
    if dr_method == "UMAP":
        reference = dimensionality_reduction_umap(
            data, n_neighbors=dr_param_n_neighbors
        )
    else:
        raise ValueError(
            f"Method '{dr_method}' is not supported. Currently, only 'UMAP' is available."  # noqa: E501
        )

    # 2. Step: feature similarity computation
    # Scale features to [0, 1] range to avoid higher influence of certain features
    data_scaled = data.copy()
    scaler = MinMaxScaler()
    for col in sim_features:
        data_scaled[col] = scaler.fit_transform(data[[col]])

    # set labels as PCA components as marker of similarity
    sim_features_reduced = PCA(n_components=1).fit_transform(data_scaled[sim_features])
    reference.labels = {
        i: sim_features_reduced[i] for i in range(len(sim_features_reduced))
    }

    # use pairwise distances for kamada kawai and mds layouts
    if layout_method == "KK" or layout_method == "MDS":
        pairwise_sims = compute_pairwise_dists(
            data_scaled, invert=False, sim_features=sim_features
        )
    # use pairwise similarities (inversed distances) for fruchterman-reingold layout
    elif layout_method == "FR":
        pairwise_sims = compute_pairwise_dists(
            data_scaled, invert=True, normalize=True, sim_features=sim_features
        )
    else:
        raise ValueError(
            f"Method '{layout_method}' is not supported. Currently, only 'KK', 'MDS', and 'FR' are available."
        )

    # 3. Step: graph construction
    # for graph_method=DR, the graph is already set (e.g. by dimensionality_reduction_umap)
    if graph_method == "DR":
        pass
    elif graph_method == "KNN":
        reference.graph, _ = compute_knn_graph(data, sim_features=sim_features)
    else:
        raise ValueError(
            f"Method '{graph_method}' is not supported. Currently, only 'DR' and 'KNN' are available."
        )

    # 4. Step: community detection (compute resolution parameters)
    weights = list(nx.get_edge_attributes(reference.graph, "weight", 1).values())
    min_w, max_w = min(weights), max(weights)

    if community_resolutions is None:
        # compute equidistant community resolutions between min_w and max_w
        community_resolutions = np.linspace(
            start=min_w, stop=max_w, num=community_resolution_amount
        )

        # apply padding to avoid extreme resolutions
        range_w = max_w - min_w
        padding = range_w * 0.05

        community_resolutions[0] = min_w + padding
        community_resolutions[-1] = max_w - padding

        community_resolutions = np.round(community_resolutions, 2)

        if verbose:
            print(
                f"Using the following community resolutions (min: {min_w}, max: {max_w}): {community_resolutions}."
            )

    if min_w > min(community_resolutions) or max_w < max(community_resolutions):
        print(
            f"WARNING: The resolution parameter(s) may be outside the recommended range ({min_w}, {max_w}). The resulting communities may not be meaningful."
        )

    # set the community partition for the reference embedding to avoid errors
    nx.set_node_attributes(reference.graph, 0, "community")
    reference.obj_id = 0
    embeddings = [reference]

    # gets increased after each new created embedding
    id_counter = 1

    for resolution in community_resolutions:
        # 4. Step: community detection (compute actual partition)
        partition_embedding = community_detection_leiden(
            reference, resolution_parameter=resolution, verbose=verbose
        )

        # 5. Step: position refinement (compute modified positions)
        # for FR layout, iterate over iteration parameters in layout_params
        if layout_method == "FR":
            if layout_params is None:
                layout_params = [1, 10, 100, 1000]
            elif not all(isinstance(x, int) and x >= 0 for x in layout_params):
                raise ValueError(
                    "Iterations for the FR-algorithm must be positive integers."
                )

            for param in layout_params:
                # use partition_embedding as a starting point of each modification
                modified_embedding = copy.deepcopy(partition_embedding)
                modified_embedding.obj_id = id_counter

                modified_embedding, _ = compute_modified_positions(
                    modified_embedding,
                    targets=pairwise_sims,
                    layout_method=layout_method,
                    layout_param=param,
                    boundary_neighbors=boundary_neighbors,
                    verbose=verbose,
                )
                embeddings.append(modified_embedding)
                id_counter += 1

        # for MDS & KK-layout, iterate over balance factors in layout_params
        elif layout_method == "MDS" or layout_method == "KK":
            if layout_params is None:
                layout_params = [0.5]
            elif not all(0 <= x <= 1 for x in layout_params):
                raise ValueError("The balance factors must be between 0 and 1.")

            # use partition_embedding as a starting point
            modified_embedding = copy.deepcopy(partition_embedding)
            modified_embedding.obj_id = id_counter

            # saves full modified positions for future balance factors
            full_modified_positions = None

            # compute modification for first balance factor
            # also saves full modified positions (balance factor=1) for future use
            modified_embedding, full_modified_positions = compute_modified_positions(
                modified_embedding,
                targets=pairwise_sims,
                layout_method=layout_method,
                layout_param=layout_params[0],
                boundary_neighbors=boundary_neighbors,
                verbose=verbose,
            )
            embeddings.append(modified_embedding)
            id_counter += 1

            # for all other balance factors, use the precomputed positions
            for param in layout_params[1:]:
                # use partition_embedding as a starting point of each modification
                modified_embedding = copy.deepcopy(partition_embedding)
                modified_embedding.obj_id = id_counter

                modified_embedding = apply_balance_factor(
                    modified_embedding, full_modified_positions, param, verbose=verbose
                )

                # set metadata accordingly, as they aren't already set by compute_modified_positions()
                modified_embedding.title += (
                    f", {layout_method} (balance factor: {param})"
                )
                modified_embedding.metadata["layout_method"] = layout_method
                modified_embedding.metadata["layout_params"]["boundary_neighbors"] = (
                    boundary_neighbors
                )
                if boundary_neighbors:
                    modified_embedding.title += ", boundary edges added"

                embeddings.append(modified_embedding)
                id_counter += 1

        else:
            raise ValueError(
                f"Method '{layout_method}' is not supported. Currently, only 'FR', 'MDS' and 'KK' are available."
            )

    if compute_metrics:
        # 6. Step: compute metrics
        evaluation.compute_metrics(
            data,
            embeddings,
            sim_features,
            fixed_k=reference.metadata["k_neighbors"],
            inplace=True,
            verbose=verbose,
        )

    if verbose:
        end_time = time.time()
        print(f"Pipeline finished after {end_time - start_time:.2f} seconds.")
        print("------------------------------------------------------------")

    return embeddings


def dimensionality_reduction_umap(
    data: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 1.0,
    random_state: int = 0,
    compute_metrics: bool = False,
) -> EmbeddingState:
    """Perform dimensionality reduction using UMAP.

    Creates an EmbeddingState object with the resulting embedding and the UMAP-generated graph structure.

    Args:
        data (pd.DataFrame): Input DataFrame containing the high-dimensional data.
        n_neighbors (int): Number of nearest neighbors to consider for UMAP.
            Controls the balance between local and global structure. Default is 15.
        min_dist (float): Minimum distance parameter for UMAP. Controls how
            tightly UMAP packs points together. Default is 1.0.
        random_state (int): Random seed for reproducible results. Default is 0.
        compute_metrics (bool): Whether to compute evaluation metrics for the
            resulting embedding. Default is False.

    Returns:
        EmbeddingState: EmbeddingState object containing the 2D embedding coordinates,
        the UMAP-generated graph, and associated metadata.
    """
    warnings.filterwarnings(
        "ignore",
        message="n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.",
    )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(data)
    embedding_dict = {i: embedding[i] for i in range(data.shape[0])}

    umap_embedding = EmbeddingState(
        embedding=embedding_dict,
        graph=nx.Graph(reducer.graph_),
        title=f"UMAP (n_neigbors: {n_neighbors}, min_dist: {min_dist})",
    )

    umap_embedding.metadata["dr_method"] = "UMAP"
    umap_embedding.metadata["dr_params"] = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "random_state": random_state,
    }
    umap_embedding.metadata["k_neighbors"] = n_neighbors

    if not compute_metrics:
        return umap_embedding

    return evaluation.compute_metrics(
        data, [umap_embedding], [], distance_metrics=False
    )[0]


def compute_pairwise_dists(
    df: pd.DataFrame,
    apply_squareform: bool = True,
    invert: bool = False,
    normalize: bool = False,
    no_null: bool = False,
    sim_features: list[str] | None = None,
) -> npt.NDArray[np.float32]:
    """Compute pairwise Euclidean distances between data points.

    This function calculates pairwise distances between all data points,
    with various options for processing and transforming the distances.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data points.
        apply_squareform (bool): Whether to convert condensed distance matrix to
            square form. If False, returns condensed form. Default is True.
        invert (bool): Whether to invert the distances. Uses 1/d if normalize=False,
            or 1-d if normalize=True. Default is False.
        normalize (bool): Whether to normalize distances to [0, 1] range using
            MinMaxScaler. Default is False.
        no_null (bool): Whether to replace zero distances with a small value (1e-9)
            to avoid division by zero. Default is False.
        sim_features (list[str] | None): List of feature column names to use for distance
            computation. If None, uses all columns. Default is None.

    Returns:
        npt.NDArray[np.float32]: Array of pairwise distances as float32. Shape depends on
        apply_squareform parameter.

    Raises:
        ValueError: If the input DataFrame is empty or if the provided
            sim_features are not in the DataFrame.
    """
    input_data = []

    if sim_features is not None and sim_features != []:
        if not all(feature in df.columns for feature in sim_features):
            raise ValueError("Not all specified sim_features are in the DataFrame.")

        input_data = df[sim_features].to_numpy()
    else:
        input_data = df.to_numpy()

    distances = pdist(input_data, metric="euclidean")

    if normalize:
        distances = (
            MinMaxScaler((0, 1)).fit_transform(distances.reshape(-1, 1)).flatten()
        )

    if no_null:
        distances = np.where(distances == 0, 1e-9, distances)

    if invert and normalize:
        print(
            "INFO: Inverting distances via 1 - distances, as normalization is applied."
        )
        distances = 1 - distances

    if invert and not normalize:
        print(
            "INFO: Inverting distances via 1 / distances, as no normalization is applied."
        )
        distances = np.where(distances == 0, 1e-9, distances)
        distances = 1 / distances

    if apply_squareform:
        distances = squareform(distances)

    return distances.astype(np.float32)


def assign_graph_edge_weights(
    embedding: EmbeddingState,
    similarity_matrix: npt.NDArray[np.float32],
    inplace: bool = False,
    verbose: bool = False,
) -> EmbeddingState:
    """Assign edge weights to a graph based on a similarity matrix.

    This function sets the 'weight' attribute of each edge in the embedding's
    graph to the corresponding similarity value from the similarity matrix.

    Args:
        embedding (EmbeddingState): EmbeddingState object containing the graph to modify.
        similarity_matrix (npt.NDArray[np.float32]): Square matrix containing similarity values between
            nodes. Matrix should be indexed by node IDs.
        inplace (bool): Whether to modify the embedding in-place or create a copy.
            Default is False.
        verbose (bool): Whether to print progress information. Default is False.

    Returns:
        EmbeddingState: EmbeddingState object with updated edge weights. If inplace=False,
        returns a deep copy of the input embedding.

    Raises:
        ValueError: If the embedding object doesn't have a graph or if the
            similarity matrix is not suitable (e.g., not square or different
            shape than the graph).
    """
    if embedding.graph is None:
        raise ValueError("Embedding object must have a similarity graph.")

    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Similarity matrix must be square.")

    if similarity_matrix.shape[0] != embedding.graph.number_of_nodes():
        raise ValueError(
            "Similarity matrix must match the number of nodes in the graph."
        )

    if verbose:
        print("------------------------------------------------------------")
        print(
            f"Set edge-weights as feature-similarities for embedding: `{embedding.title}'."
        )

    if not inplace:
        embedding = copy.deepcopy(embedding)

    for u, v in embedding.graph.edges():
        embedding.graph[u][v]["weight"] = similarity_matrix[u][v]

    if verbose:
        print(
            f"Edge-weights set for {len(embedding.graph.edges())} edges in the graph."
        )
        print("------------------------------------------------------------")

    return embedding


def community_detection_leiden(
    embedding: EmbeddingState,
    resolution_parameter: float,
    inplace: bool = False,
    verbose: bool = False,
) -> EmbeddingState:
    """Perform community detection using the Leiden algorithm.

    This function applies the Leiden algorithm for community detection on the
    embedding's graph, using edge weights and a specified resolution parameter.
    The detected communities are stored in the embedding's partition attribute
    and as node attributes in the graph.

    Uses igraph for community detection, as networkx does not support a
    native implementation of the Leiden algorithm.

    Args:
        embedding (EmbeddingState): EmbeddingState object containing the graph for community detection.
        resolution_parameter (float): Resolution parameter controlling the granularity of
            communities. Higher values lead to smaller, more communities.
        inplace (bool): Whether to modify the embedding in-place or create a copy.
            Default is False.
        verbose (bool): Whether to print progress information. Default is False.

    Returns:
        EmbeddingState: EmbeddingState object with detected communities. If inplace=False,
        returns a deep copy of the input embedding.

    Raises:
        ValueError: If the embedding object doesn't have a graph or edge weights.
    """

    if embedding.graph is None:
        raise ValueError("Embedding object must have a similarity graph.")

    edge_weights = list(nx.get_edge_attributes(embedding.graph, "weight").values())
    if len(edge_weights) == 0:
        raise ValueError(f"Graph of embedding {embedding.obj_id} has no edge weights.")

    if verbose:
        print("------------------------------------------------------------")
        print(
            f"Computing communities via Leiden detection for embedding {embedding.obj_id}: "
            f"`{embedding.title}' with resolution '{resolution_parameter}'."
        )
        start_time = time.time()

    if not inplace:
        embedding = copy.deepcopy(embedding)

    graph_igraph = Graph.from_networkx(embedding.graph)
    partition = la.find_partition(
        graph_igraph,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution_parameter,
    )

    # build partition dictionary and set community attribute for each node in the graph
    partition_dict = {}
    for i, community in enumerate(partition):
        partition_dict[i] = community

        for node in community:
            embedding.graph.nodes[graph_igraph.vs[node]["_nx_name"]]["community"] = i

    # update embedding metadata
    embedding.partition = partition_dict
    embedding.title = embedding.title + f", Leiden (resolution: {resolution_parameter})"
    embedding.metadata["com_detection"] = "Leiden"
    embedding.metadata["com_detection_params"]["resolution"] = resolution_parameter

    if verbose:
        end_time = time.time()
        print(f"Computation finished after {end_time - start_time:.2f} seconds.")
        print(f"Found {len(partition)} communities.")
        print("------------------------------------------------------------")

    return embedding


def apply_balance_factor(
    embedding: EmbeddingState,
    modified_positions: npt.NDArray[np.float32],
    balance_factor: float,
    inplace: bool = False,
    verbose: bool = False,
) -> EmbeddingState:
    """Apply a balance factor to blend original and modified positions.

    This function creates a weighted combination of the original embedding
    positions and new modified positions using a balance factor.

    Args:
        embedding (EmbeddingState): EmbeddingState object containing the original positions.
        modified_positions (npt.NDArray[np.float32]): Array of new positions to blend with originals.
        balance_factor (float): Weight for the modified positions (0-1).
            0 = only original positions, 1 = only modified positions.
        inplace (bool): Whether to modify the embedding in-place or create a copy.
            Default is False.
        verbose (bool): Whether to print progress information. Default is False.

    Returns:
        EmbeddingState: EmbeddingState object with blended positions. If inplace=False,
        returns a deep copy of the input embedding.

    Raises:
        ValueError: If embedding has no positions, modified_positions is None,
            lengths don't match, or balance_factor is not in [0, 1].
    """
    if verbose:
        print(
            f"Applying balance factor {balance_factor} for embedding: `{embedding.title}'."
        )

    if embedding.embedding is None:
        raise ValueError("Embedding object must have an embedding.")

    if not inplace:
        embedding = copy.deepcopy(embedding)

    if modified_positions is None:
        raise ValueError("Modified positions must be provided.")

    if len(embedding.embedding) != len(modified_positions):
        raise ValueError(
            "Modified positions must have the same length as the embedding."
        )

    if not (0 <= balance_factor <= 1):
        raise ValueError("Balance factor must be between 0 and 1.")

    embedding.embedding.update(
        {
            key: (1 - balance_factor) * embedding.embedding[key]
            + modified_positions[key] * balance_factor
            for key in embedding.embedding
        }
    )

    embedding.metadata["layout_params"]["balance_factor"] = balance_factor

    return embedding


def compute_modified_positions(
    embedding: EmbeddingState,
    layout_param: int | float,
    layout_method: str,
    targets: npt.NDArray[np.float32],
    boundary_neighbors: bool = False,
    inplace: bool = False,
    verbose: bool = False,
) -> tuple[EmbeddingState, dict[int, npt.NDArray[np.float32]]]:
    """Compute new positions for an embedding using various layout algorithms.

    This function applies different layout algorithms (Kamada-Kawai, MDS, or
    Fruchterman-Reingold) to modify node positions within detected communities
    while taking into account the original structure.

    Args:
        embedding (EmbeddingState): EmbeddingState object with computed partition.
        layout_param (int | float): Algorithm-specific parameter. For FR: iterations,
            for KK/MDS: balance factor.
        layout_method (str): Layout algorithm to use ("KK", "MDS", or "FR").
        targets (npt.NDArray[np.float32]): Target distance matrix for KK/MDS algorithms.
            Target similarity matrix for FR algorithm.
        boundary_neighbors (bool): Whether to include boundary neighbors between communities.
            Default is False.
        inplace (bool): Whether to modify the embedding in-place. Default is False.
        verbose (bool): Whether to print progress information. Default is False.

    Returns:
        tuple[EmbeddingState, dict[int, npt.NDArray[np.float32]]]: Tuple containing:
        - Modified EmbeddingState object
        - Dictionary of modified positions as computed by the layout algorithm (w/o balance factor influence)

    Raises:
        ValueError: If unsupported layout method is specified or required
            parameters are missing or invalid.
    """
    if not inplace:
        embedding = copy.deepcopy(embedding)

    if targets.ndim == 2 and not np.allclose(np.diag(targets), 0):
        raise ValueError(
            "Input must be a square distance matrix with zeros on the diagonal."
        )

    if targets.ndim == 1:
        targets = squareform(targets)

    if verbose:
        print("------------------------------------------------------------")
        print(f"Compute new positions for embedding: `{embedding.title}'.")
        start_time = time.time()

    # compute community graphs based on given partition from embedding
    partition_subgraphs, partition_centers, partition_boundary_neighbors = (
        compute_community_graphs(embedding, boundary_neighbors=boundary_neighbors)
    )
    embedding.community_centers = partition_centers

    # safes modified positions w/o balance factor influence (i.e. balance factor=1)
    computed_positions = None

    # scale pairwise distances for layout methods which are based on targets
    if layout_method == "KK" or layout_method == "MDS":
        embedding_df = pd.DataFrame(embedding.embedding.values(), index=None)
        embedding_dists = compute_pairwise_dists(
            embedding_df,
            apply_squareform=True,
        )

        target_scaling = compute_distance_scaling(embedding_dists, targets)
        targets = targets * target_scaling

    if layout_method == "KK":
        embedding, computed_positions = compute_kamada_kawai_layout(
            embedding,
            partition_subgraphs,
            targets,
            balance_factor=layout_param,
            boundary_neighbors=partition_boundary_neighbors
            if boundary_neighbors
            else None,
            verbose=verbose,
        )

        embedding.title += f", KK (balance factor: {layout_param})"
        embedding.metadata["layout_method"] = "KK"
        embedding.metadata["layout_params"]["balance_factor"] = layout_param

    elif layout_method == "MDS":
        embedding, computed_positions = compute_mds_layout(
            embedding,
            partition_subgraphs,
            targets,
            balance_factor=layout_param,
            boundary_neighbors=partition_boundary_neighbors
            if boundary_neighbors
            else None,
            verbose=verbose,
        )

        embedding.title += f", MDS (balance factor: {layout_param})"
        embedding.metadata["layout_method"] = "MDS"
        embedding.metadata["layout_params"]["balance_factor"] = layout_param

    elif layout_method == "FR":
        embedding = compute_fruchterman_reingold_layout(
            embedding,
            partition_subgraphs,
            pairwise_sims=targets,
            iterations=layout_param,
            boundary_neighbors=partition_boundary_neighbors
            if boundary_neighbors
            else None,
            verbose=verbose,
        )
        # as fr doesn't use a balance factor,
        # computed_positions is equal to the modified positions
        computed_positions = embedding.embedding.copy()

        embedding.title += f", FR layouting (iterations: {layout_param})"
        embedding.metadata["layout_method"] = "FR"
        embedding.metadata["layout_params"]["iterations"] = layout_param

    else:
        raise ValueError(
            f"Method '{layout_method}' is not supported. Currently, only 'FR', 'MDS' and 'KK' are available."
        )

    if verbose:
        end_time = time.time()
        print(
            f"Computation of new positions finished after {end_time - start_time:.2f} seconds."
        )
        print("------------------------------------------------------------")

    embedding.metadata["layout_params"]["boundary_neighbors"] = boundary_neighbors
    if boundary_neighbors:
        embedding.title += ", boundary edges added"

    return embedding, computed_positions


def compute_kamada_kawai_layout(
    embedding: EmbeddingState,
    partition_subgraphs: dict[int, nx.Graph],
    pairwise_dists: npt.NDArray[np.float32],
    balance_factor: float = 0.5,
    boundary_neighbors: dict[int, list[int]] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> tuple[EmbeddingState, dict[int, npt.NDArray[np.float32]]]:
    """Apply Kamada-Kawai layout algorithm to community subgraphs.

    This function applies the Kamada-Kawai force-directed algorithm to each
    community separately, using target distances and a balance factor to
    blend original and new positions.

    Args:
        embedding (EmbeddingState): EmbeddingState object with community partition.
        partition_subgraphs (dict[int, nx.Graph]): Dictionary of community subgraphs.
        pairwise_dists (npt.NDArray[np.float32]): Target distance matrix for the layout algorithm.
        balance_factor (float): Weight for blending original and new positions (0-1).
            Default is 0.5.
        boundary_neighbors (dict[int, list[int]] | None): Dictionary of boundary neighbors for each community.
            Default is None.
        inplace (bool): Whether to modify the embedding in-place. Default is False.
        verbose (bool): Whether to print progress information. Default is False.

    Returns:
        tuple[EmbeddingState, npt.NDArray[np.float32]]: Tuple containing:
        - Modified EmbeddingState object
        - Array of modified positions as computed by the layout algorithm (w/o balance factor influence)
    """
    if not inplace:
        embedding = copy.deepcopy(embedding)

    if verbose:
        print("Start computation with Kamada Kawai-algorithm.")

    # saves the original positions of the original embedding
    original_pos_dict = embedding.embedding.copy()

    # saves the updated positions after Kamada Kawai layouting (returned for precomputed positions)
    updated_pos_dict = embedding.embedding.copy()

    # saves the final updated (scaled) positions after Kamada Kawai layouting
    updated_pos_dict_scaled = embedding.embedding.copy()

    for part_key, part_graph in partition_subgraphs.items():
        if len(part_graph.nodes) == 1:
            print(
                f"INFO: Skipping partition {part_key} with only {len(part_graph.nodes)} node(s) for Kamada Kawai layouting."
            )
            skipped_node_index = embedding.partition[part_key][0]
            updated_pos_dict_scaled[skipped_node_index] = original_pos_dict[
                skipped_node_index
            ]
            continue

        subgraph_pos = {
            node: original_pos_dict[node] for node in embedding.partition[part_key]
        }

        if boundary_neighbors is not None:
            subgraph_pos.update(
                {
                    boundary_node: original_pos_dict[boundary_node]
                    for boundary_node in boundary_neighbors[part_key]
                }
            )

        subdist = pairwise_dists[
            np.ix_(list(subgraph_pos.keys()), list(subgraph_pos.keys()))
        ]

        new_post_dict = nx.kamada_kawai_layout(
            part_graph,
            dist=subdist,
            pos=subgraph_pos,
            center=embedding.community_centers[part_key],
            scale=5,
        )

        if boundary_neighbors is not None:
            for boundary_node in boundary_neighbors[part_key]:
                new_post_dict.pop(boundary_node, None)

        updated_pos_dict.update(new_post_dict)

        updated_pos_dict_scaled.update(
            {
                key: (1 - balance_factor) * original_pos_dict[key]
                + new_post_dict[key] * balance_factor
                for key in new_post_dict
            }
        )

    updated_pos_dict_scaled = dict(sorted(updated_pos_dict_scaled.items()))
    updated_pos_dict = dict(sorted(updated_pos_dict.items()))

    embedding.embedding = updated_pos_dict_scaled

    return embedding, updated_pos_dict


def compute_mds_layout(
    embedding: EmbeddingState,
    partition_subgraphs: dict[int, nx.Graph],
    pairwise_dists: npt.NDArray[np.float32],
    balance_factor: float = 0.5,
    boundary_neighbors: dict[int, list[int]] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> tuple[EmbeddingState, dict[int, npt.NDArray[np.float32]]]:
    """Apply Multidimensional Scaling (MDS) layout to community subgraphs.

    This function applies MDS to each community separately, using target
    distances to compute new positions and a balance factor to blend
    with original positions.

    Args:
        embedding (EmbeddingState): EmbeddingState object with community partition.
        partition_subgraphs (dict[int, nx.Graph]): Dictionary of community subgraphs.
        pairwise_dists (npt.NDArray[np.float32]): Target distance matrix for the MDS algorithm.
        balance_factor (float): Weight for blending original and new positions (0-1).
            Default is 0.5.
        boundary_neighbors (dict[int, list[int]] | None): Dictionary of boundary neighbors for each community.
            Default is None.
        inplace (bool): Whether to modify the embedding in-place. Default is False.
        verbose (bool): Whether to print progress information. Default is False.

    Returns:
        tuple[EmbeddingState, npt.NDArray[np.float32]]: Tuple containing:
        - Modified EmbeddingState object
        - Array of modified positions as computed by the layout algorithm (w/o balance factor influence)
    """
    if not inplace:
        embedding = copy.deepcopy(embedding)

    if verbose:
        print("Start computation with MDS-algorithm.")

    # saves the original positions of the original embedding
    original_pos_dict = embedding.embedding.copy()

    # saves the updated positions after MDS layouting (returned for precomputed positions)
    updated_pos_dict = embedding.embedding.copy()

    # saves the final updated (scaled) positions after MDS layouting
    updated_pos_dict_scaled = embedding.embedding.copy()

    for part_key, part_graph in partition_subgraphs.items():
        if len(part_graph.nodes) == 1:
            print(
                f"INFO: Skipping partition {part_key} with only {len(part_graph.nodes)} node(s) for MDS layouting."
            )
            skipped_node_index = embedding.partition[part_key][0]
            updated_pos_dict_scaled[skipped_node_index] = original_pos_dict[
                skipped_node_index
            ]
            continue

        subgraph_pos = {
            node: original_pos_dict[node] for node in embedding.partition[part_key]
        }

        if boundary_neighbors is not None:
            subgraph_pos.update(
                {
                    boundary_node: original_pos_dict[boundary_node]
                    for boundary_node in boundary_neighbors[part_key]
                }
            )

        subdist = pairwise_dists[
            np.ix_(list(subgraph_pos.keys()), list(subgraph_pos.keys()))
        ]

        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            metric=True,
            normalized_stress="auto",
            max_iter=1000,
            eps=1e-9,
            n_init=1,
        )

        new_pos = mds.fit(
            subdist, init=np.array(list(subgraph_pos.values()))
        ).embedding_

        # shift new positions by partition center
        new_pos += embedding.community_centers[part_key]
        new_post_dict = {node: new_pos[i] for i, node in enumerate(subgraph_pos)}

        if boundary_neighbors is not None:
            for boundary_node in boundary_neighbors[part_key]:
                new_post_dict.pop(boundary_node, None)

        updated_pos_dict.update(new_post_dict)

        updated_pos_dict_scaled.update(
            {
                key: (1 - balance_factor) * original_pos_dict[key]
                + new_post_dict[key] * balance_factor
                for key in new_post_dict
            }
        )

    updated_pos_dict_scaled = dict(sorted(updated_pos_dict_scaled.items()))
    updated_pos_dict = dict(sorted(updated_pos_dict.items()))

    embedding.embedding = updated_pos_dict_scaled

    return embedding, updated_pos_dict


def compute_fruchterman_reingold_layout(
    embedding: EmbeddingState,
    partition_subgraphs: dict[int, nx.Graph],
    pairwise_sims: npt.NDArray[np.float32],
    iterations: int,
    boundary_neighbors: dict[int, list[int]] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> EmbeddingState:
    """Apply Fruchterman-Reingold layout to community subgraphs.

    This function applies the Fruchterman-Reingold force-directed algorithm
    to each community separately, using similarity values as edge weights.

    Args:
        embedding (EmbeddingState): EmbeddingState object with community partition.
        partition_subgraphs (dict[int, nx.Graph]): Dictionary of community subgraphs.
        pairwise_sims (npt.NDArray[np.float32]): Similarity matrix for edge weights.
        iterations (int): Number of iterations to run the algorithm.
        boundary_neighbors (dict[int, list[int]] | None): Dictionary of boundary neighbors for each community.
            Default is None.
        inplace (bool): Whether to modify the embedding in-place. Default is False.
        verbose (bool): Whether to print progress information. Default is False.

    Returns:
        EmbeddingState: Modified EmbeddingState object with new positions.
    """
    if not inplace:
        embedding = copy.deepcopy(embedding)

    if verbose:
        print("Start computation with Fruchterman-Reingold-algorithm.")

    # assign edge weights to pairwise feature similarities
    assign_graph_edge_weights(embedding, pairwise_sims, inplace=True, verbose=verbose)

    for part_key, part_graph in partition_subgraphs.items():
        subgraph_pos = {node: embedding.embedding[node] for node in part_graph.nodes}

        subgraph_updated_pos = nx.spring_layout(
            part_graph,
            pos=subgraph_pos,
            iterations=iterations,
            fixed=boundary_neighbors[part_key]
            if boundary_neighbors is not None and len(boundary_neighbors[part_key]) > 0
            else None,
            threshold=0.0001,
            weight="weight",
            center=embedding.community_centers[part_key],
            k=1.0,
            seed=0,
        )

        if boundary_neighbors is not None:
            for boundary_node in boundary_neighbors[part_key]:
                subgraph_updated_pos.pop(boundary_node, None)

        embedding.embedding.update(subgraph_updated_pos)

    embedding.embedding = dict(sorted(embedding.embedding.items()))
    return embedding


def compute_distance_scaling(
    dists_highdim: npt.NDArray[np.float32], dists_lowdim: npt.NDArray[np.float32]
) -> float:
    """Compute scaling factor to align high-dimensional and low-dimensional distances.

    This function calculates the optimal scaling factor to multiply low-dimensional
    distances to best match high-dimensional distances in a least-squares sense.

    Args:
        dists_highdim (npt.NDArray[np.float32]): Array of pairwise distances in high-dimensional space.
            Can be square matrix or condensed vector.
        dists_lowdim (npt.NDArray[np.float32]): Array of pairwise distances in low-dimensional space.
            Can be square matrix or condensed vector. Must match shape of dists_highdim.

    Returns:
        float: Scaling factor as a float. If high-dimensional distances are all zero,
        returns 1.0.

    Raises:
        ValueError: If the input arrays have different shapes.
    """
    if dists_highdim.shape != dists_lowdim.shape:
        raise ValueError(
            f"Shape mismatch (dists_highdim.shape={dists_highdim.shape}, dists_lowdim.shape={dists_lowdim.shape}): "  # noqa: E501
            f"Both arrays must have the same shape."
        )

    # convert to vector form if necessary as distances must not be used more than once
    if dists_highdim.ndim != 1:
        dists_highdim = squareform(dists_highdim)
        dists_lowdim = squareform(dists_lowdim)

    if not dists_highdim.any():
        print("WARNING: Highdim distances are all 0. Returning 1 as scaling factor.")
        return 1.0

    numerator = np.dot(dists_highdim, dists_lowdim)
    denominator = np.dot(dists_lowdim, dists_lowdim)
    return numerator / denominator


def compute_community_graphs(
    embedding: EmbeddingState, boundary_neighbors: bool = False
) -> tuple[dict[int, nx.Graph], dict[int, tuple[float, float]], dict[int, list[Any]]]:
    """Extract subgraphs and compute centers for each community.

    This function creates individual subgraphs for each community in the
    embedding's partition and computes their centers based on node positions.

    The center position is defined as the median of all node positions within
    the community instead of bounding box centers to avoid distortions from outliers.

    Args:
        embedding (EmbeddingState): EmbeddingState object with detected communities.
        boundary_neighbors (bool): Whether to include boundary nodes which are adjacent
            to the current community. Default is False.

    Returns:
        tuple[dict[int, nx.Graph], dict[int, tuple[float, float]], dict[int, list[Any]]]: Tuple containing:
        - Dictionary mapping community IDs to their subgraphs
        - Dictionary mapping community IDs to their center coordinates
        - Dictionary mapping community IDs to boundary neighbor lists
    """
    partition_subgraphs = {}
    community_centers = {}
    community_boundary_neighbors = {}

    for part, nodes in embedding.partition.items():
        # compute community centers
        subgraph_points_coords = np.array([embedding.embedding[i] for i in nodes])
        community_centers[part] = np.median(subgraph_points_coords, axis=0)

        # create subgraph for the community
        subgraph = embedding.graph.subgraph(
            [
                node
                for node, attrs_dict in embedding.graph.nodes(data=True)
                if node in nodes
            ]
        ).copy()

        if boundary_neighbors:
            subgraph, part_boundary_neighbors = compute_boundary_subgraph(
                embedding.graph, subgraph
            )
            community_boundary_neighbors[part] = part_boundary_neighbors

        partition_subgraphs[part] = subgraph

    return partition_subgraphs, community_centers, community_boundary_neighbors


def compute_boundary_subgraph(
    graph: nx.Graph, subgraph: nx.Graph
) -> tuple[nx.Graph, list[Any]]:
    """Compute the boundary subgraph by adding boundary nodes and edges.

    This function identifies neighbors of subgraph nodes that lie outside
    the subgraph and adds them as boundary nodes with their connecting edges.
    Boundary nodes are those that are adjacent to the subgraph but not part of it.

    Args:
        graph (nx.Graph): The full graph containing all nodes and edges.
        subgraph (nx.Graph): A subgraph extracted from the full graph.

    Returns:
        tuple[nx.Graph, list[Any]]: Tuple containing:
        - Extended subgraph with boundary nodes and edges added
        - List of boundary neighbor node IDs
    """
    # actual subgraph with boundary nodes + edges
    subgraph_boundary_neighbors = subgraph.copy()

    # collect boundary neighbors separately
    boundary_neighbors = set()

    for node in subgraph.nodes():
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            # check if neighbor is not part of the subgraph (i.e. is a boundary node)
            if neighbor not in subgraph.nodes():
                boundary_neighbors.add(neighbor)
                subgraph_boundary_neighbors.add_node(neighbor)
                subgraph_boundary_neighbors.add_edge(
                    node, neighbor, weight=graph[node][neighbor]["weight"]
                )

    return subgraph_boundary_neighbors, list(boundary_neighbors)


def compute_knn_graph(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    mode: str = "distance",
    sim_features: list[str] | None = None,
) -> tuple[nx.Graph, npt.NDArray[np.float32]]:
    """Compute a k-nearest neighbors graph from the input data.

    This function creates a k-nearest neighbors graph using the specified
    features and assigns edge weights based on pairwise distances.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data points.
        n_neighbors (int): Number of nearest neighbors to connect for each node.
            Default is 15.
        mode (str): Mode for the KNN graph construction.
            Default is "distance" (euclidean).
        sim_features (list[str] | None): List of feature column names to use for similarity
            computation. If None, uses all columns. Default is None.

    Returns:
        tuple[nx.Graph, npt.NDArray[np.float32]]: Tuple containing:
        - NetworkX graph with KNN connections and distance-based edge weights
        - Array of edge weights from the KNN graph
    """
    # compute knn-graph based on feature selection
    if sim_features is None or len(sim_features) == 0:
        knn_graph = kneighbors_graph(df, n_neighbors=n_neighbors, mode=mode)
    else:
        knn_graph = kneighbors_graph(
            df.loc[:, sim_features], n_neighbors=n_neighbors, mode=mode
        )

    # compute pairwise distances and apply to edge weights in knn_graph
    pairwise_dists = compute_pairwise_dists(df, sim_features=sim_features)
    knn_graph_nx = nx.Graph(knn_graph)

    edge_weights_knn = np.array([])
    for u, v in knn_graph_nx.edges():
        edge_weights_knn = np.append(edge_weights_knn, [pairwise_dists[u][v]])
        knn_graph_nx[u][v]["weight"] = pairwise_dists[u][v]

    return knn_graph_nx, edge_weights_knn
