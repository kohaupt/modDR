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
from sklearn.manifold import MDS
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

import evaluation
from embeddingstate import EmbeddingState


def run_pipeline(
    data: pd.DataFrame,
    sim_features: list[str],
    dr_method: str = "UMAP",
    dr_param_n_neigbors: int = 15,
    graph_method: str = "DR",
    community_resolutions: list[float] = None,
    community_resolution_amount: int = 3,
    layout_method: str = "MDS",
    boundary_neighbors: bool = False,
    layout_params: list[int] | None = None,
    compute_metrics: bool = True,
    verbose: bool = False,
) -> list[EmbeddingState]:
    """
    Run the modDR pipeline for dimensionality reduction and community detection.
    """

    if verbose:
        print("------------------------------------------------------------")
        print(
            "Start modDR pipeline with the following parameters:\n"
            f"Similarity Features: {sim_features if sim_features else 'all features'}\n"
            f"Dimensionality Reduction Method: {dr_method} with {dr_param_n_neigbors} neighbors\n"  # noqa: E501
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
        reference = dimensionality_reduction_umap(data, n_neighbors=dr_param_n_neigbors)
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

    # 4. Step: community detection & position refinement
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
        # compute the partition for the current resolution
        partition_embedding = community_detection_leiden(
            reference, resolution_parameter=resolution, verbose=verbose
        )

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
                    target_dists=pairwise_sims,
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
                layout_params = [0.2, 0.4, 0.6, 0.8, 1.0]
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
                target_dists=pairwise_sims,
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
        # 5.1 Step: compute global metrics
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


def dimensionality_reduction_param_search(
    data: pd.DataFrame, method: str = "UMAP"
) -> tuple[list[EmbeddingState], int]:
    """
    1. Step of the modDR pipeline: Dimensionality Reduction.
    """
    embeddings = []

    if method == "UMAP":
        # set parameters for UMAP, add custom n-neighbors value depending on data size
        params_n_neigbors_fixed = {10, 15, 20, 50, 100}
        params_n_neigbors_data = {data.shape[0] / 80, data.shape[0] / 40}

        params_n_neigbors = list(params_n_neigbors_fixed.union(params_n_neigbors_data))
        param_min_dist = 1
        random_state = 0

        for current_n_neigbors in params_n_neigbors:
            emb = dimensionality_reduction_umap(
                data,
                n_neighbors=current_n_neigbors,
                min_dist=param_min_dist,
                random_state=random_state,
                compute_metrics=False,
            )
            embeddings.append(emb)
    else:
        raise ValueError(
            f"Method '{method}' is not supported. Currently, only 'UMAP' is available."
        )

    embeddings = evaluation.compute_metrics(
        data, embeddings, [], distance_metrics=False
    )

    recommended_embedding_idx = 0
    for i in range(1, len(embeddings) - 1):
        if (
            embeddings[recommended_embedding_idx].m_global_rank_score
            > embeddings[i].m_global_rank_score
        ):
            recommended_embedding_idx = i

    return embeddings, recommended_embedding_idx


def dimensionality_reduction_umap(
    data: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 1.0,
    random_state: int = 0,
    compute_metrics: bool = False,
) -> EmbeddingState:
    """
    1. Step of the modDR pipeline: Dimensionality Reduction.
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
    """
    2. Step of the modDR pipeline: Feature Similarity Computation.
    """
    input_data = []

    if sim_features is not None and sim_features != []:
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
    """
    3. Step of the modDR pipeline: Graph Construction via Feature Similarity.
    """
    if embedding.graph is None:
        raise ValueError("Embedding object must have a similarity graph.")

    if verbose:
        print("------------------------------------------------------------")
        print(
            f"Set edge-weights as feature-similarities for embedding: `{embedding.title}'."
        )

    if not inplace:
        embedding = copy.deepcopy(embedding)

    edge_weights = []
    for u, v in embedding.graph.edges():
        embedding.graph[u][v]["weight"] = similarity_matrix[u][v]
        edge_weights.append(similarity_matrix[u][v])

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
    """
    4.1 Step of the modDR pipeline: Community Detection via Leiden.
    """

    if verbose:
        print("------------------------------------------------------------")
        print(
            f"Computing communities via Leiden detection for embedding {embedding.obj_id}: "
            f"`{embedding.title}' with resolution '{resolution_parameter}'."
        )
        start_time = time.time()

    if not inplace:
        embedding = copy.deepcopy(embedding)

    # use igraph for community detection, as networkx does not support a native implementation of the Leiden algorithm
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
    """
    Apply a balance factor to the embedding positions.
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

    embedding.metadata["layout_params"]["balance factor"] = balance_factor

    return embedding


def compute_modified_positions(
    embedding: EmbeddingState,
    layout_param: int | float,
    layout_method: str,
    layout_scale: int = 1,
    boundary_neighbors: bool = False,
    target_dists: npt.NDArray[np.float32] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> tuple[EmbeddingState, npt.NDArray[np.float32]]:
    """
    4.2 Step of the modDR pipeline: Position refinement via kamada-kawai layouting.
    """

    if not inplace:
        embedding = copy.deepcopy(embedding)

    if verbose:
        print("------------------------------------------------------------")
        print(f"Compute new positions for embedding: `{embedding.title}'.")
        start_time = time.time()

    # compute community graphs based on given partition from embedding
    partition_subgraphs, partition_centers, partition_boundary_neighbors = (
        compute_community_graphs(embedding, boundary_neighbors=boundary_neighbors)
    )
    embedding.partition_centers = partition_centers

    # safes modified positions w/o balance factor influence (i.e. balance factor=1)
    computed_positions = None

    # scale pairwise distances for layout methods which are based on target_dists
    if layout_method == "KK" or layout_method == "MDS":
        if target_dists is None:
            raise ValueError(
                "Pairwise distances must be provided for Kamada Kawai or MDS layouting."
            )

        embedding_df = pd.DataFrame(embedding.embedding.values(), index=None)
        embedding_dists = compute_pairwise_dists(
            embedding_df,
            apply_squareform=False,
        )

        target_scaling = compute_distance_scaling(
            embedding_dists, squareform(target_dists)
        )
        target_dists = target_dists * target_scaling

    if layout_method == "KK":
        embedding, computed_positions = compute_kamada_kawai_layout(
            embedding,
            partition_subgraphs,
            target_dists,
            balance_factor=layout_param,
            boundary_neighbors=partition_boundary_neighbors
            if boundary_neighbors
            else None,
            verbose=verbose,
        )

        embedding.title += f", KK (balance factor: {layout_param})"
        embedding.metadata["layout_method"] = "KK"
        embedding.metadata["layout_params"]["balance factor"] = layout_param

    elif layout_method == "MDS":
        embedding, computed_positions = compute_mds_layout(
            embedding,
            partition_subgraphs,
            target_dists,
            balance_factor=layout_param,
            boundary_neighbors=partition_boundary_neighbors
            if boundary_neighbors
            else None,
            verbose=verbose,
        )

        embedding.title += f", MDS (balance factor: {layout_param})"
        embedding.metadata["layout_method"] = "MDS"
        embedding.metadata["layout_params"]["balance factor"] = layout_param

    elif layout_method == "FR":
        embedding = compute_fruchterman_reingold_layout(
            embedding,
            partition_subgraphs,
            scale=layout_scale,
            iterations=layout_param,
            boundary_neighbors=partition_boundary_neighbors
            if boundary_neighbors
            else None,
            verbose=verbose,
        )

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


def compute_distance_scaling(
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
        print("WARNING: Highdim distances are all 0. Returning 1 as scaling factor.")
        return 1.0

    numerator = np.dot(dists_highdim, dists_lowdim)
    denominator = np.dot(dists_lowdim, dists_lowdim)
    return numerator / denominator


def compute_community_graphs(
    embedding: EmbeddingState, boundary_neighbors: bool = False
) -> tuple[dict[int, nx.Graph], dict[int, tuple[float, float]], dict[int, list[Any]]]:
    partition_subgraphs = {}
    partition_centers = {}
    partition_boundary_neighbors = {}

    for part, nodes in embedding.partition.items():
        # partition centers are based on median rather than min/max to avoid distortion from outliers
        subgraph_points_coords = np.array([embedding.embedding[i] for i in nodes])
        partition_centers[part] = np.median(subgraph_points_coords, axis=0)

        subgraph = embedding.graph.subgraph(
            [
                node
                for node, attrs_dict in embedding.graph.nodes(data=True)
                if attrs_dict["community"] == part
            ]
        ).copy()

        if boundary_neighbors:
            subgraph, part_boundary_neighbors = add_boundary_edges(
                embedding.graph, subgraph
            )
            partition_boundary_neighbors[part] = part_boundary_neighbors

        partition_subgraphs[part] = subgraph

    return partition_subgraphs, partition_centers, partition_boundary_neighbors


def compute_kamada_kawai_layout(
    embedding: EmbeddingState,
    partition_subgraphs: dict[int, nx.Graph],
    pairwise_dists: npt.NDArray[np.float32],
    balance_factor: float = 0.5,
    boundary_neighbors: dict[int, list[int]] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> tuple[EmbeddingState, dict[int, npt.NDArray[np.float32]]]:
    """
    4.3 Step of the modDR pipeline: Execute position-movement via Kamada Kawai-layouting.
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

        subdist = {
            u: {v: float(pairwise_dists[u][v]) for v in part_graph.neighbors(u)}
            for u in part_graph.nodes
        }

        new_post_dict = nx.kamada_kawai_layout(
            part_graph,
            dist=subdist,
            pos=subgraph_pos,
            center=embedding.partition_centers[part_key],
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

    return embedding


def compute_mds_layout(
    embedding: EmbeddingState,
    partition_subgraphs: dict[int, nx.Graph],
    pairwise_dists: npt.NDArray[np.float32],
    balance_factor: float = 0.5,
    boundary_neighbors: dict[int, list[int]] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> tuple[EmbeddingState, dict[int, npt.NDArray[np.float32]]]:
    """
    4.3 Step of the modDR pipeline: Execute position-movement via Kamada Kawai-layouting.
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
        new_pos += embedding.partition_centers[part_key]
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
    scale: int,
    iterations: int,
    boundary_neighbors: dict[int, list[int]] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> EmbeddingState:
    """
    4.3 Step of the modDR pipeline: Execute position-movement via Fruchterman-Reingold-layouting.
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
            center=embedding.partition_centers[part_key],
            scale=scale,
            k=1.0,
            seed=0,
        )

        if boundary_neighbors is not None:
            for boundary_node in boundary_neighbors[part_key]:
                subgraph_updated_pos.pop(boundary_node, None)

        embedding.embedding.update(subgraph_updated_pos)

    embedding.embedding = dict(sorted(embedding.embedding.items()))
    return embedding


def add_boundary_edges(
    graph: nx.Graph, subgraph: nx.Graph
) -> tuple[nx.Graph, list[Any]]:
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
