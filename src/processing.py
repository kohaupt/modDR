import copy
import re
import time
from typing import Any

import leidenalg as la
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import umap
from community import community_louvain
from igraph import Graph
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

import evaluation
from embedding_obj import EmbeddingObj


def run_pipeline(
    data: pd.DataFrame,
    sim_features: list[str],
    dr_method: str = "UMAP",
    dr_param_n_neigbors: int = 15,
    graph_method: str = "DR",
    community_resolutions: list[float] = None,
    community_resolution_amount: int = 3,
    layout_method: str = "KK",
    boundary_neigbors: bool = False,
    layout_param: list[int] | None = None,
    compute_metrics: bool = True,
    verbose: bool = False,
) -> list[EmbeddingObj]:
    """
    Run the modDR pipeline for dimensionality reduction and community detection.
    """

    if verbose:
        print("------------------------------------------------------------")
        print("Start modDR pipeline with the following parameters:")
        print(f"Dimensionality Reduction Method: {dr_method}")
        print(f"Graph Construction Method: {graph_method}")
        print(f"Community Detection Resolutions: {community_resolutions}")
        print(f"Layout Method: {layout_method}")
        start_time = time.time()

    # 1. Step: dimensionality reduction
    if dr_method == "UMAP":
        reference = dimensionality_reduction_umap(
            data, n_neighbors=dr_param_n_neigbors, compute_metrics=False
        )
    else:
        raise ValueError(
            f"Method '{dr_method}' is not supported. Currently, only 'UMAP' is available."
        )

    # 2. Step: feature similarity computation
    data_scaled = data.copy()
    scaler = MinMaxScaler()

    for col in sim_features:
        data_scaled[col] = scaler.fit_transform(data[[col]])

    if layout_method == "KK" or layout_method == "MDS":
        feat_sim = compute_pairwise_dists(
            data_scaled, invert=False, sim_features=sim_features
        )

    if layout_method == "FR":
        feat_sim = compute_pairwise_dists(
            data_scaled, invert=True, normalize=True, sim_features=sim_features
        )

    # 3. Step: graph construction
    if graph_method == "KNN":
        reference.sim_graph, _ = compute_knn_graph(
            data, n_neighbors=3, sim_features=sim_features
        )

    # 4. Step: community detection & position refinement
    weights = [d["weight"] for _, _, d in reference.sim_graph.edges(data=True)]
    min_w, max_w = min(weights), max(weights)

    if community_resolutions is None:
        community_resolutions = np.linspace(
            start=min_w, stop=max_w, num=community_resolution_amount
        )

        range_w = max_w - min_w
        padding = range_w * 0.05

        community_resolutions[0] = min_w + padding
        community_resolutions[-1] = max_w - padding

        community_resolutions = np.round(community_resolutions, 2)

        if verbose:
            print(
                f"Using the following community resolutions: {community_resolutions} (min: {min_w}, max: {max_w})."
            )

    if min_w > min(community_resolutions) or max_w < max(community_resolutions):
        print(
            f"WARNING: The resolution parameter(s) may be outside the recommended range ({min_w}, {max_w}). The resulting communities may not be meaningful."
        )

    # set the community partition for the reference embedding to avoid errors
    reference.com_partition = {0: np.arange(len(reference.embedding))}
    nx.set_node_attributes(reference.sim_graph, 0, "community")

    reference.obj_id = 0
    embeddings = [reference]

    id_counter = 1
    for resolution in community_resolutions:
        community_embedding = com_detection_leiden(
            reference, resolution_parameter=resolution, inplace=False, verbose=verbose
        )

        assign_graph_edge_weights(
            community_embedding, feat_sim, inplace=True, verbose=verbose
        )

        if layout_method == "FR":
            if layout_param is None:
                layout_param = [10]

            for iteration in layout_param:
                modified_embedding = copy.deepcopy(community_embedding)
                modified_embedding.obj_id = id_counter
                id_counter += 1

                modified_embedding, _ = compute_modified_positions(
                    modified_embedding,
                    layout_method=layout_method,
                    layout_param=iteration,
                    boundary_neighbors=boundary_neigbors,
                    inplace=False,
                    verbose=verbose,
                )
                embeddings.append(modified_embedding)

        elif layout_method == "MDS":
            if layout_param is None:
                layout_param = [10]

            mds_positions = None
            for iteration in layout_param:
                modified_embedding = copy.deepcopy(community_embedding)
                modified_embedding.obj_id = id_counter
                id_counter += 1

                modified_embedding, mds_positions = compute_modified_positions(
                    modified_embedding,
                    target_dists=feat_sim,
                    layout_method=layout_method,
                    layout_param=iteration,
                    precomputed_positions=mds_positions,
                    boundary_neighbors=boundary_neigbors,
                    inplace=False,
                    verbose=verbose,
                )
                embeddings.append(modified_embedding)

        elif layout_method == "KK":
            modified_embedding, _ = compute_modified_positions(
                community_embedding,
                target_dists=feat_sim,
                layout_method=layout_method,
                layout_param=layout_param,
                boundary_neighbors=boundary_neigbors,
                inplace=False,
                verbose=verbose,
            )
            modified_embedding.obj_id = id_counter
            id_counter += 1

            embeddings.append(modified_embedding)

    if compute_metrics:
        # 5.1 Step: compute global metrics
        evaluation.compute_global_metrics(
            data,
            embeddings,
            sim_features,
            fixed_k=reference.metadata["k_neighbors"],
            inplace=True,
            verbose=verbose,
        )

        # 5.2 Step: compute pairwise metrics
        evaluation.compute_pairwise_metrics(
            data,
            embeddings,
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
) -> tuple[list[EmbeddingObj], int]:
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

    embeddings = evaluation.compute_global_metrics(
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
) -> EmbeddingObj:
    """
    1. Step of the modDR pipeline: Dimensionality Reduction.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(data)
    embedding_dict = {i: embedding[i] for i in range(data.shape[0])}

    umap_embedding = EmbeddingObj(
        embedding=embedding_dict,
        edge_weights=np.array([]),
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

    return evaluation.compute_global_metrics(
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
    embedding: EmbeddingObj,
    similarity_matrix: npt.NDArray[np.float32],
    inplace: bool = False,
    verbose: bool = False,
) -> EmbeddingObj:
    """
    3. Step of the modDR pipeline: Graph Construction via Feature Similarity.
    """
    if embedding.sim_graph is None:
        raise ValueError("Embedding object must have a similarity graph.")

    if verbose:
        print("------------------------------------------------------------")
        print(
            f"Set edge-weights as feature-similarities for embedding: `{embedding.title}'."
        )

    if not inplace:
        embedding = copy.deepcopy(embedding)

    edge_weights = []
    for u, v in embedding.sim_graph.edges():
        embedding.sim_graph[u][v]["weight"] = similarity_matrix[u][v]
        edge_weights.append(similarity_matrix[u][v])

    embedding.edge_weights = np.array(edge_weights, dtype=np.float32)

    if verbose:
        print(
            f"Edge-weights set for {len(embedding.sim_graph.edges())} edges in the graph."
        )
        print("------------------------------------------------------------")

    return embedding


def com_detection_leiden(
    embedding: EmbeddingObj,
    resolution_parameter: float = 0.001,
    inplace: bool = False,
    verbose: bool = False,
) -> EmbeddingObj:
    """
    4.1 Step of the modDR pipeline: Community Detection via Leiden.
    """

    if verbose:
        print("------------------------------------------------------------")
        print(
            f"Computing communities via Leiden detection for embedding: `{embedding.title}' with resolution '{resolution_parameter}'."
        )
        start_time = time.time()

    if not inplace:
        embedding = copy.deepcopy(embedding)

    g_ig = Graph.from_networkx(embedding.sim_graph)

    partition = la.find_partition(
        g_ig,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution_parameter,
    )

    comm_dict = {}

    for i, community in enumerate(partition):
        comm_dict[i] = community

        for node in community:
            embedding.sim_graph.nodes[g_ig.vs[node]["_nx_name"]]["community"] = i

    embedding.com_partition = comm_dict
    embedding.title = embedding.title + f", Leiden (resolution: {resolution_parameter})"
    embedding.metadata["com_detection"] = "Leiden"
    embedding.metadata["com_detection_params"]["resolution"] = resolution_parameter

    if verbose:
        end_time = time.time()
        print(f"Computation finished after {end_time - start_time:.2f} seconds.")
        print(f"Found {len(partition)} communities.")
        print("------------------------------------------------------------")

    return embedding


def compute_modified_positions(
    embedding: EmbeddingObj,
    layout_param: int,
    layout_method: str = "KK",
    layout_scale: int = 1,
    boundary_neighbors: bool = False,
    target_dists: npt.NDArray[np.float32] | None = None,
    precomputed_positions: dict[int, npt.NDArray[np.float32]] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> tuple[EmbeddingObj, npt.NDArray[np.float32]]:
    """
    4.2 Step of the modDR pipeline: Position refinement via kamada-kawai layouting.
    """

    if not inplace:
        embedding = copy.deepcopy(embedding)

    if verbose:
        print("------------------------------------------------------------")
        print(f"Compute new positions for embedding: `{embedding.title}'.")
        start_time = time.time()

    partition_subgraphs, partition_centers, partition_boundary_neighbors = (
        compute_community_graphs(embedding, boundary_neigbors=boundary_neighbors)
    )

    embedding.partition_centers = partition_centers
    computed_positions = None

    if layout_method == "KK":
        if target_dists is None:
            raise ValueError(
                "Pairwise distances must be provided for Kamada Kawai layouting."
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

        compute_kamada_kawai_layout(
            embedding,
            partition_subgraphs,
            target_dists,
            scale=5,
            boundary_neighbors=partition_boundary_neighbors
            if boundary_neighbors
            else None,
            inplace=True,
            verbose=verbose,
        )

        embedding.title += ", KK layouting"
        embedding.metadata["layout_method"] = "KK"

    elif layout_method == "MDS":
        if target_dists is None:
            raise ValueError("Pairwise distances must be provided for MDS layouting.")

        embedding_df = pd.DataFrame(embedding.embedding.values(), index=None)
        embedding_dists = compute_pairwise_dists(
            embedding_df,
            apply_squareform=False,
        )

        target_scaling = compute_distance_scaling(
            embedding_dists, squareform(target_dists)
        )
        target_dists = target_dists * target_scaling

        _, computed_positions = compute_mds_layout(
            embedding,
            partition_subgraphs,
            target_dists,
            balance_factor=layout_param,
            precomputed=precomputed_positions,
            boundary_neighbors=partition_boundary_neighbors
            if boundary_neighbors
            else None,
            inplace=True,
            verbose=verbose,
        )

        if precomputed_positions is not None:
            pattern = r"MDS \(balance factor: [^)]+\)"
            re.sub(pattern, f"MDS (balance factor: {layout_param})", embedding.title)
        else:
            embedding.title += f", MDS (balance factor: {layout_param})"

        embedding.metadata["layout_method"] = "MDS"
        embedding.metadata["layout_params"]["balance factor"] = layout_param

    elif layout_method == "FR":
        compute_fruchterman_reingold_layout(
            embedding,
            partition_subgraphs,
            scale=layout_scale,
            iterations=layout_param,
            boundary_neighbors=partition_boundary_neighbors
            if boundary_neighbors
            else None,
            inplace=True,
            verbose=verbose,
        )

        embedding.title += f", FR layouting (iterations: {layout_param})"
        embedding.metadata["layout_method"] = "FR"
        embedding.metadata["layout_params"]["iterations"] = layout_param

    else:
        raise ValueError(
            f"Layout method '{layout_method}' is not supported. Currently, only 'KK' (Kamada Kawai) and 'FR' (Fruchterman-Reingold) are available."
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
    highdim_dists: npt.NDArray[np.float32], lowdim_dists: npt.NDArray[np.float32]
) -> float:
    if not highdim_dists.any():
        print("WARNING: Highdim distances are all 0. Returning 1 as scaling factor.")
        return 1.0

    numerator = np.dot(highdim_dists, lowdim_dists)
    denominator = np.dot(lowdim_dists, lowdim_dists)
    return numerator / denominator


def compute_community_graphs(
    embedding: EmbeddingObj, boundary_neigbors: bool = False
) -> tuple[dict[int, nx.Graph], dict[int, tuple[float, float]], dict[int, list[Any]]]:
    partition_subgraphs = {}
    partition_centers = {}
    partition_boundary_neighbors = {}

    for part, nodes in embedding.com_partition.items():
        subgraph_points_coords = np.array([embedding.embedding[i] for i in nodes])
        # min_x, min_y = subgraph_points_coords.min(axis=0)
        # max_x, max_y = subgraph_points_coords.max(axis=0)
        # partition_centers[part] = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        partition_centers[part] = np.median(subgraph_points_coords, axis=0)

        subgraph = embedding.sim_graph.subgraph(
            [
                node
                for node, attrs_dict in embedding.sim_graph.nodes(data=True)
                if attrs_dict["community"] == part
            ]
        ).copy()

        if boundary_neigbors:
            subgraph, boundary_neighbors = add_boundary_edges(
                embedding.sim_graph, subgraph
            )
            partition_boundary_neighbors[part] = boundary_neighbors

        partition_subgraphs[part] = subgraph

    return partition_subgraphs, partition_centers, partition_boundary_neighbors


def compute_kamada_kawai_layout(
    embedding: EmbeddingObj,
    partition_subgraphs: dict[int, nx.Graph],
    pairwise_dists: npt.NDArray[np.float32],
    scale: int,
    boundary_neighbors: dict[int, list[int]] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> EmbeddingObj:
    """
    4.3 Step of the modDR pipeline: Execute position-movement via Kamada Kawai-layouting.
    """

    if not inplace:
        embedding = copy.deepcopy(embedding)

    if verbose:
        print("Start computation with Kamada Kawai-algorithm.")

    node_list = list(embedding.sim_graph.nodes)
    idx = {n: k for k, n in enumerate(node_list)}

    for part_key, part_graph in partition_subgraphs.items():
        subgraph_pos = {node: embedding.embedding[node] for node in part_graph.nodes}

        subdist = {
            u: {
                v: float(pairwise_dists[idx[u]][idx[v]])
                for v in part_graph.neighbors(u)
            }
            for u in part_graph.nodes
        }

        subgraph_updated_pos = nx.kamada_kawai_layout(
            part_graph,
            dist=subdist,
            pos=subgraph_pos,
            center=embedding.partition_centers[part_key],
            # weight="weight",
            scale=scale,
        )

        if boundary_neighbors is not None:
            for boundary_node in boundary_neighbors[part_key]:
                subgraph_updated_pos.pop(boundary_node, None)

        embedding.embedding.update(subgraph_updated_pos)

    embedding.embedding = dict(sorted(embedding.embedding.items()))
    return embedding


def compute_fruchterman_reingold_layout(
    embedding: EmbeddingObj,
    partition_subgraphs: dict[int, nx.Graph],
    scale: int,
    iterations: int,
    boundary_neighbors: dict[int, list[int]] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> EmbeddingObj:
    """
    4.3 Step of the modDR pipeline: Execute position-movement via Kamada Kawai-layouting.
    """

    if not inplace:
        embedding = copy.deepcopy(embedding)

    if verbose:
        print("Start computation with Fruchterman-Reingold-algorithm.")

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


def compute_mds_layout(
    embedding: EmbeddingObj,
    partition_subgraphs: dict[int, nx.Graph],
    pairwise_dists: npt.NDArray[np.float32],
    balance_factor: float = 0.5,
    precomputed: dict[int, npt.NDArray[np.float32]] = None,
    boundary_neighbors: dict[int, list[int]] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> tuple[EmbeddingObj, dict[int, npt.NDArray[np.float32]]]:
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

    if precomputed is not None:
        updated_pos_dict = precomputed.copy()
        updated_pos_dict_scaled.update(
            {
                key: precomputed[key] * balance_factor
                + (1 - balance_factor) * original_pos_dict[key]
                for key in updated_pos_dict
            }
        )
    else:
        for part_key, part_graph in partition_subgraphs.items():
            if len(part_graph.nodes) == 1:
                print(
                    f"INFO: Skipping partition {part_key} with only {len(part_graph.nodes)} node(s) for MDS layouting."
                )
                skipped_node_index = embedding.com_partition[part_key][0]
                updated_pos_dict_scaled[skipped_node_index] = original_pos_dict[
                    skipped_node_index
                ]
                continue

            subgraph_pos = {
                node: original_pos_dict[node]
                for node in embedding.com_partition[part_key]
            }

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

            new_pos += embedding.partition_centers[part_key]
            new_post_dict = {node: new_pos[i] for i, node in enumerate(subgraph_pos)}

            if boundary_neighbors is not None:
                for boundary_node in boundary_neighbors[part_key]:
                    new_post_dict.pop(boundary_node, None)

            updated_pos_dict.update(new_post_dict)

            updated_pos_dict_scaled.update(
                {
                    key: new_post_dict[key] * balance_factor
                    + (1 - balance_factor) * original_pos_dict[key]
                    for key in new_post_dict
                }
            )

    updated_pos_dict_scaled = dict(sorted(updated_pos_dict_scaled.items()))
    updated_pos_dict = dict(sorted(updated_pos_dict.items()))

    embedding.embedding = updated_pos_dict_scaled

    return embedding, updated_pos_dict


def deprecated_compute_kamada_kawai_layout(
    graph: nx.Graph,
    embedding_dict: dict[int, npt.NDArray[np.float32]],
    partition_centers: dict[int, npt.NDArray[np.float32]],
    partition_subgraphs: dict[int, nx.Graph],
    partition_dict: dict[int, float],
    level: int,
    threshold: float | None = None,
    mst: bool = False,
    boundary_edges: bool = False,
    pairwise_dists: npt.NDArray[np.float32] | None = None,
) -> EmbeddingObj:
    modified_embedding_dict = embedding_dict.copy()

    node_list = list(graph.nodes)

    pairwise_dists_dict = {
        node_list[i]: {
            node_list[j]: pairwise_dists[i][j] for j in range(len(node_list)) if i != j
        }
        for i in range(len(node_list))
    }

    print("------------------------------------------------------------")
    print(
        f"Computing modified embedding via Kamada Kawai-layouting at level {level + 1}"
    )
    start_time = time.time()

    for part_key, part_graph in partition_subgraphs.items():
        subgraph_pos = {node: embedding_dict[node] for node in part_graph.nodes}

        subgraph_updated_pos = nx.kamada_kawai_layout(
            part_graph,
            dist=pairwise_dists_dict,
            pos=subgraph_pos,
            center=partition_centers[part_key],
            weight=None,
            scale=6.0,
        )

        for node in part_graph.nodes():
            modified_embedding_dict[node] = subgraph_updated_pos[node]

    emb = EmbeddingObj(
        graph,
        modified_embedding_dict,
        np.array([]),
        labels=partition_dict,
        partition_centers=partition_centers,
    )

    emb.com_partition = partition_dict

    # TODO: add unique-id generation
    emb.obj_id = level + 1
    emb.title = f"Positions after Kamada Kawai-layouting at level {level + 1}"

    if boundary_edges:
        emb.title += ", boundary edges added"

    if threshold is not None:
        emb.title += f", edge-weight threshold: {threshold}"

    if mst:
        emb.title += ", MST-edges used"

    end_time = time.time()
    print(f"Computation finished after {end_time - start_time:.2f} seconds")
    print("------------------------------------------------------------")

    return emb


def generate_pairwise_threshold_graphs(
    similarity_matrix: npt.NDArray[np.float32], threshold: float
) -> tuple[nx.Graph, list[dict[str, Any]]]:
    for i in range(len(similarity_matrix)):
        similarity_matrix[i] = [
            x if x >= threshold else 0 for x in similarity_matrix[i]
        ]

    graph_thresholded = nx.Graph(similarity_matrix)
    edge_weights = [
        graph_thresholded[u][v]["weight"] for u, v in graph_thresholded.edges()
    ]
    return graph_thresholded, edge_weights


def compute_force_directed(
    graph: nx.Graph, initial_pos: npt.NDArray[np.float32], iterations: list[int]
) -> list[EmbeddingObj]:
    embeddings = []
    embedding_dict = {i: initial_pos[i] for i in range(len(initial_pos))}

    initial_emb = EmbeddingObj(graph, embedding_dict, np.array([]))

    initial_emb.obj_id = 0
    initial_emb.title = "Initial positions"
    embeddings.append(initial_emb)

    for iteration in iterations:
        print("------------------------------------------------------------")
        print("Computing modified embedding for iteration: ", iteration)

        pos_dict = nx.spring_layout(
            graph,
            pos=embedding_dict,
            iterations=iteration,
            threshold=0.0001,
        )

        emb = EmbeddingObj(graph, pos_dict, np.array([]))

        emb.obj_id = iteration
        emb.title = "Position after " + str(iteration) + " iterations"
        embeddings.append(emb)

        print("Computation finished")
        print("------------------------------------------------------------")

    return embeddings


def compute_local_force_directed(
    graph: nx.Graph,
    initial_pos: npt.NDArray[np.float32],
    iterations: list[int],
    threshold: float | None = None,
    mst: bool = False,
    boundary_edges: bool = False,
    method: str = "kawai",
    pairwise_dists: npt.NDArray[np.float32] | None = None,
) -> tuple[list[EmbeddingObj], dict[int, int]]:
    partition_dict = community_louvain.best_partition(graph, random_state=0)

    # remove edges with weight (similarity) smaller than the threshold
    if threshold is not None:
        graph_trimmed = graph.copy()
        for u, v, w in graph.edges.data("weight"):
            if w < threshold:
                graph_trimmed.remove_edge(u, v)

        graph = graph_trimmed

    embeddings = []
    embedding_dict = {i: initial_pos[i] for i in range(len(initial_pos))}
    partition_values = set(partition_dict.values())

    partition_centers = {}
    partition_subgraphs = {}
    partition_boundary_neighbors = {}
    mod_graph = nx.Graph()

    # compute center-coordinates (bounding box) of each partition used in force-directed layout
    for partition in partition_values:
        subgraph_points = [k for k, v in partition_dict.items() if v == partition]
        subgraph_points_coords = np.array([initial_pos[i] for i in subgraph_points])
        min_x, min_y = subgraph_points_coords.min(axis=0)
        max_x, max_y = subgraph_points_coords.max(axis=0)
        partition_centers[partition] = ((min_x + max_x) / 2, (min_y + max_y) / 2)

        subgraph = graph.subgraph(
            [node for node, part in partition_dict.items() if part == partition]
        ).copy()

        # TODO: Check if this is alternative subgraph computation is needed
        # subgraph_nodes = [
        #     node for node, part in partition_dict.items() if part == partition
        # ]
        # subgraph = nx.Graph()
        # subgraph.add_nodes_from(subgraph_nodes)
        #
        # for u, v in graph.edges():
        #     if u in subgraph_nodes and v in subgraph_nodes:
        #         subgraph.add_edge(u, v, weight=graph[u][v]["weight"])

        if boundary_edges:
            subgraph, boundary_neighbors = add_boundary_edges(subgraph, subgraph)
            partition_boundary_neighbors[partition] = boundary_neighbors

        if mst:
            subgraph = compute_mst(subgraph)

        partition_subgraphs[partition] = subgraph

        mod_graph.add_nodes_from(subgraph.nodes())
        mod_graph.add_edges_from(subgraph.edges(data=True))

    initial_emb = EmbeddingObj(
        mod_graph,
        embedding_dict,
        np.array([]),
        labels=partition_dict,
        partition_centers=partition_centers,
    )

    initial_emb.obj_id = 0
    initial_emb.title = "Initial positions"
    embeddings.append(initial_emb)

    if method == "fr":
        embeddings += compute_spring_electrical_layout(
            graph.copy(),
            embedding_dict.copy(),
            partition_centers,
            iterations,
            partition_subgraphs,
            partition_dict,
            partition_boundary_neighbors,
            threshold=threshold,
            mst=mst,
            boundary_edges=boundary_edges,
        )
    elif method == "kawai":
        embeddings += [
            deprecated_compute_kamada_kawai_layout(
                graph.copy(),
                embedding_dict.copy(),
                partition_centers,
                partition_subgraphs,
                partition_dict,
                level=0,
                threshold=threshold,
                mst=mst,
                boundary_edges=boundary_edges,
                pairwise_dists=pairwise_dists,
            )
        ]

    return embeddings, partition_dict


def compute_multilevel_dr(
    graph: nx.Graph,
    initial_pos: npt.NDArray[np.float32],
    pairwise_dists: npt.NDArray[np.float32],
    mst: bool = False,
    boundary_edges: bool = False,
    threshold: float | None = None,
) -> list[EmbeddingObj]:
    dendrogram = community_louvain.generate_dendrogram(graph, random_state=0)

    # remove edges with weight (similarity) smaller than the threshold
    if threshold is not None:
        graph_trimmed = graph.copy()
        for u, v, w in graph.edges.data("weight"):
            if w < threshold:
                graph_trimmed.remove_edge(u, v)

        graph = graph_trimmed

    embeddings = []
    embedding_dict = {i: initial_pos[i] for i in range(len(initial_pos))}

    initial_emb = EmbeddingObj(
        graph,
        embedding_dict,
        np.array([]),
        labels=community_louvain.partition_at_level(dendrogram, len(dendrogram) - 1),
    )

    initial_emb.obj_id = 0
    initial_emb.title = "Initial positions"
    initial_emb.com_partition = community_louvain.partition_at_level(
        dendrogram, len(dendrogram) - 1
    )
    embeddings.append(initial_emb)

    for level in range(len(dendrogram)):
        partition_dict = community_louvain.partition_at_level(dendrogram, level)
        partition_values = set(partition_dict.values())

        partition_centers = {}
        partition_subgraphs = {}
        partition_boundary_neighbors = {}

        # compute center-coordinates (bounding box) of each partition used in force-directed layout
        for partition in partition_values:
            subgraph_points = [k for k, v in partition_dict.items() if v == partition]
            subgraph_points_coords = np.array([initial_pos[i] for i in subgraph_points])
            min_x, min_y = subgraph_points_coords.min(axis=0)
            max_x, max_y = subgraph_points_coords.max(axis=0)
            partition_centers[partition] = ((min_x + max_x) / 2, (min_y + max_y) / 2)

            # subgraph = graph.subgraph(
            #     [node for node, part in partition_dict.items() if part == partition]
            # ).copy()

            # TODO: Check if this is alternative subgraph computation is needed
            subgraph_nodes = [
                node for node, part in partition_dict.items() if part == partition
            ]
            subgraph = nx.Graph()
            subgraph.add_nodes_from(subgraph_nodes)

            for u, v in graph.edges():
                if u in subgraph_nodes and v in subgraph_nodes:
                    subgraph.add_edge(u, v, weight=graph[u][v]["weight"])

            if boundary_edges:
                subgraph, boundary_neighbors = add_boundary_edges(subgraph, subgraph)
                partition_boundary_neighbors[partition] = boundary_neighbors

            if mst:
                subgraph = compute_mst(subgraph)

            partition_subgraphs[partition] = subgraph

        embeddings += [
            deprecated_compute_kamada_kawai_layout(
                graph.copy(),
                embedding_dict.copy(),
                partition_centers,
                partition_subgraphs,
                partition_dict,
                level=level,
                threshold=threshold,
                mst=mst,
                boundary_edges=boundary_edges,
                pairwise_dists=pairwise_dists,
            )
        ]

    return embeddings


def compute_multilevel_fd(
    data: pd.DataFrame, sim_features: list[str], sim_threshold: int = -2
) -> list[EmbeddingObj]:
    # TODO: maybe outsource to separate function
    knn_graph = kneighbors_graph(data, n_neighbors=3, mode="distance")

    pairwise_dists = compute_pairwise_dists(data, invert=True)
    pairwise_dists_feat = compute_pairwise_dists(
        data, sim_features=sim_features, invert=False
    )
    knn_graph_nx = nx.Graph(knn_graph)

    for u, v in knn_graph_nx.edges():
        knn_graph_nx[u][v]["weight"] = pairwise_dists[u][v]

    print("------------------------------------------------------------")
    print("Computing initial multilevel graph layout via force electrical-layouting")
    start_time = time.time()

    # I. coarsening phase
    dendrogram = community_louvain.generate_dendrogram(knn_graph_nx, random_state=0)
    dendrogram_size = len(dendrogram)

    # II. compute initial layout
    partition_dict = community_louvain.partition_at_level(
        dendrogram, dendrogram_size - 1
    )
    induced_graph = community_louvain.induced_graph(partition_dict, knn_graph_nx)
    induced_graph.remove_edges_from(nx.selfloop_edges(induced_graph))

    embeddings = []
    iterations = 50

    graph_updated_pos = nx.spring_layout(
        induced_graph,
        iterations=iterations,
        weight="weight",
        k=5.0,
        seed=0,
        scale=100.0,
    )

    initial_emb = EmbeddingObj(
        induced_graph, graph_updated_pos, np.array([]), labels=partition_dict
    )

    initial_emb.obj_id = dendrogram_size - 1
    initial_emb.title = (
        f"Multilevel spring-electric layout at level {dendrogram_size - 1}"
    )
    initial_emb.com_partition = partition_dict
    embeddings.append(initial_emb)

    end_time = time.time()
    print(f"Computation finished after {end_time - start_time:.2f} seconds")
    print("------------------------------------------------------------")

    # III. refinement step (iterate over reversed dendrogram levels, start at the top)
    for level in range(dendrogram_size - 2, -2, -1):
        print("------------------------------------------------------------")
        print(
            f"Computing multilevel graph layout via force electrical-layouting at level {level}"
        )
        start_time = time.time()

        partition_prev_dict = embeddings[-1].com_partition
        partition_centers = embeddings[-1].embedding
        partition_subgraphs = {}

        if level >= 0:
            partition_dict = community_louvain.partition_at_level(dendrogram, level)
            induced_graph = community_louvain.induced_graph(
                partition_dict, knn_graph_nx
            )
        else:
            partition_dict = embeddings[-1].com_partition
            induced_graph = knn_graph_nx

        induced_graph.remove_edges_from(nx.selfloop_edges(induced_graph))

        for partition in set(partition_prev_dict.values()):
            subgraph_points = [
                k for k, v in partition_prev_dict.items() if v == partition
            ]

            # filter abstract nodes, which are contained in the partition at the next higher level
            if level >= 0:
                com_points = set(
                    [partition_dict[k] for k in subgraph_points if k in partition_dict]
                )
                # com_points = [
                #     v
                #     for k, v in partition_dict.items()
                #     for k1 in subgraph_points
                #     if k == k1
                # ]
                # com_points = set(com_points)
            else:
                com_points = set(subgraph_points)

            subgraph = induced_graph.subgraph([node for node in com_points]).copy()
            partition_subgraphs[partition] = subgraph

        iteration_embedding_dict = {}
        for part_key, part_graph in partition_subgraphs.items():
            if part_key not in partition_centers:
                print(f"Warning: partition {part_key} has no center. Using origin.")

            initial_pos_dict = {
                node: partition_centers[part_key] for node in part_graph.nodes()
            }

            if level <= sim_threshold:
                node_list = list(part_graph.nodes)

                pairwise_dists_dict = {
                    node_list[i]: {
                        node_list[j]: pairwise_dists_feat[i][j]
                        for j in range(len(node_list))
                        if i != j
                    }
                    for i in range(len(node_list))
                }

                subgraph_updated_pos = nx.kamada_kawai_layout(
                    part_graph,
                    dist=pairwise_dists_dict,
                    pos=initial_pos_dict,
                    center=partition_centers[part_key],
                    weight=None,
                    scale=6.0,
                )
            else:
                subgraph_updated_pos = nx.spring_layout(
                    part_graph,
                    pos=initial_pos_dict,
                    iterations=100,
                    weight="weight",
                    center=partition_centers[part_key],
                    k=level + 3.0,
                    scale=10.0 + max(0, level * 10),
                    seed=0,
                )

            for node in part_graph.nodes():
                iteration_embedding_dict[node] = subgraph_updated_pos[node]

        # Define embedding object
        emb = EmbeddingObj(
            induced_graph,
            iteration_embedding_dict,
            np.array([]),
            labels=partition_prev_dict,
            partition_centers=partition_centers,
        )
        emb.com_partition = partition_dict

        emb.obj_id = level
        emb.title = f"Multilevel spring-electric layout at level {level}"

        embeddings.append(emb)

        end_time = time.time()
        print(f"Computation finished after {end_time - start_time:.2f} seconds")
        print("------------------------------------------------------------")

    return embeddings


def compute_mst(graph: nx.Graph) -> nx.Graph:
    # Invert weights, as similarity optimizes for 1, as MST-algorithms as Kruskal optimize for 0
    for u, v in graph.edges:
        graph[u][v]["weight"] = 1 - graph[u][v]["weight"]

    mst_graph = nx.minimum_spanning_tree(graph)

    # Revert weights to initial scaling
    for u, v in mst_graph.edges:
        mst_graph[u][v]["weight"] = 1 - mst_graph[u][v]["weight"]

    return mst_graph


def add_boundary_edges(
    graph: nx.Graph, subgraph: nx.Graph
) -> tuple[nx.Graph, list[Any]]:
    subgraph_boundary_neighbors = subgraph.copy()
    boundary_neighbors = set()

    for node in subgraph.nodes():
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            if neighbor not in subgraph.nodes():
                boundary_neighbors.add(neighbor)
                subgraph_boundary_neighbors.add_node(neighbor)
                subgraph_boundary_neighbors.add_edge(
                    node, neighbor, weight=graph[node][neighbor]["weight"]
                )

    return subgraph_boundary_neighbors, list(boundary_neighbors)


def compute_spring_electrical_layout(
    graph: nx.Graph,
    embedding_dict: dict[int, npt.NDArray[np.float32]],
    partition_centers: dict[int, npt.NDArray[np.float32]],
    iterations: list[int],
    partition_subgraphs=dict[int, nx.Graph],
    partition_dict=dict[int, float],
    partition_boundary_neighbors=dict[int, npt.NDArray[int]],
    threshold: float | None = None,
    mst: bool = False,
    boundary_edges: bool = False,
) -> list[EmbeddingObj]:
    embeddings = []

    for iteration in iterations:
        print("------------------------------------------------------------")
        print("Computing modified embedding for iteration: ", iteration)

        iteration_embedding_dict = embedding_dict.copy()

        for part_key, part_graph in partition_subgraphs.items():
            subgraph_pos = {node: embedding_dict[node] for node in part_graph.nodes}

            subgraph_updated_pos = nx.spring_layout(
                part_graph,
                pos=subgraph_pos,
                iterations=iteration,
                fixed=partition_boundary_neighbors[part_key]
                if boundary_edges
                else None,
                threshold=0.0001,
                weight="weight",
                center=partition_centers[part_key],
                k=5.0,
                seed=0,
            )

            for node in part_graph.nodes():
                iteration_embedding_dict[node] = subgraph_updated_pos[node]

        emb = EmbeddingObj(
            graph,
            iteration_embedding_dict,
            np.array([]),
            labels=partition_dict,
            partition_centers=partition_centers,
        )

        emb.obj_id = iteration
        emb.title = "Positions after " + str(iteration) + " iterations"

        if boundary_edges:
            emb.title += ", boundary edges added"

        if threshold is not None:
            emb.title += f", edge-weight threshold: {threshold}"

        if mst:
            emb.title += ", MST-edges used"

        embeddings.append(emb)

        print("Computation finished")
        print("------------------------------------------------------------")

    return embeddings


def compute_knn_graph(
    df: pd.DataFrame,
    n_neighbors: int = 3,
    mode: str = "distance",
    sim_features: list[str] | None = None,
) -> tuple[nx.Graph, npt.NDArray[np.float32]]:
    if sim_features is None or len(sim_features) == 0:
        knn_graph = kneighbors_graph(df, n_neighbors=n_neighbors, mode=mode)
    else:
        knn_graph = kneighbors_graph(
            df.loc[:, sim_features], n_neighbors=n_neighbors, mode=mode
        )

    pairwise_dists = compute_pairwise_dists(df, sim_features=sim_features)
    knn_graph_nx = nx.Graph(knn_graph)

    edge_weights_knn = np.array([])
    for u, v in knn_graph_nx.edges():
        edge_weights_knn = np.append(edge_weights_knn, [pairwise_dists[u][v]])
        knn_graph_nx[u][v]["weight"] = pairwise_dists[u][v]

    return knn_graph_nx, edge_weights_knn


def fit(
    data: pd.DataFrame,
    initial_pos: npt.NDArray[np.float32],
    sim_features: list[str],
    method: str = "force-directed",
    n_neighbors: int = 3,
    iterations: list[int] | None = None,
    knn_graph: nx.Graph | None = None,
) -> list[EmbeddingObj]:
    # TODO: Add assertion for falsy parameters
    if iterations is None:
        iterations = [1, 3, 5, 10]

    if knn_graph is None:
        knn_graph = kneighbors_graph(
            data.loc[:, sim_features], n_neighbors=n_neighbors, mode="distance"
        )
        knn_graph = nx.Graph(knn_graph)

    pairwise_dists = compute_pairwise_dists(data, sim_features)

    for u, v in knn_graph.edges():
        knn_graph[u][v]["weight"] = pairwise_dists[u][v]

    embeddings = []

    if method == "force-directed":
        embeddings = compute_force_directed(knn_graph, initial_pos, iterations)
    elif method == "local-force-directed":
        embeddings = compute_local_force_directed(knn_graph, initial_pos, iterations)

    return embeddings
