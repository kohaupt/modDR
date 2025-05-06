from typing import Any, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from community import community_louvain
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore

from embedding_obj import EmbeddingObj  # type: ignore


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
    threshold: Optional[float] = None,
    mst: bool = False,
    boundary_edges: bool = False,
    method: str = "kawai",
    pairwise_dists: Optional[npt.NDArray[np.float32]] = None,
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

        # subgraph = graph.subgraph(
        #     [node for node, part in partition_dict.items() if part == partition]
        # ).copy()

        subgraph_nodes = [
            node for node, part in partition_dict.items() if part == partition
        ]
        subgraph = nx.Graph()
        subgraph.add_nodes_from(subgraph_nodes)

        for u, v in graph.edges():
            if u in subgraph_nodes and v in subgraph_nodes:
                subgraph.add_edge(u, v, weight=graph[u][v]["weight"])

        if boundary_edges:
            boundary_neighbors = []

            for u, v in graph.edges():
                if u in subgraph.nodes() and v not in subgraph.nodes():
                    boundary_neighbors.append(v)
                    subgraph.add_node(v)
                    subgraph.add_edge(u, v, weight=graph[u][v]["weight"])

                if v in subgraph.nodes() and u not in subgraph.nodes():
                    boundary_neighbors.append(u)
                    subgraph.add_node(u)
                    subgraph.add_edge(v, u, weight=graph[v][u]["weight"])

            partition_boundary_neighbors[partition] = boundary_neighbors

        if mst:
            # Invert weights, as similarity optimizes for 1, as MST-algorithms as Kruskal optimize for 0
            for u, v in subgraph.edges:
                subgraph[u][v]["weight"] = 1 - subgraph[u][v]["weight"]

            subgraph = nx.minimum_spanning_tree(subgraph)

            # Revert weights to initial scaling
            for u, v in subgraph.edges:
                subgraph[u][v]["weight"] = 1 - subgraph[u][v]["weight"]

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
            compute_kamada_kawai_layout(
                graph.copy(),
                embedding_dict.copy(),
                partition_centers,
                partition_subgraphs,
                partition_dict,
                threshold=threshold,
                mst=mst,
                boundary_edges=boundary_edges,
                pairwise_dists=pairwise_dists,
            )
        ]

    return embeddings, partition_dict


def compute_spring_electrical_layout(
    graph: nx.Graph,
    embedding_dict: dict[int, npt.NDArray[np.float32]],
    partition_centers: dict[int, npt.NDArray[np.float32]],
    iterations: list[int],
    partition_subgraphs=dict[int, nx.Graph],
    partition_dict=dict[int, float],
    partition_boundary_neighbors=dict[int, npt.NDArray[int]],
    threshold: Optional[float] = None,
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

            # subgraph_pos = {
            #     node: subgraph_updated_pos[node] for node in part_graph.nodes
            # }
            # subgraph_updated_pos = nx.kamada_kawai_layout(
            #     part_graph,
            #     pos=subgraph_pos,
            #     center=partition_centers[part_key],
            #     scale=5.0,
            # )

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


def compute_kamada_kawai_layout(
    graph: nx.Graph,
    embedding_dict: dict[int, npt.NDArray[np.float32]],
    partition_centers: dict[int, npt.NDArray[np.float32]],
    partition_subgraphs=dict[int, nx.Graph],
    partition_dict=dict[int, float],
    threshold: Optional[float] = None,
    mst: bool = False,
    boundary_edges: bool = False,
    pairwise_dists: Optional[npt.NDArray[np.float32]] = None,
) -> EmbeddingObj:
    modified_embedding_dict = embedding_dict.copy()

    # for u, v in graph.edges:
    #     graph[u][v]["weight"] = 1 - graph[u][v]["weight"]

    node_list = list(graph.nodes)

    pairwise_dists_dict = {
        node_list[i]: {
            node_list[j]: pairwise_dists[i][j] for j in range(len(node_list)) if i != j
        }
        for i in range(len(node_list))
    }

    print("------------------------------------------------------------")
    print("Computing modified embedding via Kamada Kawai-layouting")

    for part_key, part_graph in partition_subgraphs.items():
        subgraph_pos = {node: embedding_dict[node] for node in part_graph.nodes}

        subgraph_updated_pos = nx.kamada_kawai_layout(
            part_graph,
            dist=pairwise_dists_dict,
            pos=subgraph_pos,
            center=partition_centers[part_key],
            weight=None,
            scale=5.0,
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

    # TODO: add unique-id generation
    emb.obj_id = 1000
    emb.title = "Positions after Kamada Kawai-layouting"

    if boundary_edges:
        emb.title += ", boundary edges added"

    if threshold is not None:
        emb.title += f", edge-weight threshold: {threshold}"

    if mst:
        emb.title += ", MST-edges used"

    print("Computation finished")
    print("------------------------------------------------------------")

    return emb


def compute_pairwise_dists(
    df: pd.DataFrame,
    sim_features: Optional[list[str]],
    normalize: bool = True,
    apply_squareform: bool = True,
    invert: bool = True,
) -> npt.NDArray[np.float32]:
    input_data = []

    if sim_features is not None and sim_features != []:
        input_data = df[sim_features].to_numpy()
    else:
        input_data = df.to_numpy()

    distances = pdist(input_data, metric="euclidean")

    if normalize:
        distances = MinMaxScaler().fit_transform(distances.reshape(-1, 1)).flatten()

    if apply_squareform:
        distances = squareform(distances)

    if invert:
        distances = 1 - distances

    return distances.astype(np.float32)


def compute_graph_weights(
    df: pd.DataFrame,
    sim_features: list[str],
    n_neighbors: int = 3,
    mode: str = "distance",
) -> tuple[nx.Graph, npt.NDArray[np.float32]]:
    knn_graph = kneighbors_graph(
        df.loc[:, sim_features], n_neighbors=n_neighbors, mode=mode
    )

    pairwise_dists = compute_pairwise_dist(df, sim_features)
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
    iterations: Optional[list[int]] = None,
    knn_graph: Optional[nx.Graph] = None,
) -> list[EmbeddingObj]:
    # TODO: Add assertion for falsy parameters
    if iterations is None:
        iterations = [1, 3, 5, 10]

    if knn_graph is None:
        knn_graph = kneighbors_graph(
            data.loc[:, sim_features], n_neighbors=n_neighbors, mode="distance"
        )
        knn_graph = nx.Graph(knn_graph)

    pairwise_dists = compute_pairwise_dist(data, sim_features)

    for u, v in knn_graph.edges():
        knn_graph[u][v]["weight"] = pairwise_dists[u][v]

    embeddings = []

    if method == "force-directed":
        embeddings = compute_force_directed(knn_graph, initial_pos, iterations)
    elif method == "local-force-directed":
        embeddings = compute_local_force_directed(knn_graph, initial_pos, iterations)

    return embeddings