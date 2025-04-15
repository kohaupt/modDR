from typing import Any, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from community import community_louvain
from sklearn.metrics import pairwise_distances  # type: ignore
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


def compute_local_force_directed(
    graph: nx.Graph, initial_pos: npt.NDArray[np.float32], iterations: list[int]
) -> npt.NDArray[np.float32]:
    embeddings = []
    embedding_dict = {i: initial_pos[i] for i in range(len(initial_pos))}

    # compute the best partition
    partition_dict = community_louvain.best_partition(graph)
    partition_values = set(partition_dict.values())

    for iteration in iterations:
        print("------------------------------------------------------------")
        print("Computing modified embedding for iteration: ", iteration)

        iteration_embedding_dict = embedding_dict

        for partition in partition_values:
            subgraph = graph.subgraph(
                [node for node, part in partition_dict.items() if part == partition]
            )

            subgraph_pos = {node: initial_pos[node] for node in subgraph.nodes}

            subgraph_updated_pos = nx.spring_layout(
                subgraph,
                pos=subgraph_pos,
                iterations=iteration,
                threshold=0.0001,
                weight="weight",
                seed=0,
            )

            for node in subgraph.nodes():
                iteration_embedding_dict[node] = subgraph_updated_pos[node]

        emb = EmbeddingObj(
            graph,
            np.array(list(iteration_embedding_dict.values())),
            np.array([]),
            labels=partition_dict,
        )

        emb.obj_id = iteration
        emb.title = "Position after " + str(iteration) + " iterations"
        embeddings.append(emb)

        print("Computation finished")
        print("------------------------------------------------------------")

    return embeddings


def compute_force_directed(
    graph: nx.Graph, initial_pos: npt.NDArray[np.float32], iterations: list[int]
) -> list[EmbeddingObj]:
    embeddings = []
    embedding_dict = {i: initial_pos[i] for i in range(len(initial_pos))}

    for iteration in iterations:
        print("------------------------------------------------------------")
        print("Computing modified embedding for iteration: ", iteration)

        pos_dict = nx.spring_layout(
            graph,
            pos=embedding_dict,
            iterations=iteration,
            threshold=0.0001,
        )

        emb = EmbeddingObj(graph, np.array(list(pos_dict.values())), np.array([]))

        emb.obj_id = iteration
        emb.title = "Position after " + str(iteration) + " iterations"
        embeddings.append(emb)

        print("Computation finished")
        print("------------------------------------------------------------")

    return embeddings


def compute_pairwise_dist(
    df: pd.DataFrame, sim_features: list[str]
) -> npt.NDArray[np.float32]:
    connectivity_saturation_pairwise = np.float32(
        pairwise_distances(df.loc[:, sim_features], metric="manhattan")
    )

    connectivity_saturation_pairwise = MinMaxScaler().fit_transform(
        connectivity_saturation_pairwise
    )
    # connectivity_saturation_pairwise = np.round(connectivity_saturation_pairwise, 3)
    return connectivity_saturation_pairwise


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
        np.append(edge_weights_knn, pairwise_dists[u][v])
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

    else:
        pairwise_dists = compute_pairwise_dist(data, sim_features)

        for u, v in knn_graph.edges():
            knn_graph[u][v]["weight"] = pairwise_dists[u][v]

    embeddings = []

    if method == "force-directed":
        embeddings = compute_force_directed(knn_graph, initial_pos, iterations)
    elif method == "local-force-directed":
        embeddings = compute_local_force_directed(
            knn_graph, initial_pos, iterations
        )

    return embeddings