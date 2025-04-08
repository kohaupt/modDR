from typing import Any, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
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


def compute_force_directed(
    graph: nx.Graph,
    iterations: int,
    initial_pos: npt.NDArray[np.float32],
    threshold: float = 0.0001,
) -> npt.NDArray[np.float32]:
    embedding_dict = {i: initial_pos[i] for i in range(len(initial_pos))}

    pos_dict = nx.spring_layout(
        graph,
        pos=embedding_dict,
        iterations=iterations,
        threshold=threshold,
    )

    return np.array(list(pos_dict.values()))


def compute_iterations(
    graph: nx.Graph,
    initial_pos: npt.NDArray[np.float32],
    iterations: list[int],
    method: str = "force-directed",
) -> list[EmbeddingObj]:
    embeddings = []

    for iteration in iterations:
        # TODO: Add switch case for different methods when implemented
        # if method == "force-directed":
        #     iteration_embedding = compute_force_directed(graph, iteration, initial_pos)
        print("------------------------------------------------------------")
        print("Computing modified embedding for iteration: ", iteration)

        iteration_embedding = compute_force_directed(graph, iteration, initial_pos)
        emb = EmbeddingObj(graph, iteration_embedding, np.array([]))

        emb.id = iteration
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


def compute_knn(
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
) -> list[EmbeddingObj]:
    # TODO: Add assertion for falsy parameters
    if iterations is None:
        iterations = [1, 3, 5, 10]
    graph, edge_weights = compute_knn(data, sim_features, n_neighbors)

    embeddings = compute_iterations(graph, initial_pos, iterations, method=method)
    return embeddings