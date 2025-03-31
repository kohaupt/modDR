import networkx
import networkx as nx
import numpy as np
import pandas
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

from embedding_obj import EmbeddingObj


def generate_pairwise_threshold_graphs(
    similarity_matrix: np.array, threshold: float
) -> tuple[networkx.Graph, np.array]:
    """
    Generate pairwise threshold graphs from a similarity matrix.

    Parameters:
        similarity_matrix: A 2 dimensional array representing the similarity matrix.
        threshold (float): The threshold value to determine the relevant edges in the graph. Values
                           below the threshold will be set to 0.

    Returns:
        tuple: A tuple containing the generated graph (networkx.Graph) and an array of edge weights.
    """
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
    graph: networkx.Graph,
    iterations: int,
    initial_pos: np.array,
    threshold: int = 0.0001,
) -> np.ndarray:
    embedding_dict = {i: initial_pos[i] for i in range(len(initial_pos))}

    pos_dict = nx.spring_layout(
        graph,
        pos=embedding_dict,
        iterations=iterations,
        threshold=threshold,
    )

    return np.array(list(pos_dict.values()))


def compute_iterations(
    graph: networkx.Graph,
    initial_pos: np.array,
    iterations: list[int],
    method: str = "force-directed",
) -> list[EmbeddingObj]:
    embeddings = []

    for iteration in iterations:
        emb = EmbeddingObj(graph)

        if method == "force-directed":
            emb.embedding = compute_force_directed(graph, iteration, initial_pos)

        emb.marker = iteration
        emb.title = "Position after " + str(iteration) + " iterations"
        embeddings.append(emb)

    return embeddings


def compute_pairwise_dist(df: pandas.DataFrame, sim_features: list[str]) -> np.ndarray:
    connectivity_saturation_pairwise = np.float16(
        pairwise_distances(df.loc[:, sim_features], metric="manhattan")
    )

    connectivity_saturation_pairwise = MinMaxScaler().fit_transform(
        connectivity_saturation_pairwise
    )
    # connectivity_saturation_pairwise = np.round(connectivity_saturation_pairwise, 3)
    return connectivity_saturation_pairwise


def compute_knn(
    df: pandas.DataFrame,
    sim_features: list[str],
    n_neighbors: int = 3,
    mode: str = "distance",
) -> tuple[nx.Graph, np.ndarray]:
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
    data: pandas.DataFrame,
    initial_pos: np.ndarray,
    features: list[str],
    method: str = "force-directed",
    n_neighbors: int = 3,
    iterations: list[int] = None,
) -> list[EmbeddingObj]:
    if iterations is None:
        iterations = [1, 3, 5, 10]
    graph, edge_weights = compute_knn(data, features, n_neighbors)

    embeddings = compute_iterations(graph, initial_pos, iterations, method=method)
    return embeddings
