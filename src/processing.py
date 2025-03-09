import networkx as nx


def generate_pairwise_threshold_graphs(similarity_matrix, threshold):
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
