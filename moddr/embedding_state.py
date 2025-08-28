from dataclasses import dataclass
from typing import TypedDict

import networkx as nx
import numpy as np
import numpy.typing as npt


class MetaDataDict(TypedDict):
    """
    MetaDataDict is a TypedDict that stores metadata for the applied dimensionality reduction, community detection and layout algorithm.

    Attributes:
        dr_method (str): The name of the dimensionality reduction method used.
        dr_params (dict[str, any]): Parameters for the dimensionality reduction method.
        k_neighbors (int): Number of neighbors used in the dimensionality reduction method.
        com_detection (str): The name of the community detection algorithm used.
        com_detection_params (dict[str, any]): Parameters for the community detection algorithm.
        layout_method (str): The name of the layout algorithm used for visualization.
        layout_params (dict[str, any]): Parameters for the layout algorithm.
    """

    dr_method: str
    dr_params: dict[str, any]
    k_neighbors: int
    com_detection: str
    com_detection_params: dict[str, any]
    layout_method: str
    layout_params: dict[str, any]


class MetricsDict(TypedDict):
    """
    Typed dictionary representing various metrics used for evaluating embeddings.

    Attributes:
        trustworthiness (float): Measures error of hard intrusions in the embedding.
        continuity (float): Measures error of hard extrusions in the embedding.
        rnx (float): Relative neighborhood preservation score (see [1]).
        sim_stress (float): Similarity-based Kruskal stress value.
        sim_stress_com (float): Local similarity-based Kruskal stress value for communities.
        sim_stress_com_diff (float): Difference in local similarity-based Kruskal stress for communities.
        rank_score (float): Overall score based on rank metrics.
        distance_score (float): Overall score based on distance metrics.
        total_score (float): Aggregated score combining rank and distance scores.
        coranking_matrix (npt.NDArray[np.int32] | None): Co-ranking matrix representing neighborhood preservation, or None if not available.

    Reference:
        [1]: Lee, J. A., et al. "Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation." Neurocomputing 112 (2013): 92-108. https://doi.org/10.1016/j.neucom.2012.12.036
    """

    trustworthiness: float
    continuity: float
    rnx: float
    sim_stress: float
    sim_stress_com: float
    sim_stress_com_diff: float
    rank_score: float
    distance_score: float
    total_score: float
    coranking_matrix: npt.NDArray[np.int32] | None


@dataclass
class EmbeddingState:
    """
    EmbeddingState is a data class representing the state of an embedding, including its associated graph, metadata, metrics, and partitioning information.

    Attributes:
        obj_id (float): (Unique) identifier for the embedding object.
        graph (nx.Graph): Graph structure associated with the embedding.
        embedding (dict[int, npt.NDArray[np.float32]]): Dictionary mapping node indices to their embedding vectors.
        metadata (MetaDataDict): Dictionary containing metadata about dimensionality reduction, community detection, and layout methods.
        metrics (MetricsDict): Dictionary containing various evaluation metrics for the embedding.
        title (str): Title or description of the embedding object.
        partition (dict[int, list[int]]): Dictionary mapping community indices to lists of node indices.
        community_centers (dict[int, npt.NDArray[np.float32]]): Dictionary mapping community indices to their center vectors.
        labels (dict[int, float]): Dictionary mapping node indices to label values. Can be used e.g. for coloring or categorization.

    Methods:
        __init__(embedding, graph=None, title=None, obj_id=None, com_partition=None, community_centers=None, labels=None):
            Initializes an EmbeddingState instance with the provided embedding, graph, title, object ID, partitioning, community centers, and labels.
            If any argument is None, a default value is assigned.

        __str__():
            Returns a formatted string representation of the EmbeddingState, including object ID, title, embedding shape, graph statistics, metadata, and metrics.
    """

    obj_id: float
    graph: nx.Graph
    embedding: dict[int, npt.NDArray[np.float32]]
    metadata: MetaDataDict
    metrics: MetricsDict
    title: str

    partition: dict[int, list[int]]
    community_centers: dict[int, npt.NDArray[np.float32]]
    labels: dict[int, float]

    def __init__(
        self,
        embedding: dict[int, npt.NDArray[np.float32]],
        graph: nx.Graph | None = None,
        title: str | None = None,
        obj_id: float | None = None,
        partition: dict[int, list[int]] | None = None,
        community_centers: dict[int, npt.NDArray[np.float32]] | None = None,
        labels: dict[int, float] | None = None,
    ) -> None:
        if graph is None:
            self.graph = nx.Graph()
        else:
            self.graph = graph

        if embedding is None:
            self.embedding = {}
        else:
            self.embedding = embedding

        if title is None:
            self.title = ""
        else:
            self.title = title

        if obj_id is None:
            self.obj_id = np.random.rand()
        else:
            self.obj_id = obj_id

        if partition is None:
            self.partition = {0: np.arange(len(self.embedding))}
        else:
            self.partition = partition

        if community_centers is None:
            self.community_centers = {}
        else:
            self.community_centers = community_centers

        if labels is None:
            self.labels = {}
        else:
            self.labels = labels

        self.metadata = MetaDataDict(
            dr_method="",
            dr_params={},
            k_neighbors=0,
            com_detection="",
            com_detection_params={},
            layout_method="",
            layout_params={},
        )

        self.metrics = MetricsDict(
            trustworthiness=None,
            continuity=None,
            rnx=None,
            sim_stress=None,
            sim_stress_com=None,
            sim_stress_com_diff=None,
            rank_score=None,
            distance_score=None,
            total_score=None,
            coranking_matrix=None,
        )

    def __str__(self) -> str:
        return (
            "---------------------------------------\n"
            f"Embedding object (ID: {self.obj_id})\n"
            f"Title: '{self.title}'\n"
            f"Embedding size: {len(self.embedding.items()) if self.embedding else 0}\n"
            f"Graph nodes: {self.graph.number_of_nodes() if self.graph else 0}\n"
            f"Graph edges: {self.graph.number_of_edges() if self.graph else 0}\n\n"
            f"Metadata: \n  {'\n  '.join(f'{k}: {v}' for k, v in self.metadata.items())}\n\n"  # noqa: E501
            f"Metrics: \n  {'\n  '.join(f'{k}: {v}' for k, v in self.metrics.items())}\n"  # noqa: E501
            "---------------------------------------"
        )
