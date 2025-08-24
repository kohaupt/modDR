from dataclasses import dataclass
from typing import TypedDict

import networkx as nx
import numpy as np
import numpy.typing as npt


class MetaDataDict(TypedDict):
    dr_method: str
    dr_params: dict[str, any]
    k_neighbors: int
    com_detection: str
    com_detection_params: dict[str, any]
    layout_method: str
    layout_params: dict[str, any]


class MetricsDict(TypedDict):
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
    jaccard: npt.NDArray[np.float32] | None


@dataclass
class EmbeddingState:
    obj_id: float
    sim_graph: nx.Graph
    embedding: dict[int, npt.NDArray[np.float32]]
    edge_weights: npt.NDArray[np.float32]
    metadata: MetaDataDict
    metrics: MetricsDict
    title: str | None

    com_partition: dict[int, list[int]] | None = None
    partition_centers: dict[int, npt.NDArray[np.float32]] | None = None
    labels: dict[int, float] | None = None

    def __init__(
        self,
        embedding: dict[int, npt.NDArray[np.float32]],
        edge_weights: npt.NDArray[np.float32],
        graph: nx.Graph | None = None,
        title: str | None = None,
        obj_id: float | None = None,
        labels: dict[int, float] | None = None,
        partition_centers: dict[int, npt.NDArray[np.float32]] | None = None,
    ) -> None:
        self.sim_graph = graph
        self.embedding = embedding
        if graph is not None and len(edge_weights) == 0:
            self.edge_weights = self.get_edge_weights()
        else:
            self.edge_weights = edge_weights

        self.title = title

        if obj_id is None:
            self.obj_id = np.random.rand()
        else:
            self.obj_id = obj_id

        self.labels = labels
        self.partition_centers = partition_centers

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
            trustworthiness=0.0,
            continuity=0.0,
            rnx=0.0,
            sim_stress=0.0,
            sim_stress_com=0.0,
            sim_stress_com_diff=0.0,
            rank_score=0.0,
            distance_score=0.0,
            total_score=0.0,
            coranking_matrix=None,
            jaccard=None,
        )

    def __str__(self) -> str:
        return (
            "---------------------------------------\n"
            f"Embedding object (ID: {self.obj_id})\n"
            f"Title: '{self.title}'\n"
            f"Graph: {self.sim_graph}\n"
            f"Embedding shape: {len(self.embedding.items())}\n"
            f"Shape of edge weights: {self.edge_weights.shape}\n\n"
            "---------------------------------------"
        )

    def get_edge_weights(self) -> npt.NDArray[np.float32]:
        edges = self.sim_graph.edges(data=True)
        return np.array([e[-1]["weight"] for e in edges if "weight" in e[-1]])

    def get_metadata_str(self) -> str:
        return " | ".join(f"{k}: {v}" for k, v in self.metadata.items() if v)
