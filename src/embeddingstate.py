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


@dataclass
class EmbeddingState:
    obj_id: float
    graph: nx.Graph
    embedding: dict[int, npt.NDArray[np.float32]]
    metadata: MetaDataDict
    metrics: MetricsDict
    title: str | None

    com_partition: dict[int, list[int]] | None = None
    partition_centers: dict[int, npt.NDArray[np.float32]] | None = None
    labels: dict[int, float] | None = None

    def __init__(
        self,
        embedding: dict[int, npt.NDArray[np.float32]],
        graph: nx.Graph | None = None,
        title: str | None = None,
        obj_id: float | None = None,
        labels: dict[int, float] | None = None,
        partition_centers: dict[int, npt.NDArray[np.float32]] | None = None,
    ) -> None:
        self.graph = graph
        self.embedding = embedding

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
        )

    def __str__(self) -> str:
        return (
            "---------------------------------------\n"
            f"Embedding object (ID: {self.obj_id})\n"
            f"Title: '{self.title}'\n"
            f"Embedding shape: {len(self.embedding.items())}\n"
            f"Graph nodes: {self.graph.number_of_nodes() if self.graph else 0}\n"
            f"Graph edges: {self.graph.number_of_edges() if self.graph else 0}\n\n"
            f"Metadata: \n  {'\n  '.join(f'{k}: {v}' for k, v in self.metadata.items())}\n\n"  # noqa: E501
            f"Metrics: \n  {'\n  '.join(f'{k}: {v}' for k, v in self.metrics.items())}\n"  # noqa: E501
            "---------------------------------------"
        )
