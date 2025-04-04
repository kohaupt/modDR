from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
import numpy.typing as npt


@dataclass
class EmbeddingObj:

    sim_graph: nx.Graph
    embedding: npt.NDArray[np.float32]
    edge_weights: npt.NDArray[np.float32]
    title: Optional[str]
    marker: Optional[float]

    m_jaccard: Optional[npt.NDArray[np.float32]] = None
    m_q_local: Optional[npt.NDArray[np.float32]] = None
    m_trustworthiness: Optional[npt.NDArray[np.float32]] = None
    m_continuity: Optional[npt.NDArray[np.float32]] = None
    m_spearman: Optional[npt.NDArray[np.float32]] = None
    m_normalized_stress: Optional[float] = None
    m_total_score: Optional[float] = None

    def __init__(self,
                 graph: nx.Graph,
                 embedding: npt.NDArray[np.float32],
                 edge_weights: npt.NDArray[np.float32],
                 title: Optional[str] = None,
                 marker: Optional[float] = None) -> None:
        self.sim_graph = graph
        self.embedding = embedding
        if len(edge_weights) == 0:
            self.edge_weights = self.get_edge_weights()
        else:
            self.edge_weights = edge_weights

        self.title = title
        self.marker = marker

    def __str__(self) -> str:
        return ("---------------------------------------\n"
                f"Embedding object (Marker: {self.marker})\n"
                f"Title: '{self.title}'\n"
                f"Graph: {self.sim_graph}\n"
                f"Embedding shape: {self.embedding.shape}\n"
                f"Shape of edge weights: {self.edge_weights.shape}\n\n"
                f"Total score: {self.m_total_score if self.m_total_score is not None else 'not computed'}\n"
                "---------------------------------------")

    def get_edge_weights(self) -> npt.NDArray[np.float32]:
        edges = self.sim_graph.edges(data=True)
        return np.array([e[-1]["weight"] for e in edges if "weight" in e[-1]])

    def metrics_info(self) -> None:
        output_str = "---------------------------------------\n"
        output_str += f"Embedding object (Marker: {self.marker})\n"

        output_str += f"Total score: {self.m_total_score if self.m_total_score is not None else 'not computed'}\n"

        if self.m_jaccard is not None:
            output_str += f"Jaccard-Scores: {self.m_jaccard.size} (values)\n"
        else:
            output_str += "Jaccard-Scores: not computed\n"

        output_str += f"Q local: {self.m_q_local if self.m_q_local is not None else 'not computed'}\n"
        output_str += f"Trustworthiness: {self.m_trustworthiness if self.m_trustworthiness is not None else 'not computed'}\n"
        output_str += f"Continuity: {self.m_continuity if self.m_continuity is not None else 'not computed'}\n"
        output_str += f"Spearman Score: {self.m_spearman if self.m_spearman is not None else 'not computed'}\n"
        output_str += f"Normalized Stress: {self.m_normalized_stress if self.m_normalized_stress is not None else 'not computed'}\n"

        output_str += "---------------------------------------\n\n"

        print(output_str)