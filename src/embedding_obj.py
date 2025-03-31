from dataclasses import dataclass
from typing import Optional

import networkx
import numpy as np


@dataclass
class EmbeddingObj:

    sim_graph: networkx.Graph
    embedding: np.ndarray
    edge_weights: np.ndarray
    title: Optional[str]
    marker: Optional[float]

    m_jaccard: Optional[np.ndarray] = None
    m_q_local: Optional[np.ndarray] = None
    m_trustworthiness: Optional[np.ndarray] = None
    m_continuity: Optional[np.ndarray] = None
    m_spearman: Optional[np.ndarray] = None
    m_normalized_stress: Optional[float] = None
    m_total_score: Optional[float] = None

    def __init__(self,
                 graph: networkx.Graph,
                 embedding = np.array([]),
                 edge_weights: np.ndarray = np.array([]),
                 title: Optional[str] = None,
                 marker: Optional[float] = None):
        self.sim_graph = graph
        self.embedding = embedding
        if len(edge_weights) == 0:
            self.edge_weights = self.get_edge_weights()
        else:
            self.edge_weights = edge_weights

        self.title = title
        self.marker = marker

    def __str__(self):
        return ("---------------------------------------\n"
                f"Embedding object (Marker: {self.marker})\n"
                f"Title: '{self.title}'\n"
                f"Graph: {self.sim_graph}\n"
                f"Embedding shape: {self.embedding.shape}\n"
                f"Shape of edge weights: {self.edge_weights.shape}\n\n"
                f"Total score: {self.m_total_score if self.m_total_score is not None else 'not computed'}\n"
                "---------------------------------------")

    def get_edge_weights(self):
        edges = self.sim_graph.edges(data=True)
        return np.array([e[-1]["weight"] for e in edges if "weight" in e[-1]])

    def metrics_info(self):
        str = "---------------------------------------\n"
        str += f"Embedding object (Marker: {self.marker})\n"

        str += f"Total score: {self.m_total_score if self.m_total_score is not None else 'not computed'}\n"

        if self.m_jaccard is not None:
            str += f"Jaccard-Scores: {self.m_jaccard.size} (values)\n"
        else:
            str += "Jaccard-Scores: not computed\n"

        str += f"Q local: {self.m_q_local if self.m_q_local is not None else 'not computed'}\n"
        str += f"Trustworthiness: {self.m_trustworthiness if self.m_trustworthiness is not None else 'not computed'}\n"
        str += f"Continuity: {self.m_continuity if self.m_continuity is not None else 'not computed'}\n"
        str += f"Spearman Score: {self.m_spearman if self.m_spearman is not None else 'not computed'}\n"
        str += f"Normalized Stress: {self.m_normalized_stress if self.m_normalized_stress is not None else 'not computed'}\n"

        print(str)