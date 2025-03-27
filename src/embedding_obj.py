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

    m_jaccard: np.ndarray

    def __init__(self,
                 graph: networkx.Graph,
                 embedding = np.array([]),
                 edge_weights: np.ndarray = np.array([]),
                 title: Optional[str] = None,
                 marker: Optional[float] = None,
                 m_jaccard: np.ndarray = np.array([])):
        self.sim_graph = graph
        self.embedding = embedding
        if len(edge_weights) == 0:
            self.edge_weights = self.get_edge_weights()
        else:
            self.edge_weights = edge_weights

        self.title = title
        self.marker = marker
        self.m_jaccard = m_jaccard


    def get_edge_weights(self):
        edges = self.sim_graph.edges(data=True)
        return np.array([e[-1]["weight"] for e in edges if "weight" in e[-1]])
