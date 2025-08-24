import pickle
import time
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt

import evaluation
from embeddingstate import EmbeddingState


def save_numpy(
    data: npt.NDArray[np.float32],
    folderpath: str = "/interim/",
    filename: str = "results",
) -> None:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    rootpath = "../../data"
    targetdir = rootpath + folderpath
    np.save(targetdir + filename + "_" + timestr + ".npy", data)


def save_pickle(
    data: Any, folderpath: str = "/interim/", filename: str = "results"
) -> None:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    rootpath = "../../data"
    targetdir = rootpath + folderpath
    with open(targetdir + filename + "_" + timestr + ".pickle", "wb") as outfile:
        pickle.dump(data, outfile)


def load_pickle(folderpath: str = "/interim/", filename: str = "results") -> Any:
    rootpath = "../../data"
    targetdir = rootpath + folderpath
    with open(targetdir + filename + ".pickle", "rb") as infile:
        data = pickle.load(infile)

    return data


def export_to_gexf(data: EmbeddingState, folderpath: str = "/interim/") -> None:
    graph_gexf = data.sim_graph.copy()

    # set minimum edge weight, as gephi requires a minimum edge weight
    for u, v in graph_gexf.edges():
        if graph_gexf[u][v]["weight"] == 0.0:
            graph_gexf[u][v]["weight"] = 0.0001

    for node in graph_gexf.nodes():
        graph_gexf.nodes[node]["viz"] = {
            "position": {
                "x": float(data.embedding[node][0]) * 1000,
                "y": float(data.embedding[node][1]) * 1000,
                "z": 0.0,
            }
        }

        graph_gexf.nodes[node]["feat"] = data.labels[node]

    rootpath = "../../data"
    targetdir = rootpath + folderpath
    nx.write_gexf(graph_gexf, targetdir + "graph_new_feat_it2.gexf")
