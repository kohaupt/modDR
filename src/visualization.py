import math
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from embedding_obj import EmbeddingObj


def display_reduction_results(results, figsize_columns, figsize=(15, 15)):
    """
    Display the reduction results in a grid layout.

    Parameters:
        results (list of tuples): Computation results as an array of tuples with the form
                                  [("<<plot title>>", [<<2-dimensional array>>])], where the
                                  2-dimensional array describes the x- & y-position of each
                                  data point & the title should describe the chosen parameter-values.
        figsize_columns (int): Number of columns which are used to layout the plotted graphs.
                               The number of rows is depending on the number of 'results' and
                               the given 'figsize_columns'.
        figsize (tuple, optional): Adjusts the width/height of the individual graphs, in case
                                   the scaling is different for different computations. Default
                                   is (15, 15).

    Returns:
        None
    """
    figsize_rows = math.ceil(len(results) / figsize_columns)
    fig = plt.subplots(figsize_rows, figsize_columns, figsize=figsize)
    # fig.tight_layout(pad=3)

    for i in range(len(results)):
        plt.subplot(figsize_rows, figsize_columns, i + 1).set_title(results[i][0])

        if len(results[i][1][0]) <= 2:
            plt.subplot(figsize_rows, figsize_columns, i + 1).scatter(
                results[i][1][:, 0], results[i][1][:, 1], alpha=0.4
            )
        else:
            plt.subplot(figsize_rows, figsize_columns, i + 1).scatter(
                results[i][1][:, 0],
                results[i][1][:, 1],
                c=results[i][1][:, 2],
                alpha=0.4,
            )


def display_graphs(
    results: list[EmbeddingObj],
    figsize_columns: int,
    labels: np.ndarray,
    figsize: tuple[int, int] = (15, 15),
    cmap: plt.cm = plt.cm.Accent,
    edge_cmap: plt.cm = plt.cm.plasma,
    show_cbar: bool = True,
    cbar_labels: Optional[list[str]] = None,
    show_edges: bool = True,
) -> None:
    figsize_rows = math.ceil(len(results) / figsize_columns)
    fig, axs = plt.subplots(figsize_rows, figsize_columns, figsize=figsize)

    if show_cbar:
        if cbar_labels is None:
            edge_weights = []
            for result in results:
                edge_weights.append(result.edge_weights)
            sm = plt.cm.ScalarMappable(cmap=edge_cmap)
            # sm.set_array(edge_weights)
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap)
            # sm.set_array(cbar_labels)

    if len(results) == 1:
        nx.draw(
            results[0].sim_graph,
            pos=results[0].embedding,
            node_size=30,
            node_color=labels if len(labels.shape) == 1 else labels[0],
            edge_color=results[0].edge_weights if show_edges else "white",
            edge_vmin=0,
            edge_vmax=1,
            width=0.4,
            alpha=0.6,
            edge_cmap=edge_cmap,
            cmap=cmap,
        )

        plt.title(results[0].title, fontsize=10)

        if show_cbar:
            cbar = fig.colorbar(sm, ax=axs, shrink=0.8)

            if cbar_labels is not None:
                # cbar.set_ticks([norm(v) for v in range(len(cbar_labels))])
                cbar.set_ticklabels(cbar_labels)

    else:
        axs = axs.flatten()

        for i in range(len(axs)):
            if i < len(results):
                nx.draw(
                    results[i].sim_graph,
                    ax=axs[i],
                    pos=results[i].embedding,
                    node_size=30,
                    node_color=labels if len(labels.shape) == 1 else labels[i],
                    edge_color=results[i].edge_weights if show_edges else "white",
                    edge_vmin=0,
                    edge_vmax=1,
                    width=0.4,
                    alpha=0.6,
                    edge_cmap=edge_cmap,
                    cmap=cmap,
                )

                axs[i].set_title(results[i].title, fontsize=10)

                if show_cbar:
                    cbar = fig.colorbar(sm, ax=axs[i], shrink=0.8)

                    if cbar_labels is not None:
                        cbar.set_ticklabels(cbar_labels)
            else:
                fig.delaxes(axs[i])
