import math
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sb
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import MinMaxScaler

from embedding_obj import EmbeddingObj  # type: ignore


def display_graphs(
    results: list[EmbeddingObj],
    figsize_columns: int = 2,
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
            node_color=[results[0].labels[n] for n in results[0].sim_graph.nodes()]
            if results[0].labels is not None
            else "blue",
            # edge_color=results[0].edge_weights if show_edges else "white",
            edge_vmin=0,
            edge_vmax=1,
            width=0.4,
            alpha=1.0,
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
                    node_color=[
                        results[i].labels[n] for n in results[i].sim_graph.nodes()
                    ]
                    if results[i].labels is not None
                    else "blue",
                    edge_color=results[i].edge_weights if show_edges else "white",
                    edge_vmin=0,
                    edge_vmax=1,
                    width=0.4,
                    alpha=1.0,
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


def compute_shepard_curve(
    original_similarities: ArrayLike, transformed_similarities: ArrayLike
) -> tuple[list[float], npt.NDArray[np.float32]]:
    # Sort values based on transformed similarities
    sorted_indices = np.argsort(transformed_similarities)
    sorted_transformed = transformed_similarities[sorted_indices]
    sorted_original = original_similarities[sorted_indices]

    # Apply isotonic regression to enforce monotonicity
    iso_reg = IsotonicRegression(increasing=True)
    shepard_curve = iso_reg.fit_transform(sorted_transformed, sorted_original)

    return sorted_transformed.tolist(), shepard_curve


def plot_shepard_diagram(
    highdim_df: pd.DataFrame,
    embedding: EmbeddingObj,
    target_features: list[str],
    show_stress: bool = True,
) -> None:
    highdim_df_filtered = highdim_df[target_features].values.reshape(-1, 1)

    sim_highdim = pdist(highdim_df_filtered)
    sim_lowdim = pdist(np.array(embedding.embedding.values()))

    scaler = MinMaxScaler()
    sim_highdim = scaler.fit_transform(sim_highdim.reshape(-1, 1)).flatten()
    sim_lowdim = scaler.fit_transform(sim_lowdim.reshape(-1, 1)).flatten()

    sim_data_df = pd.DataFrame(
        {
            "Transformed Similarity": sim_lowdim,
            "Original Similarity": sim_highdim,
        }
    )

    # Plot the Shepard diagram using Seaborn
    fig = sb.jointplot(
        x="Transformed Similarity",
        y="Original Similarity",
        data=sim_data_df,
        kind="hist",
    )

    ax = fig.ax_joint
    # sb.lineplot(x=[0, 1], y=[0, 1], color="red", linestyle="--")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--")

    if show_stress:
        shepard_x, shepard_y = compute_shepard_curve(
            sim_data_df["Original Similarity"], sim_data_df["Transformed Similarity"]
        )
        # sb.lineplot(x=shepard_x, y=shepard_y, color="blue")
        ax.plot(shepard_x, shepard_y, color="blue")

    plt.suptitle(f"Shepard Diagram: Similarity for '{str(target_features)}'", y=1.02)
    plt.show()


def plot_metrics_report(data: pd.DataFrame) -> None:
    df_melted = data.drop("metric_jaccard (size)", axis=1).melt(
        id_vars="marker", var_name="Feature", value_name="Value"
    )
    sb.set_style("whitegrid", {"axes.grid": False})
    sb.lineplot(data=df_melted, x="marker", y="Value", hue="Feature", palette="muted",
                marker="o")