import math
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.figure_factory as ff
import seaborn as sb
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import MinMaxScaler

import processing
from embedding_obj import EmbeddingObj


def display_graphs(
    results: list[EmbeddingObj],
    figsize_columns: int = 2,
    figsize: tuple[int, int] = (15, 15),
    cmap: plt.cm = plt.cm.Accent,
    edge_cmap: plt.cm = plt.cm.plasma,
    show_cbar: bool = True,
    cbar_labels: Optional[list[str]] = None,
    show_edges: bool = True,
    show_partition_centers: bool = False,
    node_labels: Optional[str] = None,
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

    axs = [axs] if len(results) == 1 else axs.flatten()

    for i in range(len(axs)):
        if i < len(results):
            graph = results[i].sim_graph.copy()
            positions = results[i].embedding.copy()
            node_sizes = [30] * graph.number_of_nodes()
            node_colors = ["blue"] * graph.number_of_nodes()
            edge_colors = ["white"] * graph.number_of_edges()

            # add node labels (colors), if provided
            if node_labels is not None:
                node_colors = list(nx.get_node_attributes(graph, node_labels).values())
            elif results[i].labels is not None:
                node_colors = [results[i].labels[n] for n in graph.nodes()]

            if show_edges:
                edge_colors = results[i].edge_weights

            # add partition centers to the graph (with dedicated size, color and position)
            if show_partition_centers and results[i].partition_centers is not None:
                start_idx = len(graph.nodes())
                end_idx = len(graph.nodes()) + len(results[i].partition_centers)
                node_idx = list(range(start_idx, end_idx))

                graph.add_nodes_from(node_idx)

                node_colors += [0] * len(results[i].partition_centers)
                node_sizes += [120] * len(results[i].partition_centers)

                center_dict = dict(zip(node_idx, results[i].partition_centers.values()))
                positions.update(center_dict)

            nx.draw(
                graph,
                ax=axs[i],
                pos=positions,
                node_size=node_sizes,
                node_color=node_colors,
                edge_color=edge_colors,
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


def plot_community_graphs(
    results: list[EmbeddingObj],
    figsize_columns: int = 2,
    figsize: tuple[int, int] = (15, 15),
    cmap: plt.cm = plt.cm.Accent,
    edge_cmap: plt.cm = plt.cm.plasma,
    show_partition_centers: bool = False,
    node_labels: Optional[str] = None,
    community_ids: Optional[list[int]] = None,
    boundary_edges: bool = False,
) -> None:
    figsize_rows = math.ceil(len(results) / figsize_columns)
    fig, axs = plt.subplots(figsize_rows, figsize_columns, figsize=figsize)

    axs = [axs] if len(results) == 1 else axs.flatten()

    for i in range(len(axs)):
        if i < len(results):
            graph = nx.Graph()
            graph.add_nodes_from(results[i].sim_graph.nodes(data=True))

            positions = results[i].embedding.copy()
            node_sizes = [30] * results[i].sim_graph.number_of_nodes()
            node_colors = ["blue"] * results[i].sim_graph.number_of_nodes()
            edge_colors = []

            # add node labels (colors), if provided
            if node_labels is not None:
                node_colors = list(nx.get_node_attributes(graph, node_labels).values())
            elif results[i].labels is not None:
                node_colors = [results[i].labels[n] for n in graph.nodes()]

            partition_subgraphs, _, partition_boundary_neighbors = (
                processing.compute_community_graphs(
                    results[i], boundary_edges=boundary_edges
                )
            )

            if community_ids is not None:
                # Filter the subgraphs based on the specified community IDs
                partition_subgraphs = {
                    k: v for k, v in partition_subgraphs.items() if k in community_ids
                }

                for community_id in community_ids:
                    graph.add_edges_from(
                        partition_subgraphs[community_id].edges(data=True)
                    )

                    for u, v in partition_subgraphs[community_id].edges():
                        if (
                            u in partition_boundary_neighbors[community_id]
                            or v in partition_boundary_neighbors[community_id]
                        ):
                            edge_colors.append(-1)
                        else:
                            edge_colors.append(node_colors[u])

            else:
                # Add all subgraphs to the main graph
                for _, subgraph in partition_subgraphs.items():
                    graph.add_edges_from(subgraph.edges(data=True))

            # add partition centers to the graph (with dedicated size, color and position)
            if show_partition_centers and results[i].partition_centers is not None:
                start_idx = len(graph.nodes())
                end_idx = len(graph.nodes()) + len(results[i].partition_centers)
                node_idx = list(range(start_idx, end_idx))

                graph.add_nodes_from(node_idx)

                node_colors += [0] * len(results[i].partition_centers)
                node_sizes += [120] * len(results[i].partition_centers)

                center_dict = dict(zip(node_idx, results[i].partition_centers.values()))
                positions.update(center_dict)

            nx.draw(
                graph,
                ax=axs[i],
                pos=positions,
                node_size=node_sizes,
                node_color=node_colors,
                edge_color=edge_colors,
                edge_vmin=0,
                edge_vmax=1,
                width=0.4,
                alpha=1.0,
                edge_cmap=edge_cmap,
                cmap=cmap,
            )

            axs[i].set_title(results[i].title, fontsize=10)

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
    sim_lowdim = pdist(np.array(list(embedding.embedding.values())))

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
        id_vars="Modified Version", var_name="Metric", value_name="Score"
    )
    sb.set_style("whitegrid", {"axes.grid": False})
    ax = sb.lineplot(
        data=df_melted,
        x="Modified Version",
        y="Score",
        hue="Metric",
        palette="muted",
        marker="o",
    )

    sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


def plot_metrics_report_new(
    data: pd.DataFrame, dual_axis_feature: str = "m_kruskal_stress"
) -> None:
    # Entferne das Jaccard-Merkmal
    data_clean = data.drop("metric_jaccard (size)", axis=1)

    data_clean = data_clean.melt(
        id_vars="marker", var_name="Feature", value_name="Value"
    )

    # # Setze Marker-Spalte als Index
    # data_clean = data_clean.set_index("marker")

    # Trenne in primäre und sekundäre Features
    primary_features = [col for col in data_clean.columns if col != dual_axis_feature]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot für primäre Features
    ax1.plot(
        data_clean,
        marker="o",
        linewidth=2,
    )

    ax1.set_ylabel("Metrics Score", fontsize=12)
    ax1.set_xlabel("Modified Version", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.tick_params(axis="x", rotation=45)

    # Sekundäre Y-Achse für das spezifizierte Feature
    ax2 = ax1.twinx()
    ax2.plot(
        data_clean.index,
        data_clean[dual_axis_feature],
        label=dual_axis_feature,
        marker="o",
        linewidth=2,
        linestyle="--",
    )
    ax2.set_ylabel(f"Metrics Score ({dual_axis_feature})", fontsize=12)
    ax2.tick_params(axis="y", rotation=45)

    # Legenden kombinieren
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        frameon=False,
    )

    plt.title("Metrics Report", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()


def plot_pos_movements(
    source: EmbeddingObj,
    target: EmbeddingObj,
    figsize: tuple[int, int] = (15, 15),
    filtered_communities: Optional[list[int]] = None,
    community_colors: bool = False,
    community_centers: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    if filtered_communities is None:
        coords_source = np.array(list(source.embedding.values()))
        coords_target = np.array(list(target.embedding.values()))
    else:
        # Filter nodes based on the specified communities
        filtered_node_ids = [
            node
            for node, community in target.sim_graph.nodes(data="community")
            if community in filtered_communities
        ]

        # Filter embeddings to only include nodes from the specified communities
        filtered_source_dict = {
            node: source.embedding[node] for node in filtered_node_ids
        }
        filtered_target_dict = {
            node: target.embedding[node] for node in filtered_node_ids
        }

        coords_source = np.array(list(filtered_source_dict.values()))
        coords_target = np.array(list(filtered_target_dict.values()))

    u = coords_target[:, 0] - coords_source[:, 0]
    v = coords_target[:, 1] - coords_source[:, 1]

    fig, ax = plt.subplots(figsize=figsize)

    if community_colors:
        if filtered_communities is not None:
            # Filter the community dictionary to only include the specified communities
            community_dict = {
                node: target.sim_graph.nodes[node]["community"]
                for node in filtered_node_ids
            }
        else:
            community_dict = nx.get_node_attributes(target.sim_graph, "community")

        plt.quiver(
            coords_source[:, 0],
            coords_source[:, 1],
            u,
            v,
            list(community_dict.values()),
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.001,
            headwidth=4,
            alpha=0.5,
        )
    else:
        plt.quiver(
            coords_source[:, 0],
            coords_source[:, 1],
            u,
            v,
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.001,
            headwidth=4,
            alpha=0.5,
        )

    if community_centers:
        if filtered_communities is not None:
            coords_community_centers = [
                coords
                for community, coords in target.partition_centers.items()
                if community in filtered_communities
            ]
        else:
            coords_community_centers = target.partition_centers.values()

        coords_community_centers = np.array(coords_community_centers)

        ax.scatter(
            coords_community_centers[:, 0],
            coords_community_centers[:, 1],
            # c=list(set(community_dict.keys())) if community_colors else "black",
        )

    all_x = np.concatenate([coords_source[:, 0], coords_target[:, 0]])
    all_y = np.concatenate([coords_source[:, 1], coords_target[:, 1]])
    ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
    ax.set_ylim(all_y.min() - 0.5, all_y.max() + 0.5)

    ax.set_title(
        f"Position movements from '{source.title}' to '{target.title}'", fontsize=10
    )

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_pos_movements_px(
    source: EmbeddingObj,
    target: EmbeddingObj,
) -> None:
    x = np.array(list(source.embedding.values()))
    y = np.array(list(target.embedding.values()))

    u = y[:, 0] - x[:, 0]
    v = y[:, 1] - x[:, 1]

    # Create quiver figure
    fig = ff.create_quiver(
        x[:, 0],
        y[:, 1],
        u,
        v,
        scale=1,
        name="quiver",
        line_width=1.5,
    )

    fig.update_xaxes(
        visible=False, showgrid=False, zeroline=False, showticklabels=False
    )
    fig.update_yaxes(
        visible=False, showgrid=False, zeroline=False, showticklabels=False
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        template="presentation",
        height=1000,
        width=1000,
    )

    fig.show()
