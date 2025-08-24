import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sb
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import MinMaxScaler

import processing
from embeddingstate import EmbeddingState


def display_graphs(
    results: list[EmbeddingState],
    figsize_columns: int = 2,
    figsize: tuple[int, int] = (15, 15),
    cmap: plt.cm = plt.cm.viridis,
    edge_cmap: plt.cm = plt.cm.viridis,
    show_cbar: bool = True,
    cbar_labels: list[str] | None = None,
    show_edges: bool = True,
    show_partition_centers: bool = False,
    show_title: bool = True,
    node_colors: str | None = None,
    node_labels: str | None = None,
) -> plt.Figure:
    figsize_rows = math.ceil(len(results) / figsize_columns)
    fig_width = figsize_columns * figsize[0]
    fig_height = figsize_rows * figsize[1]
    fig, axs = plt.subplots(
        figsize_rows, figsize_columns, figsize=(fig_width, fig_height)
    )

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
            node_sizes = [20] * graph.number_of_nodes()
            edge_colors = results[i].edge_weights

            # add node labels (colors), if provided
            node_color_list = []
            if node_colors is not None:
                node_color_list = list(
                    nx.get_node_attributes(graph, node_colors).values()
                )

            if len(node_color_list) == 0:
                if results[i].labels is not None:
                    node_color_list = [results[i].labels[n] for n in graph.nodes()]
                else:
                    node_color_list = ["blue"] * graph.number_of_nodes()

            # add partition centers to the graph (with dedicated size, color and position)
            if show_partition_centers and results[i].partition_centers is not None:
                start_idx = len(graph.nodes())
                end_idx = len(graph.nodes()) + len(results[i].partition_centers)
                node_idx = list(range(start_idx, end_idx))

                graph.add_nodes_from(node_idx)

                node_colors += [0] * len(results[i].partition_centers)
                node_sizes += [120] * len(results[i].partition_centers)

                center_dict = dict(
                    zip(node_idx, results[i].partition_centers.values(), strict=False)
                )
                positions.update(center_dict)

            nx.draw(
                graph,
                ax=axs[i],
                pos=positions,
                node_size=node_sizes,
                node_color=node_color_list,
                edge_color=edge_colors,
                edgelist=[] if not show_edges else graph.edges(),
                width=0.4,
                alpha=1.0,
                edge_cmap=edge_cmap,
                cmap=cmap,
            )

            if show_title:
                axs[i].set_title(
                    f"ID: {results[i].obj_id} \n{results[i].title}", fontsize=10
                )

            if node_labels is not None:
                if node_labels == "id":
                    nx.draw_networkx_labels(
                        graph,
                        positions,
                        font_size=12,
                        horizontalalignment="left",
                        verticalalignment="bottom",
                        ax=axs[i],
                    )
                else:
                    nx.draw_networkx_labels(
                        graph,
                        positions,
                        nx.get_node_attributes(graph, node_labels),
                        font_size=12,
                        horizontalalignment="left",
                        verticalalignment="bottom",
                        ax=axs[i],
                    )

            if show_cbar:
                cbar = fig.colorbar(sm, ax=axs[i], shrink=0.8)

                if cbar_labels is not None:
                    cbar.set_ticklabels(cbar_labels)

            # axs[i].set_xlim(
            #     np.array(list(positions.values()))[:, 0].min() - 1,
            #     np.array(list(positions.values()))[:, 0].max() + 1,
            # )
            # axs[i].set_ylim(
            #     np.array(list(positions.values()))[:, 1].min() - 1,
            #     np.array(list(positions.values()))[:, 1].max() + 1,
            # )
            # axs[i].set_aspect("equal", adjustable="box")

        else:
            fig.delaxes(axs[i])

    return fig



def plot_community_graphs(
    results: list[EmbeddingObj],
    figsize_columns: int = 2,
    figsize: tuple[int, int] = (15, 15),
    cmap: plt.cm = plt.cm.viridis,
    edge_cmap: plt.cm = plt.cm.viridis,
    show_partition_centers: bool = False,
    only_communities: bool = False,
    node_labels: str | None = None,
    community_ids: list[int] | None = None,
    boundary_edges: bool = False,
    show_title: bool = True,
) -> plt.Figure:
    figsize_rows = math.ceil(len(results) / figsize_columns)
    fig_width = figsize_columns * figsize[0]
    fig_height = figsize_rows * figsize[1]
    fig, axs = plt.subplots(
        figsize_rows, figsize_columns, figsize=(fig_width, fig_height)
    )

    axs = [axs] if len(results) == 1 else axs.flatten()

    for i in range(len(axs)):
        if i < len(results):
            partition_subgraphs, _, _ = processing.compute_community_graphs(
                results[i], boundary_neighbors=boundary_edges
            )

            graph = nx.Graph()
            positions = {}
            if not only_communities:
                graph.add_nodes_from(results[i].sim_graph.nodes(data=True))
                positions = results[i].embedding.copy()

            if community_ids is not None:
                # Filter the subgraphs based on the specified community IDs
                filtered_community_ids = [
                    community_id
                    for community_id in community_ids
                    if community_id in partition_subgraphs
                ]
                if len(filtered_community_ids) != len(community_ids):
                    print(
                        f"Warning: Community IDs {set(community_ids) - set(filtered_community_ids)} "
                        f"are not present in embedding '{results[i].title}'. Only plotting available communities."
                    )

                partition_subgraphs = {
                    k: v
                    for k, v in partition_subgraphs.items()
                    if k in filtered_community_ids
                }

                for community_id in filtered_community_ids:
                    graph = nx.compose(graph, partition_subgraphs[community_id])
                    if only_communities:
                        for node in partition_subgraphs[community_id].nodes():
                            positions[node] = results[i].embedding[node]

            else:
                # Add all subgraphs to the main graph
                for _, subgraph in partition_subgraphs.items():
                    graph = nx.compose(graph, subgraph)

            node_sizes = [30] * graph.number_of_nodes()
            node_colors = ["blue"] * graph.number_of_nodes()
            edge_colors = []

            # add node labels (colors), if provided
            if node_labels is not None:
                node_colors = list(nx.get_node_attributes(graph, node_labels).values())
            elif results[i].labels is not None:
                node_colors = [results[i].labels[n] for n in graph.nodes()]

            edge_colors = [graph.nodes[u]["community"] for u, v in graph.edges()]

            # add partition centers to the graph (with dedicated size, color and position)
            if show_partition_centers and results[i].partition_centers is not None:
                start_idx = len(graph.nodes())
                end_idx = len(graph.nodes()) + len(results[i].partition_centers)
                node_idx = list(range(start_idx, end_idx))

                graph.add_nodes_from(node_idx)

                node_colors += [0] * len(results[i].partition_centers)
                node_sizes += [120] * len(results[i].partition_centers)

                center_dict = dict(
                    zip(node_idx, results[i].partition_centers.values(), strict=False)
                )
                positions.update(center_dict)

            nx.draw(
                graph,
                ax=axs[i],
                pos=positions,
                node_size=node_sizes,
                node_color=node_colors,
                edge_color=edge_colors,
                # edge_vmin=min(edge_colors) if edge_colors else 0,
                # edge_vmax=max(edge_colors) if edge_colors else 1,
                width=0.4,
                alpha=1.0,
                edge_cmap=edge_cmap,
                cmap=cmap,
            )

            if show_title:
                axs[i].set_title(
                    f"ID: {results[i].obj_id} \n{results[i].title}", fontsize=10
                )

        else:
            fig.delaxes(axs[i])

    return fig


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
    embedding: EmbeddingState,
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


def plot_metrics_report(
    data: pd.DataFrame, division: list[any] | None = None, save_path: str = None
) -> None:
    df_melted = data.melt(id_vars="obj_id", var_name="Metric", value_name="Score")

    metrics = data.columns.drop("obj_id")

    marker_list = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H"]
    marker_map = {
        metric: marker_list[i % len(marker_list)] for i, metric in enumerate(metrics)
    }

    palette = sb.color_palette("colorblind", n_colors=len(metrics))
    color_map = {metric: palette[i] for i, metric in enumerate(metrics)}

    sb.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 7))
    obj_ids = sorted(data["obj_id"].unique())

    if division is not None:
        step = (len(obj_ids)) // len(division)
        gray_shades = ["#f0f0f0", "#e0e0e0"]

        for i in range(1, len(obj_ids) - 1, step):
            xmin = obj_ids[i]
            xmax = obj_ids[i + step] - 1 if i + step < len(obj_ids) else obj_ids[-1]
            shade_color = gray_shades[(i // step) % len(gray_shades)]
            ax.axvspan(xmin, xmax, color=shade_color, alpha=0.5, zorder=0)
            ax.text(
                (xmin + xmax) / 2,
                0.25,
                f"Community size: \n {division[i // step]}",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
                alpha=0.8,
            )

    for metric in metrics:
        subset = df_melted[df_melted["Metric"] == metric]
        ax.plot(
            subset["obj_id"],
            subset["Score"],
            label=metric,
            marker=marker_map[metric],
            color=color_map[metric],
            linestyle="solid",
            linewidth=1.2,
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=color_map[metric],
        )

    ax.set_xlabel("Object ID", fontsize=12)
    ax.set_xticks(obj_ids)
    ax.set_ylabel("Score", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=1.0)
    ax.xaxis.grid(False)
    ax.legend(title=None, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    sb.despine()

    plt.tight_layout()
    plt.show()

    return fig


def plot_metrics_report_plotly(data: pd.DataFrame) -> None:
    df_melted = data.melt(id_vars="obj_id", var_name="Metric", value_name="Score")
    fig = px.line(
        df_melted,
        x="obj_id",
        y="Score",
        color="Metric",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    fig.update_layout(
        template="simple_white",
        font=dict(size=12),
        width=600,
        height=400,
        legend=dict(title=None, x=1.02, y=1, borderwidth=0, font=dict(size=10)),
        margin=dict(l=40, r=40, t=20, b=40),
    )

    fig.show()


def plot_pos_movements(
    source: EmbeddingState,
    target: EmbeddingState,
    figsize: tuple[int, int] = (15, 15),
    filtered_communities: list[int] | None = None,
    community_colors: bool = False,
    community_centers: bool = False,
    plot_target_nodes: bool = False,
    show_title: bool = True,
) -> plt.Figure:
    source_dict = dict(sorted(source.embedding.items()))
    target_dict = dict(sorted(target.embedding.items()))

    if filtered_communities is None:
        coords_source = np.array(list(source_dict.values()))
        coords_target = np.array(list(target_dict.values()))
    else:
        # Filter nodes based on the specified communities
        filtered_node_ids = [
            node
            for node, community in target.sim_graph.nodes(data="community")
            if community in filtered_communities
        ]

        # Filter embeddings to only include nodes from the specified communities
        filtered_source_dict = {node: source_dict[node] for node in filtered_node_ids}
        filtered_target_dict = {node: target_dict[node] for node in filtered_node_ids}

        coords_source = np.array(list(filtered_source_dict.values()))
        coords_target = np.array(list(filtered_target_dict.values()))

    # shorten the vectors to avoid overlapping with target nodes
    scale_factor = 1.0
    if plot_target_nodes:
        scale_factor = 0.95

    u = (coords_target[:, 0] - coords_source[:, 0]) * scale_factor
    v = (coords_target[:, 1] - coords_source[:, 1]) * scale_factor

    fig, ax = plt.subplots(figsize=figsize)

    if community_colors:
        if filtered_communities is not None:
            # Filter the community dictionary to only include the specified communities
            community_dict = {
                node: target.sim_graph.nodes[node]["community"]
                for node in filtered_node_ids
            }
        else:
            community_dict = {
                node: target.sim_graph.nodes[node]["community"]
                for node in list(target.embedding.keys())
            }

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
            headwidth=6,
            alpha=1,
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
            headwidth=6,
            alpha=1,
        )

    if community_centers:
        if filtered_communities is None:
            coords_community_centers = list(target.partition_centers.values())
        else:
            coords_community_centers = [
                coords
                for community, coords in target.partition_centers.items()
                if community in filtered_communities
            ]

        coords_community_centers = np.array(coords_community_centers)

        ax.scatter(
            coords_community_centers[:, 0],
            coords_community_centers[:, 1],
            c=list(set(community_dict.keys())) if community_colors else "black",
        )

    if plot_target_nodes:
        ax.scatter(
            coords_target[:, 0],
            coords_target[:, 1],
            c=list(target.labels.values()),
            cmap=plt.cm.viridis,
            zorder=3,
        )

    all_x = np.concatenate([coords_source[:, 0], coords_target[:, 0]])
    all_y = np.concatenate([coords_source[:, 1], coords_target[:, 1]])
    ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
    ax.set_ylim(all_y.min() - 0.5, all_y.max() + 0.5)

    if show_title:
        ax.set_title(
            f"Position movements from '{source.title}' to '{target.title}'", fontsize=10
        )

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return fig


def plot_pos_movements_px(
    source: EmbeddingState,
    target: EmbeddingState,
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
