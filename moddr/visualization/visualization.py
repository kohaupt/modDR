import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sb

from .. import processing
from ..embedding_state import EmbeddingState


def display_embeddings(
    embeddings: list[EmbeddingState],
    figsize_columns: int = 2,
    figsize: tuple[int, int] = (15, 15),
    show_edges: bool = True,
    node_cmap: plt.cm = plt.cm.viridis,
    edge_cmap: plt.cm = plt.cm.viridis,
    show_edge_cbar: bool = False,
    show_node_cbar: bool = False,
    show_title: bool = True,
    show_community_centers: bool = False,
    node_color_attribute: str | None = None,
    node_label_attribute: str | None = None,
) -> plt.Figure:
    """Display multiple embeddings as network graphs in a subplot grid.

    This function creates a visualization of multiple EmbeddingState objects,
    showing nodes, edges, and various attributes in a customizable grid layout.

    Args:
        embeddings (list[EmbeddingState]): List of EmbeddingState objects to visualize.
        figsize_columns (int): Number of columns in the subplot grid. Default is 2.
        figsize (tuple[int, int]): Size of each individual subplot (width, height).
            Default is (15, 15).
        show_edges (bool): Whether to display edges based on the graph of the EmbeddingState. Default is True.
        node_cmap (plt.cm): Colormap for node colors. Default is plt.cm.viridis.
        edge_cmap (plt.cm): Colormap for edge colors. Default is plt.cm.viridis.
        show_edge_cbar (bool): Whether to show colorbar for edge weights. Default is False.
        show_node_cbar (bool): Whether to show colorbar for node colors. Default is False.
        show_title (bool): Whether to display titles for each subplot. Default is True.
        show_community_centers (bool): Whether to display community centers as larger nodes.
            Default is False.
        node_color_attribute (str | None): Graph node attribute to use for node coloring.
            If None, tries to use labels from EmbeddingState. Otherwise, falls back to default colors. Default is None.
        node_label_attribute (str | None): Graph node attribute to use for node labels.
            Use "id" for node IDs or specify custom attribute name. Default is None.

    Returns:
        plt.Figure: matplotlib Figure object containing the visualization.
    """
    # compute figure sizes and create subplots
    figsize_rows = math.ceil(len(embeddings) / figsize_columns)
    fig_width = figsize_columns * figsize[0]
    fig_height = figsize_rows * figsize[1]
    fig, axs = plt.subplots(
        figsize_rows, figsize_columns, figsize=(fig_width, fig_height)
    )

    axs = [axs] if len(embeddings) == 1 else axs.flatten()

    for i in range(len(axs)):
        if i < len(embeddings):
            graph = embeddings[i].graph.copy()
            positions = embeddings[i].embedding.copy()
            node_sizes = [20] * graph.number_of_nodes()
            edge_colors = np.array(
                list(nx.get_edge_attributes(embeddings[i].graph, "weight", 1).values())
            )

            # add node labels (colors), if provided
            node_colors = []
            if node_color_attribute is not None:
                node_colors = list(
                    nx.get_node_attributes(graph, node_color_attribute).values()
                )
                if len(node_colors) == 0:
                    print(
                        f"WARNING for embedding {embeddings[i].obj_id}: Node color attribute '{node_color_attribute}' not found in graph nodes. "  # noqa: E501
                        f"Falling back to labels or default color."
                    )

            # fallback to labels or default color if no node color attribute is set
            if len(node_colors) == 0:
                if (
                    embeddings[i].labels is not None
                    and len(embeddings[i].labels) == graph.number_of_nodes()
                ):
                    node_colors = [embeddings[i].labels[n] for n in graph.nodes()]
                else:
                    node_colors = [0] * graph.number_of_nodes()

            # add partition centers to graph and positions
            if show_community_centers and embeddings[i].community_centers is not None:
                start_idx = len(graph.nodes())
                end_idx = len(graph.nodes()) + len(embeddings[i].community_centers)
                node_idx = list(range(start_idx, end_idx))
                graph.add_nodes_from(node_idx)

                node_colors += [0] * len(embeddings[i].community_centers)
                node_sizes += [140] * len(embeddings[i].community_centers)

                center_dict = dict(
                    zip(
                        node_idx, embeddings[i].community_centers.values(), strict=False
                    )
                )
                positions.update(center_dict)

            nx.draw(
                graph,
                ax=axs[i],
                pos=positions,
                node_size=node_sizes,
                node_color=node_colors,
                edge_color=edge_colors,
                edgelist=[] if not show_edges else graph.edges(),
                width=0.4,
                alpha=1.0,
                edge_cmap=edge_cmap,
                cmap=node_cmap,
            )

            if show_title:
                axs[i].set_title(
                    f"ID: {embeddings[i].obj_id} \n{embeddings[i].title}", fontsize=10
                )

            if node_label_attribute is not None:
                # use default implementation of function if label is id
                if node_label_attribute == "id":
                    nx.draw_networkx_labels(
                        graph,
                        positions,
                        font_size=12,
                        horizontalalignment="left",
                        verticalalignment="bottom",
                        ax=axs[i],
                    )
                # use specific node attributes otherwise
                else:
                    label_attributes = nx.get_node_attributes(
                        graph, node_label_attribute
                    )
                    if label_attributes is None or len(label_attributes) == 0:
                        print(
                            f"WARNING for embedding {embeddings[i].obj_id}: Node label attribute '{node_label_attribute}' not found in graph nodes. "  # noqa: E501
                            f"Skipping node labels."
                        )
                    else:
                        nx.draw_networkx_labels(
                            graph,
                            positions,
                            labels=label_attributes,
                            font_size=12,
                            horizontalalignment="left",
                            verticalalignment="bottom",
                            ax=axs[i],
                        )

            if show_node_cbar:
                vmin = min(node_colors)
                vmax = max(node_colors)

                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=norm)
                sm.set_array([])

                cbar = fig.colorbar(sm, ax=axs[i], shrink=0.8)
                ticks = np.linspace(vmin, vmax, 5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
                cbar.set_label("Node Values")

            if show_edge_cbar:
                vmin = min(edge_colors)
                vmax = max(edge_colors)

                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
                sm.set_array([])

                cbar = fig.colorbar(sm, ax=axs[i], shrink=0.8)
                ticks = np.linspace(vmin, vmax, 5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
                cbar.set_label("Edge Weights")

        # clean up empty subplots if figsize_columns do not divide evenly
        else:
            fig.delaxes(axs[i])

    return fig


def plot_community_graphs(
    embeddings: list[EmbeddingState],
    figsize_columns: int = 2,
    figsize: tuple[int, int] = (15, 15),
    node_cmap: plt.cm = plt.cm.viridis,
    edge_cmap: plt.cm = plt.cm.viridis,
    only_communities: bool = False,
    community_ids: list[int] | None = None,
    show_boundary_edges: bool = False,
    unify_edge_colors: bool = False,
    show_title: bool = True,
    show_community_centers: bool = False,
    node_color_attribute: str | None = None,
    node_label_attribute: str | None = None,
) -> plt.Figure:
    """Display community-focused visualizations of embeddings.

    This function creates visualizations that emphasize community structure,
    allowing filtering by specific communities and showing boundary connections.

    Args:
        embeddings (list[EmbeddingState]): List of EmbeddingState objects to visualize.
        figsize_columns (int): Number of columns in the subplot grid. Default is 2.
        figsize (tuple[int, int]): Size of each individual subplot (width, height).
            Default is (15, 15).
        node_cmap (plt.cm): Colormap for node colors. Default is plt.cm.viridis.
        edge_cmap (plt.cm): Colormap for edge colors. Default is plt.cm.viridis.
        only_communities (bool): If True, show only specified communities in community_ids.
            If False, show all nodes with highlighted communities. Default is False.
        community_ids (list[int] | None): List of specific community IDs to display.
            If None and only_communities=True, shows all communities. Default is None.
        show_boundary_edges (bool): Whether to include edges connecting to nodes
            outside each community. Default is False.
        unify_edge_colors (bool): If True, color edges by source node community.
            If False, use edge weights for coloring. Default is False.
        show_title (bool): Whether to display titles for each subplot. Default is True.
        show_community_centers (bool): Whether to display community centers as larger nodes.
            Default is False.
        node_color_attribute (str | None): Graph node attribute to use for node coloring.
            Default is None.
        node_label_attribute (str | None): Graph node attribute to use for node labels.
            Default is None.

    Returns:
        plt.Figure: matplotlib Figure object containing the community visualization.
    """
    if only_communities and community_ids is None:
        print("WARNING: Community IDs not specified. Plotting all communities.")

    # compute figure sizes and create subplots
    figsize_rows = math.ceil(len(embeddings) / figsize_columns)
    fig_width = figsize_columns * figsize[0]
    fig_height = figsize_rows * figsize[1]
    fig, axs = plt.subplots(
        figsize_rows, figsize_columns, figsize=(fig_width, fig_height)
    )

    axs = [axs] if len(embeddings) == 1 else axs.flatten()

    for i in range(len(axs)):
        if i < len(embeddings):
            emb_community_ids = community_ids
            partition_subgraphs, _, boundary_neighbors_dict = (
                processing.compute_community_graphs(
                    embeddings[i], boundary_neighbors=show_boundary_edges
                )
            )

            if show_boundary_edges:
                for community, subgraph in partition_subgraphs.items():
                    neighbor_community_dict = {
                        n: community for n in boundary_neighbors_dict[community]
                    }
                    nx.set_node_attributes(
                        subgraph, neighbor_community_dict, "community"
                    )

            graph = nx.Graph()
            positions = {}
            if only_communities and emb_community_ids is None:
                emb_community_ids = list(embeddings[i].partition.keys())

            if not only_communities:
                # add all nodes and their positions to the graph
                graph.add_nodes_from(embeddings[i].graph.nodes(data=True))
                positions = embeddings[i].embedding.copy()

            if emb_community_ids is not None:
                # filter the subgraphs based on the specified community IDs
                filtered_community_ids = [
                    community_id
                    for community_id in emb_community_ids
                    if community_id in partition_subgraphs
                ]
                if len(filtered_community_ids) != len(emb_community_ids):
                    print(
                        f"WARNING: Community IDs {set(emb_community_ids) - set(filtered_community_ids)} are not present in embedding '{embeddings[i].title}'. "  # noqa: E501
                        f"Only plotting available communities: {filtered_community_ids}."  # noqa: E501
                    )

                # filter the subgraphs based on the filtered community IDs
                partition_subgraphs = {
                    k: v
                    for k, v in partition_subgraphs.items()
                    if k in filtered_community_ids
                }

                # add the filtered subgraphs to the main graph
                for community_id in filtered_community_ids:
                    graph = nx.compose(graph, partition_subgraphs[community_id])
                    if only_communities:
                        for node in partition_subgraphs[community_id].nodes():
                            positions[node] = embeddings[i].embedding[node]

            else:
                # add all subgraphs to the main graph
                for _, subgraph in partition_subgraphs.items():
                    graph = nx.compose(graph, subgraph)

            node_sizes_dict = {n: 20 for n in graph.nodes()}
            node_colors_dict = {}

            if unify_edge_colors:
                edge_colors = [graph.nodes[u]["community"] for u, v in graph.edges()]
            else:
                edge_colors = list(nx.get_edge_attributes(graph, "weight").values())

            # add node labels (colors), if provided
            if node_color_attribute is not None:
                node_colors_dict = nx.get_node_attributes(graph, node_color_attribute)
                if len(node_colors_dict.items()) == 0:
                    print(
                        f"WARNING for embedding {embeddings[i].obj_id}: Node color attribute '{node_color_attribute}' not found in graph nodes. "  # noqa: E501
                        f"Falling back to labels or default color."
                    )

            if len(node_colors_dict.items()) == 0:
                if (
                    embeddings[i].labels is not None
                    and len(embeddings[i].labels) == graph.number_of_nodes()
                ):
                    node_colors_dict = {
                        n: embeddings[i].labels[n] for n in graph.nodes()
                    }
                    print(
                        f"INFO for embedding {embeddings[i].obj_id}: Using labels as node colors."  # noqa: E501
                    )
                else:
                    node_colors_dict = {n: 0 for n in graph.nodes()}

            # add partition centers to graph and positions
            if show_community_centers and embeddings[i].community_centers is not None:
                if emb_community_ids is None:
                    partition_center_ids = list(embeddings[i].partition.keys())
                else:
                    partition_center_ids = emb_community_ids

                filtered_centers = {
                    cid: embeddings[i].community_centers[cid]
                    for cid in partition_center_ids
                    if cid in embeddings[i].community_centers
                }

                start_idx = len(graph.nodes())
                end_idx = len(graph.nodes()) + len(filtered_centers)
                node_idx = list(range(start_idx, end_idx))
                graph.add_nodes_from(node_idx)

                for n in node_idx:
                    node_sizes_dict[n] = 140
                    node_colors_dict[n] = 0

                center_dict = dict(
                    zip(node_idx, filtered_centers.values(), strict=False)
                )
                positions.update(center_dict)

            node_sizes = [node_sizes_dict[n] for n in graph.nodes()]
            node_colors = [node_colors_dict[n] for n in graph.nodes()]

            nx.draw(
                graph,
                ax=axs[i],
                pos=positions,
                node_size=node_sizes,
                node_color=node_colors,
                edge_color=edge_colors,
                width=0.4,
                alpha=1.0,
                edge_cmap=edge_cmap,
                cmap=node_cmap,
            )

            if show_title:
                axs[i].set_title(
                    f"ID: {embeddings[i].obj_id} \n{embeddings[i].title}", fontsize=10
                )

            if node_label_attribute is not None:
                # use default implementation of function if label is id
                if node_label_attribute == "id":
                    nx.draw_networkx_labels(
                        graph,
                        positions,
                        font_size=12,
                        horizontalalignment="left",
                        verticalalignment="bottom",
                        ax=axs[i],
                    )
                # use specific node attributes otherwise
                else:
                    label_attributes = nx.get_node_attributes(
                        graph, node_label_attribute
                    )
                    if label_attributes is None or len(label_attributes) == 0:
                        print(
                            f"WARNING for embedding {embeddings[i].obj_id}: Node label attribute '{node_label_attribute}' not found in graph nodes. Skipping node labels."  # noqa: E501
                        )
                    else:
                        nx.draw_networkx_labels(
                            graph,
                            positions,
                            labels=label_attributes,
                            font_size=12,
                            horizontalalignment="left",
                            verticalalignment="bottom",
                            ax=axs[i],
                        )

        # clean up empty subplots if figsize_columns do not divide evenly
        else:
            fig.delaxes(axs[i])

    return fig


def plot_metrics_report(
    data_df: pd.DataFrame,
    division: list[any] | None = None,
    division_label: str | None = None,
    label_height: float = 0.1,
) -> plt.Figure:
    """Create a line plot visualization of metrics across different embeddings.

    This function generates a comprehensive metrics report showing how different
    evaluation metrics vary across embedding objects, with optional grouping
    by parameter divisions.

    Args:
        data_df (pd.DataFrame): DataFrame containing metrics data. Must include 'obj_id'
            column and numeric metric columns.
        division (list[any] | None): List of division values to group embeddings
            (e.g., resolution parameters). Length should match the number of embedding
            groups. Default is None.
        division_label (str | None): Label for the division parameter (e.g., "Resolution").
            Required if division is provided. Default is None.
        label_height (float): Vertical position for division labels on the plot.
            Default is 0.1.

    Returns:
        plt.Figure: matplotlib Figure object containing the metrics visualization.

    Raises:
        ValueError: If input validation fails.

    Notes:
        - The function automatically adds background shading for different
          parameter groups when division is provided.
        - Each metric is plotted with a unique marker and color.
        - The plot includes grid lines and a legend positioned outside the plot area.
    """
    if data_df.empty:
        raise ValueError("WARNING: DataFrame is empty.")
    if "obj_id" not in data_df.columns:
        raise ValueError("WARNING: 'obj_id' column not found in DataFrame.")
    if data_df.select_dtypes(include="number").shape[1] != data_df.shape[1]:
        raise ValueError("WARNING: DataFrame contains non-numeric columns.")
    if division is not None and division_label is None:
        raise ValueError(
            "WARNING: 'division_label' must be provided if 'division' is specified."
        )

    df_melted = data_df.melt(id_vars="obj_id", var_name="Metric", value_name="Score")
    metrics = data_df.columns.drop("obj_id")

    x_label = "Object ID"
    y_label = "Score"
    legend_labels = metrics.tolist()

    marker_list = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H"]
    marker_map = {
        metric: marker_list[i % len(marker_list)] for i, metric in enumerate(metrics)
    }

    palette = sb.color_palette("colorblind", n_colors=len(metrics))
    color_map = {metric: palette[i] for i, metric in enumerate(metrics)}

    sb.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 5))
    obj_ids = sorted(data_df["obj_id"].unique())

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
                label_height,
                division_label + f" ${str(division[i // step]).replace('.', ',')}$",
                ha="center",
                va="center",
                fontsize=12,
                alpha=1,
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

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_xticks(obj_ids)
    ax.set_ylabel(y_label, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=1.0)
    ax.xaxis.grid(False)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        title=None,
        handles=handles,
        labels=legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=14,
    )
    sb.despine()

    plt.tight_layout()
    plt.show()

    return fig


def plot_pos_movements(
    reference: EmbeddingState,
    target: EmbeddingState,
    figsize: tuple[int, int] = (15, 15),
    filtered_communities: list[int] | None = None,
    community_colors: bool = False,
    community_centers: bool = False,
    plot_target_nodes: bool = False,
    show_title: bool = True,
    node_cmap: plt.cm = plt.cm.viridis,
) -> plt.Figure:
    """Visualize node position movements between two embeddings using arrows.

    This function creates a quiver plot showing how node positions change
    from a reference embedding to a target embedding, useful for analyzing
    the effects of layout algorithms or parameter changes.

    Args:
        reference (EmbeddingState): Source EmbeddingState showing initial positions.
        target (EmbeddingState): Target EmbeddingState showing modified positions.
        figsize (tuple[int, int]): Figure size (width, height). Default is (15, 15).
        filtered_communities (list[int] | None): List of community IDs to focus on.
            If None, shows movements for all nodes. Default is None.
        community_colors (bool): Whether to color arrows by target community membership.
            Default is False.
        community_centers (bool): Whether to display community centers as black dots.
            Default is False.
        plot_target_nodes (bool): Whether to show target node positions as scatter points.
            Default is False.
        show_title (bool): Whether to display a descriptive title. Default is True.
        node_cmap (plt.cm): Colormap for arrow and node colors. Default is plt.cm.viridis.

    Returns:
        plt.Figure: matplotlib Figure object containing the movement visualization.

    Notes:
        - Arrows point from reference positions to target positions.
        - Arrow length is scaled down when plot_target_nodes=True to avoid overlap.
        - The plot automatically adjusts axis limits to show all movements.
    """
    source_dict = dict(sorted(reference.embedding.items()))
    target_dict = dict(sorted(target.embedding.items()))

    if filtered_communities is None:
        coords_source = np.array(list(source_dict.values()))
        coords_target = np.array(list(target_dict.values()))
    else:
        # filter nodes based on the specified communities
        filtered_node_ids = [
            node
            for node, community in target.graph.nodes(data="community")
            if community in filtered_communities
        ]

        # filter embeddings to only include nodes from the specified communities
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
            # filter the community dictionary to only include the specified communities
            community_dict = {
                node: target.graph.nodes[node]["community"]
                for node in filtered_node_ids
            }
        else:
            community_dict = {
                node: target.graph.nodes[node]["community"]
                for node in list(target.embedding.keys())
            }

        plt.quiver(
            coords_source[:, 0],
            coords_source[:, 1],
            u,
            v,
            list(community_dict.values()),
            cmap=node_cmap,
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
            cmap=node_cmap,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.001,
            headwidth=6,
            alpha=1,
        )

    if community_centers:
        # plot all community centers if no communities are filtered
        if filtered_communities is None:
            coords_community_centers = list(target.community_centers.values())
        else:
            coords_community_centers = [
                coords
                for community, coords in target.community_centers.items()
                if community in filtered_communities
            ]

        coords_community_centers = np.array(coords_community_centers)

        ax.scatter(
            coords_community_centers[:, 0],
            coords_community_centers[:, 1],
            c="black",
            s=100,
        )

    if plot_target_nodes:
        ax.scatter(
            coords_target[:, 0],
            coords_target[:, 1],
            c=list(target.labels.values()),
            cmap=plt.cm.viridis,
            zorder=3,
            s=40,
        )

    all_x = np.concatenate([coords_source[:, 0], coords_target[:, 0]])
    all_y = np.concatenate([coords_source[:, 1], coords_target[:, 1]])
    ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
    ax.set_ylim(all_y.min() - 0.5, all_y.max() + 0.5)

    if show_title:
        ax.set_title(
            f"Position movements from '{reference.title}' to '{target.title}'",
            fontsize=10,
        )

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return fig
