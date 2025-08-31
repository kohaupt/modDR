import copy
import math

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sb
from matplotlib.ticker import FuncFormatter

from moddr import EmbeddingState, evaluation


def export_to_gexf(data: EmbeddingState, folderpath: str = "/interim/") -> None:
    """Export an EmbeddingState graph to GEXF format for visualization in Gephi.

    This function converts the embedding data into a GEXF file that can be
    imported into Gephi for advanced graph visualization. Node positions
    are set based on the embedding coordinates, and node features are
    preserved as attributes.

    Args:
        data (EmbeddingState): The embedding state containing graph and
            embedding information.
        folderpath (str, optional): Relative folder path under data directory.
            Defaults to "/interim/".

    Note:
        - Minimum edge weight of 0.0001 is set for edges with zero weight
          (Gephi requirement)
        - Node positions are scaled by 1000 for better visualization
        - Labels are stored as 'feat' attribute for each node
        - Output file is named 'graph_new_feat_it2.gexf'
    """
    graph_gexf = data.graph

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


def export_plot(fig: plt.Figure, save_path: str):
    """Export a matplotlib figure to SVG format with LaTeX rendering.

    This function configures matplotlib to use LaTeX for text rendering
    and exports the figure as a high-quality SVG file suitable for
    academic publications.

    Args:
        fig (plt.Figure): The matplotlib figure to export.
        save_path (str): The file path where the SVG should be saved
            (without extension).

    Note:
        - Enables LaTeX text rendering with Computer Modern Roman font
        - Saves as SVG format with 300 DPI resolution
        - Uses tight bounding box and transparent background
        - Requires LaTeX installation on the system
    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )

    fig.savefig(
        f"{save_path}.svg",
        bbox_inches="tight",
        dpi=300,
        format="svg",
        transparent=True,
    )


def metrics_to_latex(
    results: list[EmbeddingState], layout_param: str = "balance factor"
) -> str:
    """Convert evaluation metrics to LaTeX table format.

    This function processes a list of EmbeddingState objects, extracts
    their metrics and metadata, and formats them as a LaTeX table string
    suitable for inclusion in academic documents.

    Args:
        results (list[EmbeddingState]): List of embedding states with
            computed metrics and metadata.
        layout_param (str, optional): The layout parameter to extract
            from metadata. Defaults to "balance factor".

    Returns:
        str: LaTeX-formatted table string with metrics and parameters.

    Note:
        - Combines metrics and metadata dataframes
        - Extracts resolution parameter from community detection
        - Converts float values to German decimal format (comma separator)
        - Removes object ID column from output
        - Uses 3 decimal places for float formatting
    """
    metrics_df = evaluation.create_report(results, metadata=False, metrics=True)
    metadata_df = evaluation.create_report(results, metadata=True, metrics=False)

    metrics_df.insert(1, "com_detection_params", metadata_df["com_detection_params"])
    metrics_df.insert(2, "placeholder", metadata_df["layout_params"])
    metrics_df.insert(3, "layout_params", metadata_df["layout_params"])

    # extract relevant parameters
    metrics_df["com_detection_params"] = metrics_df["com_detection_params"].apply(
        lambda d: d.get("resolution", "-")
    )
    metrics_df["layout_params"] = metrics_df["layout_params"].apply(
        lambda d: d.get(layout_param, "-")
    )

    metrics_df["placeholder"] = metrics_df["placeholder"].apply(
        lambda d: d.get("resolution", "-")
    )

    metrics_df = metrics_df.drop(columns=["obj_id"])

    # covert floats to strings with comma as decimal separator
    metrics_df = metrics_df.map(
        lambda x: f"{x:.3f}".replace(".", ",") if isinstance(x, float) else x
    )

    return metrics_df.to_latex(index=False)


def plot_metrics_report_single(
    data: pd.DataFrame,
    division: list[any] | None = None,
    export_mode: bool = False,
    label_height: float = 0.1,
) -> plt.Figure:
    """Create a single plot showing metrics across object IDs.

    This function creates a line plot displaying various evaluation metrics
    across different object IDs, with optional divisions and export formatting.

    Args:
        data (pd.DataFrame): DataFrame containing metrics data with 'obj_id'
            column and metric columns.
        division (list[any] | None, optional): List of division values for
            creating background shading and labels. Defaults to None.
        export_mode (bool, optional): Whether to use LaTeX formatting and
            German labels for publication. Defaults to False.
        label_height (float, optional): Height position for division labels.
            Defaults to 0.1.

    Returns:
        plt.Figure: The matplotlib figure containing the metrics plot.

    Note:
        - Uses colorblind-friendly palette
        - Different markers for each metric
        - Alternating background shading when divisions are provided
        - LaTeX formatting in export mode with German labels
        - Legend positioned outside plot area
    """
    df_melted = data.melt(id_vars="obj_id", var_name="Metric", value_name="Score")

    metrics = data.columns.drop("obj_id")

    if export_mode:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )
        x_label = "ID des Durchgangs"
        y_label = "Wert der Metrik"
        division_label = r"$\gamma=$"
        legend_labels = [
            r"$T(k)$",
            r"$C(k)$",
            r"$R_{NX}(k)$",
            r"$S^{sim}$",
            r"$S^{sim}_\mathcal{C}$",
            r"$\Delta S^{sim}_\mathcal{C}$",
            r"$M_R(k)$",
            r"$M_D$",
            r"$M_V(k)$",
        ]
    else:
        x_label = "Object ID"
        y_label = "Score"
        division_label = "Community size: \n"
        legend_labels = metrics.tolist()

    marker_list = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H"]
    marker_map = {
        metric: marker_list[i % len(marker_list)] for i, metric in enumerate(metrics)
    }

    palette = sb.color_palette("colorblind", n_colors=len(metrics))
    color_map = {metric: palette[i] for i, metric in enumerate(metrics)}

    sb.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 5))
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


def plot_metrics_report_multiple(
    data: pd.DataFrame,
    division: list[any] | None = None,
    export_mode: bool = False,
    ticks_decimal: bool = True,
) -> plt.Figure:
    """Create multiple subplots showing metrics across layout parameters.

    This function creates a grid of subplots, each showing metrics for
    different divisions of the data across varying layout parameters.

    Args:
        data (pd.DataFrame): DataFrame containing metrics data with 'obj_id',
            'layout_params' columns and metric columns.
        division (list[any] | None, optional): List of division values for
            creating separate subplots. Defaults to None.
        export_mode (bool, optional): Whether to use LaTeX formatting and
            German labels for publication. Defaults to False.
        ticks_decimal (bool, optional): Whether to format x-axis ticks as
            decimals or integers. Defaults to True.

    Returns:
        plt.Figure: The matplotlib figure containing the subplot grid.

    Note:
        - Automatically determines subplot grid layout (square-ish)
        - Uses colorblind-friendly palette with distinct markers
        - LaTeX formatting in export mode with German labels
        - Shared y-axis across subplots
        - Legend positioned outside plot area
        - German decimal formatting (comma separator) in export mode
    """
    df_melted = data.melt(
        id_vars=["obj_id", "layout_params"], var_name="Metric", value_name="Score"
    )

    metrics = data.columns.drop(["layout_params", "obj_id"])

    if export_mode:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )
        x_label = "Balancefaktor" if ticks_decimal else "Anzahl Iterationen"
        y_label = "Wert der Metrik"
        division_label = r"$\gamma=\ $"
        legend_labels = [
            r"$T(k)$",
            r"$C(k)$",
            r"$R_{NX}(k)$",
            r"$S^{sim}$",
            r"$S^{sim}_\mathcal{C}$",
            r"$\Delta S^{sim}_\mathcal{C}$",
            r"$M_R(k)$",
            r"$M_D$",
            r"$M_V(k)$",
        ]
    else:
        x_label = "layout parameter"
        y_label = "score"
        division_label = "resolution parameter: "
        legend_labels = metrics.tolist()

    marker_list = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H"]
    marker_map = {
        metric: marker_list[i % len(marker_list)] for i, metric in enumerate(metrics)
    }

    palette = sb.color_palette("colorblind", n_colors=len(metrics))
    color_map = {metric: palette[i] for i, metric in enumerate(metrics)}

    sb.set_style("white")
    obj_ids = sorted(data["obj_id"].unique())

    if division is not None:
        step = len(obj_ids) // len(division)

        ncols = math.ceil(math.sqrt(len(division)))
        nrows = math.ceil(len(division) / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True
        )
        axes = axes.flatten()

        if len(division) == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes = [axes]

    x_vals = sorted(data["layout_params"].unique())
    x_indices = range(len(x_vals))

    for idx, ax in enumerate(axes):
        if division is not None:
            start = idx * step
            end = (idx + 1) * step if idx < len(axes) - 1 else len(obj_ids)
            obj_range = obj_ids[start:end]
            df_range = df_melted[df_melted["obj_id"].isin(obj_range)]
        else:
            df_range = df_melted

        for metric in metrics:
            subset = df_range[df_range["Metric"] == metric]
            x_idx = [x_vals.index(v) for v in subset["layout_params"]]
            ax.plot(
                x_idx,
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

        ax.set_xticks(list(x_indices))
        ax.set_xticklabels(x_vals)
        ax.set_xlabel(x_label, fontsize=16)

        if idx % ncols == 0:
            ax.set_ylabel(y_label, fontsize=16)

        ax.tick_params(labelsize=12)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=1.0)
        ax.xaxis.grid(False)

        if division is not None:
            if export_mode:
                ax.set_title(
                    f"{division_label}${str(division[idx]).replace('.', ',')}$",
                    fontsize=16,
                )

                if ticks_decimal:
                    ax.xaxis.set_major_formatter(
                        FuncFormatter(
                            lambda x, _: f"{x_vals[int(x)]:.1f}".replace(".", ",")
                        )
                    )
                else:
                    ax.xaxis.set_major_formatter(
                        FuncFormatter(lambda x, _: f"{int(x_vals[int(x)])}")
                    )

                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda y, _: f"{y:.1f}".replace(".", ","))
                )
            else:
                ax.set_title(f"{division_label}{str(division[idx])}", fontsize=14)

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        fontsize=16,
    )

    sb.despine()
    plt.tight_layout(h_pad=3.0)
    plt.show()

    return fig


def insert_references(
    data: pd.DataFrame,
    target_features: list[str],
    embeddings: list[EmbeddingState],
    com_detection_param: str,
    layout_param: str,
) -> list[EmbeddingState]:
    """Insert reference embeddings at parameter intervals for comparison.

    This function creates reference embeddings using the original embedding
    coordinates but with modified metadata to enable comparison across
    different parameter values. Reference embeddings are inserted at the
    beginning of each parameter group.

    Args:
        data (pd.DataFrame): The original data used for computing metrics.
        target_features (list[str]): List of target feature column names.
        embeddings (list[EmbeddingState]): List of embedding states to process.
        com_detection_param (str): Community detection parameter name to
            group by.
        layout_param (str): Layout parameter name to set to 0 for references.

    Returns:
        list[EmbeddingState]: New list of embeddings with reference embeddings
        inserted and metrics recomputed.

    Note:
        - Reference embeddings use coordinates from the first embedding
        - Layout parameter is set to 0 for reference embeddings
        - Object IDs are reassigned sequentially
        - Metrics are recomputed for all embeddings including references
        - References are inserted at the start of each parameter group
    """
    params = []

    for mod_embedding in embeddings[1:]:
        params.append(
            mod_embedding.metadata["com_detection_params"][com_detection_param]
        )

    params = list(set(params))
    step_count = len(embeddings) // len(params) if params else 1

    new_embeddings = embeddings[1:]

    new_embeddings = []
    for i, emb in enumerate(embeddings[1:], start=1):
        if (i - 1) % step_count == 0:
            inserted_embedding = copy.deepcopy(emb)
            inserted_embedding.metadata["layout_params"][layout_param] = 0
            inserted_embedding.partition = emb.partition
            inserted_embedding.embedding = embeddings[0].embedding.copy()
            new_embeddings.append(inserted_embedding)

        new_embeddings.append(copy.deepcopy(emb))

    for i, emb in enumerate(new_embeddings, start=1):
        emb.obj_id = i - 1

    new_embeddings = evaluation.compute_metrics(
        data, new_embeddings, target_features, new_embeddings[0].metadata["k_neighbors"]
    )

    return new_embeddings
