import copy
import pickle
import time
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sb
from matplotlib.ticker import FuncFormatter

from . import EmbeddingState, evaluation


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


def add_patches(fig: plt.Figure, patch_coords: list[tuple]) -> plt.Figure:
    ax = fig.axes[0]

    ellipse_a = patches.Ellipse(
        patch_coords[0],
        0.5,
        0.5,
        angle=0,
        edgecolor="black",
        facecolor="none",
        linewidth=1,
        linestyle="dashed",
    )
    ax.add_patch(ellipse_a)
    # ax.text(
    #     patch_coords[0][0],
    #     patch_coords[0][1],
    #     "Bereich A",
    #     color="black",
    #     fontsize=10,
    #     ha="center",
    #     va="center",
    # )

    plt.show()

    return fig


def export_plot(fig: plt.Figure, save_path: str):
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

    # covert floats to strings with comma as decimal separator
    metrics_df = metrics_df.map(
        lambda x: f"{x:.3f}".replace(".", ",") if isinstance(x, float) else x
    )

    return metrics_df.to_latex(index=False)


def plot_metrics_report(
    data: pd.DataFrame,
    division: list[any] | None = None,
    export_mode: bool = False,
    label_height: float = 0.1,
) -> plt.Figure:
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
        # division_label = "Anzahl Communities: \n"
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


def plot_metrics_report_new(
    data: pd.DataFrame,
    division: list[any] | None = None,
    export_mode: bool = False,
    label_height: float = 0.1,
) -> plt.Figure:
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
        x_label = "Anzahl Iterationen"
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

    # --- Subplots erstellen ---
    if division is not None:
        step = len(obj_ids) // len(division)
        fig, axes = plt.subplots(
            1, len(division), figsize=(5 * len(division), 5), sharey=True
        )
        if len(division) == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes = [axes]

    x_vals = sorted(data["layout_params"].unique())

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
            ax.plot(
                subset["layout_params"],
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
        ax.set_xticks(x_vals)
        if idx == 0:
            ax.set_ylabel(y_label, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=1.0)
        ax.xaxis.grid(False)

        if division is not None:
            if export_mode:
                ax.set_title(
                    f"{division_label}${str(division[idx]).replace('.', ',')}$",
                    fontsize=14,
                )
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ","))
                )
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda y, _: f"{y:.1f}".replace(".", ","))
                )
            else:
                ax.set_title(f"{division_label}{str(division[idx])}", fontsize=14)

    # --- Gemeinsame Legende ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",  # unterhalb
        bbox_to_anchor=(0.5, -0.1),  # etwas Abstand nach unten
        ncol=len(legend_labels),  # alles in einer Zeile
        fontsize=14,
    )

    sb.despine()
    plt.tight_layout()
    plt.show()

    return fig


def insert_references(
    data: pd.DataFrame,
    target_features: list[str],
    embeddings: list[EmbeddingState],
    com_detection_param: str,
    layout_param: str,
) -> list[EmbeddingState]:
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
