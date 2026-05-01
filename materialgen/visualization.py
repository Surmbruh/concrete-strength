"""Сводный модуль визуализации для всех стадий пайплайна.

Сгруппировано по разделам:

* Графики обучения и валидации (`write_training_plot`, `write_predictions_plot`,
  `write_residuals_plot`, `write_fitness_history_plot`).
* T-SNE проекции (`write_tsne_plot`).
* Граф трейненной NEAT-BNN топологии в Graphviz (`write_bnn_topology`).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

matplotlib.use("Agg")


# =============================================================================
# 1. Графики обучения и валидации
# =============================================================================

def write_training_plot(
    epoch_train_losses: list[float],
    epoch_val_losses: list[float],
    output_path: str | Path,
    title: str = "Training progress",
) -> str:
    """Сохранить график train/val потерь в PNG."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(epoch_train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, epoch_train_losses, label="Train loss", linewidth=1.6)
    if epoch_val_losses:
        ax.plot(epochs, epoch_val_losses, label="Validation loss", linewidth=1.6, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalised MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return str(output_path)


def write_predictions_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None,
    property_names: list[str],
    output_path: str | Path,
    title: str = "Predicted vs Actual",
) -> str:
    """Сохранить scatter «предсказание vs истина» (одна панель на свойство)."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(property_names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, name in enumerate(property_names):
        ax = axes[idx // ncols][idx % ncols]
        true_col = y_true[:, idx]
        pred_col = y_pred[:, idx]

        if y_std is not None:
            std_col = y_std[:, idx]
            ax.errorbar(
                true_col, pred_col,
                yerr=std_col,
                fmt="o", alpha=0.55, markersize=4, linewidth=0.6,
                ecolor="gray", capsize=2,
            )
        else:
            ax.scatter(true_col, pred_col, alpha=0.55, s=18)

        lo = min(true_col.min(), pred_col.min())
        hi = max(true_col.max(), pred_col.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", linewidth=1.2, label="ideal")

        ss_res = float(np.sum((true_col - pred_col) ** 2))
        ss_tot = float(np.sum((true_col - true_col.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}  (R²={r2:.3f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def write_fitness_history_plot(
    best_fitness: list[float],
    mean_fitness: list[float],
    median_fitness: list[float],
    stdev_fitness: list[float] | None,
    output_path: str | Path,
    title: str = "NEAT fitness progress",
) -> str:
    """Сохранить график динамики fitness по поколениям NEAT."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generations = list(range(1, len(best_fitness) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(generations, best_fitness, label="Best fitness", linewidth=1.6)
    if mean_fitness:
        ax.plot(generations, mean_fitness, label="Mean fitness", linewidth=1.6, linestyle="--")
        if stdev_fitness and len(stdev_fitness) == len(mean_fitness):
            mean_arr = np.asarray(mean_fitness)
            std_arr = np.asarray(stdev_fitness)
            ax.fill_between(generations, mean_arr - std_arr, mean_arr + std_arr, alpha=0.2)
    if median_fitness:
        ax.plot(generations, median_fitness, label="Median fitness", linewidth=1.2, linestyle=":")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return str(output_path)


def write_residuals_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    property_names: list[str],
    output_path: str | Path,
    title: str = "Residuals",
) -> str:
    """Сохранить гистограмму остатков (одна панель на свойство)."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(property_names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, name in enumerate(property_names):
        ax = axes[idx // ncols][idx % ncols]
        residuals = y_true[:, idx] - y_pred[:, idx]
        mae = float(np.mean(np.abs(residuals)))
        std = float(np.std(residuals))

        ax.hist(residuals, bins=20, edgecolor="white", linewidth=0.4)
        ax.axvline(0.0, color="red", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Residual (actual − predicted)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}  (MAE={mae:.4g}, σ={std:.4g})")
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


# =============================================================================
# 2. T-SNE проекции
# =============================================================================

def write_tsne_plot(
    X: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    title: str = "T-SNE",
) -> str | None:
    """Сохранить T-SNE с разметкой train/val. Возвращает None если данных мало."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = X.shape[0]
    if n < 4:
        return None

    perplexity = min(30.0, max(2.0, n // 4))
    embedded = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        embedded[train_idx, 0], embedded[train_idx, 1],
        alpha=0.6, s=18, label=f"Обучающая ({len(train_idx)} точек)",
    )
    ax.scatter(
        embedded[val_idx, 0], embedded[val_idx, 1],
        alpha=0.8, s=22, marker="^", label=f"Валидационная ({len(val_idx)} точек)",
    )
    ax.set_xlabel("T-SNE 1")
    ax.set_ylabel("T-SNE 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


# =============================================================================
# 3. Хелперы для Graphviz-визуализации NEAT-BNN
# =============================================================================

def _variational_param(name: str) -> np.ndarray | None:
    """Прочитать вариационный параметр из Pyro param store, если он есть."""

    try:
        import pyro

        store = pyro.get_param_store()
        if name not in store:
            return None
        return store[name].detach().cpu().numpy()
    except Exception:
        return None



# =============================================================================
# 4. Граф NEAT-BNN топологии
# =============================================================================

def _bnn_html_label(title: str, lines: list[str]) -> str:
    rows = [f'<TR><TD><B>{title}</B></TD></TR>']
    for line in lines:
        rows.append(f'<TR><TD><FONT POINT-SIZE="8">{line}</FONT></TD></TR>')
    return "<" + '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1">' + "".join(rows) + "</TABLE>>"


def _bnn_normal_label(mean: float, std: float) -> str:
    return f"N({mean:.2f},{std:.2f})"


def write_bnn_topology(
    regressor,
    output_dir: str | Path,
) -> dict[str, str]:
    """Граф NEAT-топологии байесовской сети в DOT/PNG/SVG.

    `regressor` — обученный `NeatBNNRegressor` с заполненным Pyro param store.
    """

    topology = regressor.topology
    input_keys: list[int] = topology["input_keys"]
    output_keys: list[int] = topology["output_keys"]
    layers: list[list[int]] = topology["layers"]
    masks: list = topology["masks"]

    input_names = regressor.input_names or [f"x{i}" for i in range(len(input_keys))]
    output_names = regressor.output_names or [f"y{i}" for i in range(len(output_keys))]
    output_key_set = set(output_keys)

    layer_posteriors: list[dict[str, np.ndarray | None]] = []
    for k in range(len(masks)):
        layer_posteriors.append({
            "weight_loc": _variational_param(f"layer_{k}.weight_loc"),
            "weight_scale": _variational_param(f"layer_{k}.weight_scale"),
            "bias_loc": _variational_param(f"layer_{k}.bias_loc"),
            "bias_scale": _variational_param(f"layer_{k}.bias_scale"),
        })

    lines: list[str] = [
        "digraph bnn_network {",
        '  graph [rankdir=LR, labelloc="t", labeljust="l", pad="0.25", nodesep="0.40", ranksep="0.9"];',
        '  label="NEAT-BNN Topology (posterior weight distributions)";',
        '  node [fontname="Helvetica", fontsize=10, shape=box, style="filled,rounded"];',
        '  edge [fontname="Helvetica", fontsize=8, arrowsize=0.7];',
    ]

    all_locs: list[float] = []
    all_scales: list[float] = []
    for k, post in enumerate(layer_posteriors):
        mask_np = masks[k].numpy() if hasattr(masks[k], "numpy") else np.asarray(masks[k])
        wl = post["weight_loc"]
        ws = post["weight_scale"]
        if wl is not None:
            all_locs.extend(np.abs(wl[mask_np > 0]).tolist())
        if ws is not None:
            all_scales.extend(ws[mask_np > 0].tolist())
    loc_ref = float(np.mean(all_locs)) if all_locs else 1.0
    scale_ref = float(np.mean(all_scales)) if all_scales else 1.0

    for idx, node_id in enumerate(input_keys):
        label = _bnn_html_label(input_names[idx], ["property input"])
        lines.append(f'  "n{node_id}" [label={label}, fillcolor="#dbeafe", color="#1d4ed8"];')

    for layer_idx in range(1, len(layers)):
        post = layer_posteriors[layer_idx - 1]
        bias_loc = post["bias_loc"]
        bias_scale = post["bias_scale"]
        for t_idx, node_id in enumerate(layers[layer_idx]):
            if node_id in output_key_set:
                out_idx = output_keys.index(node_id)
                name = output_names[out_idx]
                details = ["component output"]
                if bias_loc is not None and bias_scale is not None:
                    details.append(f"b~{_bnn_normal_label(float(bias_loc[t_idx]), float(bias_scale[t_idx]))}")
                label = _bnn_html_label(name, details)
                lines.append(f'  "n{node_id}" [label={label}, fillcolor="#dcfce7", color="#15803d"];')
            else:
                details = [f"hidden {node_id}"]
                if bias_loc is not None and bias_scale is not None:
                    details.append(f"b~{_bnn_normal_label(float(bias_loc[t_idx]), float(bias_scale[t_idx]))}")
                label = _bnn_html_label(f"h{node_id}", details)
                lines.append(
                    f'  "n{node_id}" [label={label}, shape=ellipse, style="filled",'
                    f' fillcolor="#f3f4f6", color="#4b5563"];'
                )

    for layer_idx in range(1, len(layers)):
        post = layer_posteriors[layer_idx - 1]
        mask_np = masks[layer_idx - 1].numpy() if hasattr(masks[layer_idx - 1], "numpy") else np.asarray(masks[layer_idx - 1])
        wl = post["weight_loc"]
        ws = post["weight_scale"]

        target_nodes = layers[layer_idx]
        source_pool: list[int] = []
        for prev in range(layer_idx):
            source_pool.extend(layers[prev])

        edge_count = int(mask_np.sum())
        show_labels = edge_count <= 24

        for t_idx, t_node in enumerate(target_nodes):
            for s_idx, s_node in enumerate(source_pool):
                if mask_np[t_idx, s_idx] < 0.5:
                    continue
                w_loc = float(wl[t_idx, s_idx]) if wl is not None else 0.0
                w_scale = float(ws[t_idx, s_idx]) if ws is not None else 0.0

                penwidth = 0.7 + min(2.8, abs(w_loc) / max(loc_ref, 1e-6))
                color = "#0f766e" if w_loc >= 0 else "#b45309"
                style = "solid" if w_scale <= max(scale_ref, 1e-6) else "dashed"

                attrs = [
                    f'color="{color}"',
                    f'fontcolor="{color}"',
                    f'penwidth={penwidth:.2f}',
                    f'style="{style}"',
                ]
                if show_labels:
                    lbl = json.dumps(f"w~{_bnn_normal_label(w_loc, w_scale)}")
                    attrs.append(f"label={lbl}")
                lines.append(f'  "n{s_node}" -> "n{t_node}" [{", ".join(attrs)}];')

    for layer_nodes in layers:
        rank_items = " ".join(f'"n{nid}";' for nid in layer_nodes)
        lines.append(f"  {{ rank=same; {rank_items} }}")

    lines.append("}")
    dot_source = "\n".join(lines)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dot_path = out / "bnn_network.dot"
    png_path = out / "bnn_network.png"
    svg_path = out / "bnn_network.svg"

    dot_path.write_text(dot_source, encoding="utf-8")
    artifacts: dict[str, str] = {"dot": str(dot_path)}
    try:
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
        artifacts["png"] = str(png_path)
    except Exception:
        pass
    try:
        subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
        artifacts["svg"] = str(svg_path)
    except Exception:
        pass

    return artifacts
