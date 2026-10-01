"""Visual demo of label propagation on credit-card fraud.

Implements the transductive diffusion step of Iscen et al.,
"Label Propagation for Deep Semi-supervised Learning" (arXiv:1904.04717).

A 2-NN graph is built on L2-normalized transaction descriptors. Labels
diffuse for 200 epochs of the Zhou iteration cited in that paper. The
epochs are drawn on a fixed 2D UMAP of the same descriptors.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Protocol requested for the demo.
N_POINTS = 2000
N_FRAUD = 10
N_LABELED = 200
N_LABELED_FRAUD = 4
K_NEIGHBORS = 2
GAMMA = 3.0
ALPHA = 0.99
N_EPOCHS = 200
SEED = 0

REQUESTED_DATASET = "mlegrad/msbd5013-creditcard-fraud"
FALLBACK_DATASET = "jyunyilin/credit-card-fraud-detection"

FRAUD_BLUE = "#2166ac"
NONFRAUD_RED = "#b2182b"
UNLABELED_GRAY = "#bdbdbd"
FIGSIZE = (12.8, 7.2)
DPI = 100

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FRAME_DIR = OUTPUT_DIR / "frames"
KEYFRAME_DIR = OUTPUT_DIR / "keyframes"

# Red (non-fraud) -> pale (mixed weight) -> blue (fraud).
WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "fraud_weight",
    [NONFRAUD_RED, "#f7f7f7", FRAUD_BLUE],
)


def get_fraud_data():
    """Load the credit-card fraud table as a pandas frame.

    Tries the requested Hub dataset first. If that repository cannot be
    read, falls back to a public mirror of the same ULB table.
    """
    from datasets import load_dataset

    try:
        dataset = load_dataset(REQUESTED_DATASET, split="train")
        return dataset.to_pandas(), REQUESTED_DATASET
    except Exception as exc:
        print(
            f"Could not load {REQUESTED_DATASET!r} "
            f"({type(exc).__name__}: {exc})."
        )
        print(
            "Using the public ULB credit-card fraud mirror "
            f"{FALLBACK_DATASET!r}."
        )
        dataset = load_dataset(FALLBACK_DATASET, split="train")
        return dataset.to_pandas(), FALLBACK_DATASET


def feature_matrix(df):
    """Descriptors used for the graph and for UMAP.

    V1–V28 are the dataset's PCA embeddings. Amount is kept and scaled
    with them. Time is a timestamp, so it is not used as a descriptor.
    """
    label_names = {"class", "label", "target"}
    drop = {"time"} | label_names
    columns = [c for c in df.columns if c.lower() not in drop]
    if not columns:
        raise ValueError(f"No feature columns in {list(df.columns)}")
    labels = None
    for column in df.columns:
        if column.lower() in label_names:
            labels = column
            break
    if labels is None:
        raise ValueError(f"No class column in {list(df.columns)}")
    y = df[labels].to_numpy()
    y = np.rint(y.astype(np.float64)).astype(np.int64)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError(f"Expected binary Class labels, found {np.unique(y)}")
    x = df[columns].to_numpy(dtype=np.float64)
    return x, y, columns


def subsample(x, y, rng):
    """2000 rows with 10 fraud cases, then 200 seeds (4 of them fraud)."""
    fraud = np.flatnonzero(y == 1)
    normal = np.flatnonzero(y == 0)
    if len(fraud) < N_FRAUD or len(normal) < N_POINTS - N_FRAUD:
        raise ValueError(
            f"Need at least {N_FRAUD} fraud and {N_POINTS - N_FRAUD} "
            f"non-fraud rows, found {len(fraud)} and {len(normal)}."
        )
    chosen = np.concatenate(
        [
            rng.choice(fraud, size=N_FRAUD, replace=False),
            rng.choice(normal, size=N_POINTS - N_FRAUD, replace=False),
        ]
    )
    rng.shuffle(chosen)
    x_sub = x[chosen]
    y_sub = y[chosen]

    fraud_pos = np.flatnonzero(y_sub == 1)
    normal_pos = np.flatnonzero(y_sub == 0)
    labeled = np.zeros(N_POINTS, dtype=bool)
    labeled[rng.choice(fraud_pos, size=N_LABELED_FRAUD, replace=False)] = True
    labeled[
        rng.choice(normal_pos, size=N_LABELED - N_LABELED_FRAUD, replace=False)
    ] = True
    return x_sub, y_sub, labeled


def l2_normalize(x):
    scaled = StandardScaler().fit_transform(x)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.clip(norms, 1e-12, None)


def knn_affinity(descriptors, k, gamma):
    """Equation (9): a_ij = [v_i · v_j]_+^γ when i is a k-NN of j."""
    n = descriptors.shape[0]
    # Ask for extra neighbors so exact duplicate rows still leave k others
    # after self is removed.
    neighbors = NearestNeighbors(n_neighbors=k + 3, metric="euclidean")
    neighbors.fit(descriptors)
    _, index = neighbors.kneighbors(descriptors)
    nbr = np.empty((n, k), dtype=np.int64)
    for j, row in enumerate(index):
        others = [int(i) for i in row if i != j]
        if len(others) < k:
            raise RuntimeError(f"Point {j} has fewer than {k} neighbors.")
        nbr[j] = others[:k]
    rows = nbr.ravel()
    cols = np.repeat(np.arange(n), k)
    sims = np.sum(descriptors[rows] * descriptors[cols], axis=1)
    weights = np.clip(sims, 0.0, None) ** gamma
    affinity = np.zeros((n, n), dtype=np.float64)
    np.add.at(affinity, (rows, cols), weights)
    np.fill_diagonal(affinity, 0.0)
    return affinity


def normalized_adjacency(affinity):
    """W = A + Aᵀ, then S = D^{-1/2} W D^{-1/2}."""
    adjacency = affinity + affinity.T
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = (inv_sqrt[:, None] * adjacency) * inv_sqrt[None, :]
    return adjacency, normalized


def label_matrix(y, labeled):
    """Y is one-hot on labeled rows and zero on held-out rows. Column 1 is fraud."""
    matrix = np.zeros((y.shape[0], 2), dtype=np.float64)
    matrix[labeled & (y == 0), 0] = 1.0
    matrix[labeled & (y == 1), 1] = 1.0
    return matrix


def propagate(normalized, labels, alpha, epochs):
    """Zhou iteration Z ← α S Z + (1 − α) Y, one step per epoch.

    The limit is (1 − α) (I − α S)^{-1} Y. Iscen et al. solve
    (I − α S) Z = Y, which is the same matrix up to that positive scale,
    so row-normalized weights and argmax agree.
    """
    current = labels.copy()
    history = [current.copy()]
    deltas = [0.0]
    for _ in range(epochs):
        updated = alpha * (normalized @ current) + (1.0 - alpha) * labels
        deltas.append(float(np.linalg.norm(updated - current)))
        current = updated
        history.append(current.copy())
    return history, deltas


def project_umap(descriptors):
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        # A larger min_dist keeps the fraud cases from stacking on the
        # non-fraud mass. t-SNE packed that group even tighter.
        min_dist=0.8,
        metric="cosine",
        random_state=SEED,
    )
    return reducer.fit_transform(descriptors)


def edge_segments(adjacency, xy):
    row, col = np.where(np.triu(adjacency, k=1) > 0)
    return np.stack([xy[row], xy[col]], axis=1)


def class_weights(scores):
    """Row-normalized fraud weight. Negative mass marks 'no label yet'."""
    mass = scores.sum(axis=1)
    weight = np.full(scores.shape[0], -1.0, dtype=np.float64)
    known = mass > 1e-12
    weight[known] = scores[known, 1] / mass[known]
    return weight


def colors_from_weights(weight):
    rgba = np.empty((weight.shape[0], 4), dtype=np.float64)
    unknown = weight < 0
    rgba[unknown] = matplotlib.colors.to_rgba(UNLABELED_GRAY)
    if np.any(~unknown):
        rgba[~unknown] = WEIGHT_CMAP(np.clip(weight[~unknown], 0.0, 1.0))
    return rgba


def hard_colors(classes, known):
    """classes: 0 non-fraud, 1 fraud. Unknown rows are gray."""
    rgba = np.empty((classes.shape[0], 4), dtype=np.float64)
    rgba[:] = matplotlib.colors.to_rgba(UNLABELED_GRAY)
    rgba[known & (classes == 0)] = matplotlib.colors.to_rgba(NONFRAUD_RED)
    rgba[known & (classes == 1)] = matplotlib.colors.to_rgba(FRAUD_BLUE)
    return rgba


def _scatter(ax, xy, rgba, labeled, weight_for_size):
    """Non-fraud is a light background. Fraud is large, opaque, and on top.

    Ten fraud points inside 1,990 non-fraud markers disappear if both are
    drawn the same way. Seeds stay triangles.
    """
    fraudish = weight_for_size > 0.5
    background = rgba.copy()
    background[~fraudish, 3] = 0.22
    unknown = weight_for_size < 0
    background[unknown, 3] = 0.38

    for mask, marker, size in (
        (~labeled & ~fraudish, "o", 10.0),
        (labeled & ~fraudish, "^", 16.0),
    ):
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=background[mask],
            s=size,
            marker=marker,
            linewidths=0,
            zorder=2 if marker == "o" else 3,
        )

    for mask, marker, size in (
        (~labeled & fraudish, "o", 70.0),
        (labeled & fraudish, "^", 120.0),
    ):
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=size * 2.4,
            c="white",
            marker=marker,
            linewidths=0,
            zorder=5,
        )
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=rgba[mask],
            s=size,
            marker=marker,
            linewidths=0.7,
            edgecolors="#08306b",
            zorder=6,
        )


def _draw_graph(ax, segments):
    if len(segments) == 0:
        return
    ax.add_collection(
        LineCollection(
            segments,
            colors="#9aa0a6",
            linewidths=0.35,
            alpha=0.28,
            zorder=1,
        )
    )


def _style_axis(ax, xy):
    center = xy.mean(axis=0)
    span = np.max(xy.max(axis=0) - xy.min(axis=0))
    half = 0.5 * span * 1.12
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#e0e0e0")
    ax.set_facecolor("white")


def _footer(fig, x=0.46):
    fig.text(
        x,
        0.015,
        "Blue fraud is drawn on top    Red non-fraud is lighter"
        "    Gray = no propagated mass    Triangle = labeled seed",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )


def _inches_to_fig(width_in, height_in):
    return width_in / FIGSIZE[0], height_in / FIGSIZE[1]


def render_evolution_frame(path, xy, segments, labeled, weight, epoch, delta):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("white")
    # Square axes, centered, with the colorbar immediately to its right.
    # A wide 16:9 subplot plus equal aspect otherwise pins the cloud to one side.
    side = 5.35
    width, height = _inches_to_fig(side, side)
    colorbar_width = 0.018
    gap = 0.02
    left = (1.0 - width - gap - colorbar_width) / 2
    bottom = 0.11
    ax = fig.add_axes([left, bottom, width, height])
    color_ax = fig.add_axes([left + width + gap, bottom + 0.03, colorbar_width, height - 0.06])
    _draw_graph(ax, segments)
    rgba = colors_from_weights(weight)
    _scatter(ax, xy, rgba, labeled, weight)
    _style_axis(ax, xy)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=WEIGHT_CMAP, norm=plt.Normalize(0, 1)),
        cax=color_ax,
    )
    colorbar.set_label("fraud weight", fontsize=9)
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["non-fraud", "mixed", "fraud"])

    fig.suptitle(
        "Label propagation on a 2-NN graph of credit-card transactions",
        fontsize=15,
        fontweight="bold",
        y=0.96,
    )
    if epoch == 0:
        subtitle = "Epoch 0 / 200    initial seeds only (4 fraud, 196 non-fraud)"
    else:
        subtitle = f"Epoch {epoch} / 200    ||ΔZ|| = {delta:.2e}"
    ax.set_title(subtitle, fontsize=12, pad=8)
    _footer(fig)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def render_final_frame(path, xy, segments, labeled, scores, y):
    weight = class_weights(scores)
    known = weight >= 0
    predicted = scores.argmax(axis=1)
    pred_weight = np.where(known, predicted.astype(np.float64), -1.0)
    truth_weight = y.astype(np.float64)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("white")
    side = 4.55
    width, height = _inches_to_fig(side, side)
    gap = 0.07
    left = (1.0 - 2 * width - gap) / 2
    bottom = 0.16
    axes = [
        fig.add_axes([left, bottom, width, height]),
        fig.add_axes([left + width + gap, bottom, width, height]),
    ]
    fig.suptitle(
        "Epoch 200    argmax pseudo-labels beside held-out ground truth",
        fontsize=15,
        fontweight="bold",
        y=0.95,
    )

    panels = (
        (
            axes[0],
            hard_colors(predicted, known),
            pred_weight,
            "Argmax of propagated weights",
        ),
        (
            axes[1],
            hard_colors(y, np.ones(len(y), dtype=bool)),
            truth_weight,
            "Ground truth, including held-out labels",
        ),
    )
    for ax, rgba, size_weight, title in panels:
        _draw_graph(ax, segments)
        _scatter(ax, xy, rgba, labeled, size_weight)
        _style_axis(ax, xy)
        ax.set_title(title, fontsize=12, pad=8)
    _footer(fig, x=0.50)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write_video(mp4_path, gif_path):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the movie.")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "10",
            "-start_number",
            "0",
            "-i",
            str(FRAME_DIR / "frame_%03d.png"),
            "-vf",
            "tpad=stop_mode=clone:stop_duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(mp4_path),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "10",
            "-start_number",
            "0",
            "-i",
            str(FRAME_DIR / "frame_%03d.png"),
            "-vf",
            "tpad=stop_mode=clone:stop_duration=3,scale=800:-1:flags=lanczos,"
            "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse",
            str(gif_path),
        ],
        check=True,
        capture_output=True,
    )


def evaluate(scores, y, labeled):
    weight = class_weights(scores)
    known = weight >= 0
    predicted = scores.argmax(axis=1)
    unlabeled = ~labeled
    held_out_fraud = unlabeled & (y == 1)

    def accuracy(mask):
        if not np.any(mask):
            return None
        return float((predicted[mask] == y[mask]).mean())

    return {
        "unlabeled_accuracy": accuracy(unlabeled),
        "reached_unlabeled_accuracy": accuracy(unlabeled & known),
        "labeled_seed_accuracy": accuracy(labeled),
        "held_out_fraud_recall": float(predicted[held_out_fraud].mean())
        if np.any(held_out_fraud)
        else None,
        "held_out_fraud_predicted": predicted[held_out_fraud].astype(int).tolist(),
        "held_out_fraud_weight": [
            None if weight[i] < 0 else round(float(weight[i]), 4)
            for i in np.flatnonzero(held_out_fraud)
        ],
        "fraction_reached": float(known.mean()),
        "predicted_fraud": int((predicted[known] == 1).sum()) if np.any(known) else 0,
    }


def main():
    rng = np.random.default_rng(SEED)
    frame = get_fraud_data()
    table, dataset_id = frame
    features, labels, columns = feature_matrix(table)
    print(
        f"Loaded {dataset_id} with shape {table.shape} "
        f"and {int((labels == 1).sum())} fraud rows."
    )
    features, labels, labeled = subsample(features, labels, rng)
    n_labeled_fraud = int((labeled & (labels == 1)).sum())
    n_held_fraud = int((~labeled & (labels == 1)).sum())
    print(
        f"Subset: {len(labels)} points, {(labels == 1).sum()} fraud. "
        f"Seeds: {int(labeled.sum())} ({n_labeled_fraud} fraud, "
        f"{int((labeled & (labels == 0)).sum())} non-fraud). "
        f"Held-out fraud: {n_held_fraud}."
    )

    descriptors = l2_normalize(features)
    affinity = knn_affinity(descriptors, K_NEIGHBORS, GAMMA)
    adjacency, normalized = normalized_adjacency(affinity)
    seeds = label_matrix(labels, labeled)
    history, deltas = propagate(normalized, seeds, ALPHA, N_EPOCHS)
    print("Projecting descriptors with UMAP...")
    xy = project_umap(descriptors)
    segments = edge_segments(adjacency, xy)
    n_edges = int(len(segments))
    print(f"2-NN graph has {n_edges} undirected edges.")

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    # Frames 0..198: epoch 0 (seeds) through epoch 198 (soft weights).
    # Frame 199: argmax after epoch 200, next to ground truth.
    for epoch in range(N_EPOCHS - 1):
        path = FRAME_DIR / f"frame_{epoch:03d}.png"
        render_evolution_frame(
            path,
            xy,
            segments,
            labeled,
            class_weights(history[epoch]),
            epoch,
            deltas[epoch],
        )
        frame_paths.append(path)
        if epoch % 20 == 0:
            print(f"Rendered epoch {epoch}")
    final_path = FRAME_DIR / f"frame_{N_EPOCHS - 1:03d}.png"
    render_final_frame(final_path, xy, segments, labeled, history[N_EPOCHS], labels)
    frame_paths.append(final_path)
    print(f"Rendered {len(frame_paths)} frames.")

    from PIL import Image

    sizes = {Image.open(path).size for path in frame_paths}
    if len(sizes) != 1:
        raise RuntimeError(f"Frame sizes differ: {sizes}")

    mp4_path = OUTPUT_DIR / "label_propagation.mp4"
    gif_path = OUTPUT_DIR / "label_propagation.gif"
    print("Encoding mp4 and gif...")
    write_video(mp4_path, gif_path)

    key_epochs = {
        "epoch_000_seeds.png": 0,
        "epoch_001.png": 1,
        "epoch_005.png": 5,
        "epoch_010.png": 10,
        "epoch_050.png": 50,
        "epoch_200_argmax_vs_truth.png": N_EPOCHS - 1,
    }
    for name, index in key_epochs.items():
        shutil.copyfile(frame_paths[index], KEYFRAME_DIR / name)

    metrics = evaluate(history[N_EPOCHS], labels, labeled)
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    n_components, _ = connected_components(
        csr_matrix(adjacency > 0), directed=False
    )
    summary = {
        "dataset": dataset_id,
        "requested_dataset": REQUESTED_DATASET,
        "n_points": N_POINTS,
        "n_fraud": int((labels == 1).sum()),
        "n_labeled": int(labeled.sum()),
        "n_labeled_fraud": n_labeled_fraud,
        "n_labeled_nonfraud": int((labeled & (labels == 0)).sum()),
        "n_held_out_fraud": n_held_fraud,
        "features": columns,
        "k": K_NEIGHBORS,
        "gamma": GAMMA,
        "alpha": ALPHA,
        "epochs": N_EPOCHS,
        "seed": SEED,
        "undirected_edges": n_edges,
        "connected_components": int(n_components),
        "delta_epoch_10": deltas[10],
        "delta_epoch_200": deltas[200],
        "frame_size": list(sizes.pop()),
        **metrics,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {mp4_path}")
    print(f"Wrote {gif_path}")


if __name__ == "__main__":
    main()
