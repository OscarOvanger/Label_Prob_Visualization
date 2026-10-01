"""Visual demo of label propagation on three MNIST digits.

Implements the transductive diffusion step of Iscen et al.,
"Label Propagation for Deep Semi-supervised Learning" (arXiv:1904.04717).

A 2-NN graph is built on L2-normalized image descriptors. Labels diffuse
for 200 epochs of the Zhou iteration cited in that paper. The epochs are
drawn on a fixed 2D UMAP of the same descriptors.
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
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

# Same protocol as the fraud demo, on three classes instead of two.
N_POINTS = 2000
N_LABELED = 200
K_NEIGHBORS = 2
GAMMA = 3.0
ALPHA = 0.99
N_EPOCHS = 200
SEED = 0

DATASET_ID = "ylecun/mnist"
DIGITS = (0, 1, 2)

# One color per class. Soft labels mix these in proportion to the weights.
CLASS_COLORS = ("#e41a1c", "#377eb8", "#4daf4a")
UNLABELED_GRAY = "#bdbdbd"
FIGSIZE = (12.8, 7.2)
DPI = 100

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FRAME_DIR = OUTPUT_DIR / "frames"
KEYFRAME_DIR = OUTPUT_DIR / "keyframes"


def class_names():
    return tuple(f"digit {digit}" for digit in DIGITS)


def class_rgb():
    return np.array(
        [matplotlib.colors.to_rgb(color) for color in CLASS_COLORS],
        dtype=np.float64,
    )


def get_mnist_labels():
    """Load MNIST train labels. Images are fetched only for the subset."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="train")
    labels = np.asarray(dataset["label"], dtype=np.int64)
    return dataset, labels


def subsample(labels, rng):
    """2000 images split across the three digits, then 200 seeds.

    Counts are as even as the totals allow. The other 1800 labels are held out.
    Returned labels are 0, 1, 2 in DIGITS order, not the original digit ids.
    """
    n_classes = len(DIGITS)
    base, remainder = divmod(N_POINTS, n_classes)
    chosen = []
    subset_labels = []
    counts = []
    for class_id, digit in enumerate(DIGITS):
        count = base + (1 if class_id < remainder else 0)
        pool = np.flatnonzero(labels == digit)
        if len(pool) < count:
            raise ValueError(f"Digit {digit} has {len(pool)} rows, need {count}.")
        pick = rng.choice(pool, size=count, replace=False)
        chosen.append(pick)
        subset_labels.append(np.full(count, class_id, dtype=np.int64))
        counts.append(count)
    indices = np.concatenate(chosen)
    y = np.concatenate(subset_labels)
    order = rng.permutation(len(indices))
    indices = indices[order]
    y = y[order]

    labeled = np.zeros(N_POINTS, dtype=bool)
    label_base, label_remainder = divmod(N_LABELED, n_classes)
    labeled_counts = []
    for class_id in range(n_classes):
        count = label_base + (1 if class_id < label_remainder else 0)
        positions = np.flatnonzero(y == class_id)
        labeled[rng.choice(positions, size=count, replace=False)] = True
        labeled_counts.append(count)
    return indices, y, labeled, counts, labeled_counts


def load_images(dataset, indices):
    """Flatten the chosen MNIST digits to [0, 1] pixel vectors."""
    rows = []
    for index in indices:
        image = np.asarray(dataset[int(index)]["image"], dtype=np.float64)
        rows.append(image.reshape(-1) / 255.0)
    return np.stack(rows, axis=0)


def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def knn_affinity(descriptors, k, gamma):
    """Equation (9): a_ij = [v_i · v_j]_+^γ when i is a k-NN of j."""
    n = descriptors.shape[0]
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


def label_matrix(y, labeled, n_classes):
    """Y is one-hot on labeled rows and zero on held-out rows."""
    matrix = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    rows = np.flatnonzero(labeled)
    matrix[rows, y[rows]] = 1.0
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
        min_dist=0.2,
        metric="cosine",
        random_state=SEED,
    )
    return reducer.fit_transform(descriptors)


def edge_segments(adjacency, xy):
    row, col = np.where(np.triu(adjacency, k=1) > 0)
    return np.stack([xy[row], xy[col]], axis=1)


def class_probabilities(scores):
    """Row-normalized class weights. `known` is False where no mass has arrived."""
    mass = scores.sum(axis=1)
    known = mass > 1e-12
    probabilities = np.zeros_like(scores)
    probabilities[known] = scores[known] / mass[known, None]
    return probabilities, known


def colors_from_probabilities(probabilities, known):
    rgba = np.empty((probabilities.shape[0], 4), dtype=np.float64)
    rgba[:] = matplotlib.colors.to_rgba(UNLABELED_GRAY)
    if np.any(known):
        rgb = probabilities[known] @ class_rgb()
        rgba[known, :3] = rgb
        rgba[known, 3] = 1.0
    return rgba


def hard_colors(classes, known):
    rgba = np.empty((classes.shape[0], 4), dtype=np.float64)
    rgba[:] = matplotlib.colors.to_rgba(UNLABELED_GRAY)
    for class_id, color in enumerate(CLASS_COLORS):
        rgba[known & (classes == class_id)] = matplotlib.colors.to_rgba(color)
    return rgba


def _scatter(ax, xy, rgba, labeled):
    """Triangles are seeds. Circles are unlabeled. Classes share one size."""
    if np.any(~labeled):
        ax.scatter(
            xy[~labeled, 0],
            xy[~labeled, 1],
            c=rgba[~labeled],
            s=16,
            marker="o",
            linewidths=0.15,
            edgecolors="#333333",
            zorder=2,
        )
    if np.any(labeled):
        ax.scatter(
            xy[labeled, 0],
            xy[labeled, 1],
            c=rgba[labeled],
            s=36,
            marker="^",
            linewidths=0.4,
            edgecolors="#222222",
            zorder=3,
        )


def _draw_graph(ax, segments):
    if len(segments) == 0:
        return
    ax.add_collection(
        LineCollection(
            segments,
            colors="#9aa0a6",
            linewidths=0.35,
            alpha=0.35,
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


def _legend_handles():
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="#333333",
            markersize=8,
            label=name,
        )
        for color, name in zip(CLASS_COLORS, class_names())
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=UNLABELED_GRAY,
            markeredgecolor="#333333",
            markersize=8,
            label="no mass yet",
        )
    )
    return handles


def _inches_to_fig(width_in, height_in):
    return width_in / FIGSIZE[0], height_in / FIGSIZE[1]


def _add_side_legend(fig, ax):
    position = ax.get_position()
    fig.legend(
        handles=_legend_handles(),
        loc="center left",
        bbox_to_anchor=(position.x1 + 0.02, position.y0 + position.height / 2),
        frameon=False,
        fontsize=10,
        title="class weight",
    )


def render_evolution_frame(path, xy, segments, labeled, scores, epoch, delta):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("white")
    side = 5.2
    width, height = _inches_to_fig(side, side)
    legend_w = 0.16
    gap = 0.02
    left = (1.0 - width - gap - legend_w) / 2
    bottom = 0.12
    ax = fig.add_axes([left, bottom, width, height])
    _draw_graph(ax, segments)
    probabilities, known = class_probabilities(scores)
    _scatter(ax, xy, colors_from_probabilities(probabilities, known), labeled)
    _style_axis(ax, xy)
    _add_side_legend(fig, ax)

    fig.suptitle(
        "Label propagation on MNIST digits 0, 1, and 2",
        fontsize=15,
        fontweight="bold",
        y=0.96,
    )
    if epoch == 0:
        subtitle = "Epoch 0 / 200    seeds only, unlabeled points in gray"
    else:
        subtitle = f"Epoch {epoch} / 200    ||ΔZ|| = {delta:.2e}"
    ax.set_title(subtitle, fontsize=12, pad=8)
    fig.text(
        0.42,
        0.02,
        "Color mixes the three class weights    Triangle = labeled seed    Circle = unlabeled",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def render_final_frame(path, xy, segments, labeled, scores, y):
    _, known = class_probabilities(scores)
    predicted = scores.argmax(axis=1)

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
        (axes[0], hard_colors(predicted, known), "Argmax of propagated weights"),
        (
            axes[1],
            hard_colors(y, np.ones(len(y), dtype=bool)),
            "Ground truth, including held-out labels",
        ),
    )
    for ax, rgba, title in panels:
        _draw_graph(ax, segments)
        _scatter(ax, xy, rgba, labeled)
        _style_axis(ax, xy)
        ax.set_title(title, fontsize=12, pad=8)
    fig.legend(
        handles=_legend_handles()[:3],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
    )
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
    _, known = class_probabilities(scores)
    predicted = scores.argmax(axis=1)
    unlabeled = ~labeled

    def accuracy(mask):
        if not np.any(mask):
            return None
        return float((predicted[mask] == y[mask]).mean())

    per_class = {}
    for class_id, name in enumerate(class_names()):
        mask = unlabeled & (y == class_id)
        per_class[name] = accuracy(mask)
    return {
        "unlabeled_accuracy": accuracy(unlabeled),
        "reached_unlabeled_accuracy": accuracy(unlabeled & known),
        "labeled_seed_accuracy": accuracy(labeled),
        "unlabeled_recall_per_class": per_class,
        "fraction_reached": float(known.mean()),
    }


def main():
    rng = np.random.default_rng(SEED)
    dataset, all_labels = get_mnist_labels()
    indices, labels, labeled, counts, labeled_counts = subsample(all_labels, rng)
    print(
        f"Loaded {DATASET_ID} ({len(all_labels)} train rows). "
        f"Subset counts {dict(zip(class_names(), counts))}. "
        f"Seeds {dict(zip(class_names(), labeled_counts))}."
    )
    features = load_images(dataset, indices)
    descriptors = l2_normalize(features)
    affinity = knn_affinity(descriptors, K_NEIGHBORS, GAMMA)
    adjacency, normalized = normalized_adjacency(affinity)
    seeds = label_matrix(labels, labeled, len(DIGITS))
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
            history[epoch],
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
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_components, _ = connected_components(csr_matrix(adjacency > 0), directed=False)
    summary = {
        "dataset": DATASET_ID,
        "classes": list(class_names()),
        "n_points": N_POINTS,
        "n_per_class": dict(zip(class_names(), counts)),
        "n_labeled": int(labeled.sum()),
        "n_labeled_per_class": dict(zip(class_names(), labeled_counts)),
        "features": "flattened 28x28 pixels, scaled to [0, 1], L2-normalized",
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
