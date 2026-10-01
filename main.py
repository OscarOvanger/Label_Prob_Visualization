"""Label propagation on MNIST digits 0, 1, and 2.

Visual demo of the transductive step in Iscen et al.,
"Label Propagation for Deep Semi-supervised Learning" (arXiv:1904.04717).

Change the block below to retarget the demo. The movie is a 2-NN graph
drawn on a fixed UMAP, with one frame per diffusion epoch.
"""

from __future__ import annotations

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

# --- knobs -----------------------------------------------------------------
N_POINTS = 2000
N_LABELED = 200
K_NEIGHBORS = 2
GAMMA = 3.0
DIFFUSION = 0.99  # weight on S Z inside one full Zhou step
ALPHA = 0.05  # fraction of that step taken each epoch
MASS_SCALE = 0.08  # propagated mass drawn as a full class color
N_EPOCHS = 200
SEED = 0
DATASET_ID = "ylecun/mnist"
DIGITS = (0, 1, 2)
CLASS_COLORS = ("#e41a1c", "#377eb8", "#4daf4a")  # digits 0, 1, 2
UNLABELED_GRAY = "#bdbdbd"
FIGSIZE = (12.8, 7.2)
DPI = 100
# ---------------------------------------------------------------------------

OUT = Path(__file__).resolve().parent / "outputs"
FRAMES = OUT / "frames"
CLASS_RGB = np.array([matplotlib.colors.to_rgb(c) for c in CLASS_COLORS], dtype=np.float64)
GRAY_RGB = np.array(matplotlib.colors.to_rgb(UNLABELED_GRAY), dtype=np.float64)


def even_counts(total, n):
    base, extra = divmod(total, n)
    return [base + (i < extra) for i in range(n)]


def load_subset(rng):
    """Even split of N_POINTS across DIGITS, then an even split of N_LABELED seeds."""
    from datasets import load_dataset

    data = load_dataset(DATASET_ID, split="train")
    labels = np.asarray(data["label"], dtype=np.int64)
    chosen, subset = [], []
    for class_id, (digit, count) in enumerate(zip(DIGITS, even_counts(N_POINTS, len(DIGITS)))):
        pool = np.flatnonzero(labels == digit)
        if len(pool) < count:
            raise ValueError(f"Digit {digit} has {len(pool)} rows, need {count}.")
        chosen.append(rng.choice(pool, size=count, replace=False))
        subset.append(np.full(count, class_id, dtype=np.int64))
    order = rng.permutation(N_POINTS)
    indices = np.concatenate(chosen)[order]
    y = np.concatenate(subset)[order]
    labeled = np.zeros(N_POINTS, dtype=bool)
    for class_id, count in enumerate(even_counts(N_LABELED, len(DIGITS))):
        labeled[rng.choice(np.flatnonzero(y == class_id), size=count, replace=False)] = True
    pixels = np.stack(
        [np.asarray(data[int(i)]["image"], dtype=np.float64).reshape(-1) / 255.0 for i in indices]
    )
    return pixels / np.clip(np.linalg.norm(pixels, axis=1, keepdims=True), 1e-12, None), y, labeled


def knn_diffusion(descriptors):
    """Affinity (eq. 9), symmetrized and degree-normalized to S."""
    n = len(descriptors)
    _, index = NearestNeighbors(n_neighbors=K_NEIGHBORS + 3, metric="euclidean").fit(
        descriptors
    ).kneighbors(descriptors)
    nbr = np.empty((n, K_NEIGHBORS), dtype=np.int64)
    for j, row in enumerate(index):
        others = row[row != j][:K_NEIGHBORS]
        if len(others) < K_NEIGHBORS:
            raise RuntimeError(f"Point {j} has fewer than {K_NEIGHBORS} neighbors.")
        nbr[j] = others
    rows, cols = nbr.ravel(), np.repeat(np.arange(n), K_NEIGHBORS)
    weights = np.clip((descriptors[rows] * descriptors[cols]).sum(1), 0, None) ** GAMMA
    affinity = np.zeros((n, n))
    np.add.at(affinity, (rows, cols), weights)
    np.fill_diagonal(affinity, 0)
    adjacency = affinity + affinity.T
    np.fill_diagonal(adjacency, 0)
    degree = adjacency.sum(1)
    inv = np.zeros_like(degree)
    inv[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    return adjacency, inv[:, None] * adjacency * inv[None, :]


def propagate(normalized, seeds):
    """Z* = DIFFUSION * S Z + (1 - DIFFUSION) * Y, then a step of size ALPHA toward Z*."""
    current = seeds.copy()
    history = [current.copy()]
    for _ in range(N_EPOCHS):
        target = DIFFUSION * (normalized @ current) + (1.0 - DIFFUSION) * seeds
        current = (1.0 - ALPHA) * current + ALPHA * target
        history.append(current.copy())
    return history


def embed(descriptors):
    import umap

    return umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.2, metric="cosine", random_state=SEED
    ).fit_transform(descriptors)


def soft_colors(scores):
    """Class mix, faded toward gray until the propagated mass reaches MASS_SCALE."""
    mass = scores.sum(1)
    known = mass > 1e-12
    probs = np.zeros_like(scores)
    probs[known] = scores[known] / mass[known, None]
    fade = np.clip(mass / MASS_SCALE, 0, 1)[known, None]
    rgba = np.empty((len(scores), 4))
    rgba[:, :3] = GRAY_RGB
    rgba[:, 3] = 1
    rgba[known, :3] = (1 - fade) * GRAY_RGB + fade * (probs[known] @ CLASS_RGB)
    return rgba, known


def hard_colors(classes, known):
    rgba = np.tile(matplotlib.colors.to_rgba(UNLABELED_GRAY), (len(classes), 1))
    for class_id, color in enumerate(CLASS_COLORS):
        rgba[known & (classes == class_id)] = matplotlib.colors.to_rgba(color)
    return rgba


def draw(ax, xy, segments, rgba, labeled):
    if len(segments):
        ax.add_collection(
            LineCollection(segments, colors="#9aa0a6", linewidths=0.35, alpha=0.35, zorder=1)
        )
    for mask, marker, size, edge, lw, z in (
        (~labeled, "o", 16, "#333333", 0.15, 2),
        (labeled, "^", 36, "#222222", 0.4, 3),
    ):
        if mask.any():
            ax.scatter(
                xy[mask, 0], xy[mask, 1], c=rgba[mask], s=size, marker=marker,
                linewidths=lw, edgecolors=edge, zorder=z,
            )
    center = xy.mean(0)
    half = 0.5 * np.max(xy.max(0) - xy.min(0)) * 1.12
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#e0e0e0")


def legend(include_gray=True):
    handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=color,
            markeredgecolor="#333333", markersize=8, label=f"digit {digit}",
        )
        for digit, color in zip(DIGITS, CLASS_COLORS)
    ]
    if include_gray:
        handles.append(
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=UNLABELED_GRAY,
                markeredgecolor="#333333", markersize=8, label="no mass yet",
            )
        )
    return handles


def add_axes(fig, side_in, left, bottom):
    return fig.add_axes([left, bottom, side_in / FIGSIZE[0], side_in / FIGSIZE[1]])


def save_evolution(path, xy, segments, labeled, scores, epoch, delta):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("white")
    side, legend_w, gap = 5.2, 0.16, 0.02
    ax = add_axes(fig, side, (1 - side / FIGSIZE[0] - gap - legend_w) / 2, 0.12)
    draw(ax, xy, segments, soft_colors(scores)[0], labeled)
    box = ax.get_position()
    fig.legend(
        handles=legend(), loc="center left", frameon=False, fontsize=10, title="class weight",
        bbox_to_anchor=(box.x1 + 0.02, box.y0 + box.height / 2),
    )
    fig.suptitle(
        "Label propagation on MNIST digits 0, 1, and 2", fontsize=15, fontweight="bold", y=0.96
    )
    ax.set_title(
        "Epoch 0 / 200    seeds only, unlabeled points in gray"
        if epoch == 0
        else f"Epoch {epoch} / 200    α = {ALPHA:g}    ||ΔZ|| = {delta:.2e}",
        fontsize=12, pad=8,
    )
    fig.text(
        0.42, 0.02,
        "Pale color is a small propagated mass    Triangle = labeled seed    Circle = unlabeled",
        ha="center", va="bottom", fontsize=9, color="#333333",
    )
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def save_final(path, xy, segments, labeled, scores, y):
    _, known = soft_colors(scores)
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("white")
    side, gap = 4.55, 0.07
    width = side / FIGSIZE[0]
    left = (1 - 2 * width - gap) / 2
    panels = (
        (add_axes(fig, side, left, 0.16), hard_colors(scores.argmax(1), known), "Argmax of propagated weights"),
        (add_axes(fig, side, left + width + gap, 0.16), hard_colors(y, np.ones(len(y), dtype=bool)), "Ground truth, including held-out labels"),
    )
    fig.suptitle(
        "Epoch 200    argmax pseudo-labels beside held-out ground truth",
        fontsize=15, fontweight="bold", y=0.95,
    )
    for ax, rgba, title in panels:
        draw(ax, xy, segments, rgba, labeled)
        ax.set_title(title, fontsize=12, pad=8)
    fig.legend(
        handles=legend(include_gray=False), loc="lower center", ncol=3,
        frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.02),
    )
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def encode():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the movie.")
    hold = "tpad=stop_mode=clone:stop_duration=3"
    gif_vf = hold + ",scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse"
    for dest, extra in ((OUT / "label_propagation.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]),
                        (OUT / "label_propagation.gif", [])):
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "10", "-start_number", "0", "-i", str(FRAMES / "frame_%03d.png"),
             "-vf", hold if dest.suffix == ".mp4" else gif_vf, *extra, str(dest)],
            check=True, capture_output=True,
        )


def main():
    rng = np.random.default_rng(SEED)
    descriptors, y, labeled = load_subset(rng)
    adjacency, normalized = knn_diffusion(descriptors)
    seeds = np.zeros((N_POINTS, len(DIGITS)))
    rows = np.flatnonzero(labeled)
    seeds[rows, y[rows]] = 1
    history = propagate(normalized, seeds)
    xy = embed(descriptors)
    row, col = np.where(np.triu(adjacency, k=1) > 0)
    segments = np.stack([xy[row], xy[col]], axis=1)

    FRAMES.mkdir(parents=True, exist_ok=True)
    for epoch in range(N_EPOCHS - 1):
        delta = 0.0 if epoch == 0 else float(np.linalg.norm(history[epoch] - history[epoch - 1]))
        save_evolution(
            FRAMES / f"frame_{epoch:03d}.png", xy, segments, labeled, history[epoch], epoch, delta
        )
        if epoch % 20 == 0:
            print(f"Rendered epoch {epoch}")
    save_final(FRAMES / f"frame_{N_EPOCHS - 1:03d}.png", xy, segments, labeled, history[N_EPOCHS], y)

    pred = history[N_EPOCHS].argmax(1)
    held_out = ~labeled
    print(f"Unlabeled accuracy {(pred[held_out] == y[held_out]).mean():.4f} on {held_out.sum()} points.")
    print("Encoding mp4 and gif...")
    encode()
    print(f"Wrote {OUT / 'label_propagation.mp4'}")
    print(f"Wrote {OUT / 'label_propagation.gif'}")


if __name__ == "__main__":
    main()
