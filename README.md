# Label propagation on three MNIST digits

Visual demo of the transductive step in [Label Propagation for Deep Semi-supervised Learning](https://arxiv.org/abs/1904.04717) (Iscen, Tolias, Avrithis, Chum, 2019).

<video src="https://github.com/OscarOvanger/Label_Prob_Visualization/raw/main/outputs/label_propagation.mp4" controls autoplay loop muted playsinline width="800"></video>

The same movie, as a GIF, for viewers that do not play the file above:

![Label propagation on MNIST digits 0, 1, and 2](outputs/label_propagation.gif)

[Download the mp4](outputs/label_propagation.mp4) (200 frames at 10 fps, then the final comparison is held for 3 seconds).

The movie projects digit images with UMAP, draws the 2-nearest-neighbor graph in that plane, and plays 200 epochs of label diffusion. Each image has one of three labels: digit 0 (red), digit 1 (blue), or digit 2 (green). A soft label mixes those colors in proportion to the three class weights.

## What the movie shows

1. **Opening frame.** 2000 points, split as evenly as possible across the three digits. The 200 seeds are triangles, also split across the three classes. The other 1800 points are gray circles.
2. **Middle frames.** One frame per epoch. Color is the mix of the three class weights, and it stays pale until enough mass has propagated to that point.
3. **Final frame.** Hard labels from `argmax` of the propagated weights, beside the ground truth for all 2000 points, including the labels that were held out.

Each epoch takes a small step, so the three colors creep outward over the full 200 frames.

## Algorithm

Descriptors are the 28×28 pixels scaled to [0, 1] and L2-normalized, the same unit-norm embedding step the paper uses before building the graph.

The graph follows equation (9): for each point, the `k = 2` nearest neighbors get affinity `[cosine]_+^3`, then the affinity is symmetrized and degree-normalized. One full Zhou step, the update cited in the paper, is

```
Z* = 0.99 S Z + 0.01 Y
```

`Y` is one-hot on the 200 seeds and zero elsewhere. A full step of that update repaints the next graph hop in a single epoch, so the picture looks finished almost immediately. Each epoch instead moves only a fraction α of the way toward Z*:

```
Z ← (1 − α) Z + α Z*,    α = 0.05
```

The destination is still the paper’s diffusion. The small step is what makes the spread last for the whole movie. A point is drawn in a full class color only once its propagated mass reaches 0.08. Before that, the color is mixed with gray.

## Data

The images come from [`ylecun/mnist`](https://huggingface.co/datasets/ylecun/mnist) on Hugging Face. The demo keeps a deterministic subset of 2,000 training digits (`numpy` seed 0): digits 0, 1, and 2 only.

Each point has a single class. The propagation step ends in an `argmax`, so this is a three-class problem rather than a multi-label one in which one image would carry several labels at once.

## Run

```bash
pip install -r requirements.txt
python main.py
```

The first run downloads MNIST into the Hugging Face cache. Outputs:

| File | Contents |
| --- | --- |
| `outputs/label_propagation.mp4` | 200 frames at 10 fps, then the final comparison is held for 3 seconds |
| `outputs/label_propagation.gif` | same movie as a GIF |
| `outputs/keyframes/` | opening, early epochs, and the final comparison |
| `outputs/summary.json` | graph stats and held-out accuracy |

`ffmpeg` is required for the mp4 and gif.
