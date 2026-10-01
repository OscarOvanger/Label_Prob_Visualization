# Label propagation on credit-card fraud

Visual demo of the transductive step in [Label Propagation for Deep Semi-supervised Learning](https://arxiv.org/abs/1904.04717) (Iscen, Tolias, Avrithis, Chum, 2019).

The movie projects transaction descriptors with UMAP, draws the 2-nearest-neighbor graph in that plane, and plays 200 epochs of label diffusion. Fraud is blue and non-fraud is red.

## What the movie shows

1. **Opening frame.** 2000 points. The 200 seeds are triangles: 4 fraud (blue) and 196 non-fraud (red). The other 1800 points are gray circles. Six of the ten fraud transactions are among those unlabeled points.
2. **Middle frames.** One frame per epoch. Every point is colored by its propagated class weight, from red (non-fraud) through pale (mixed) to blue (fraud). Points that still have no mass stay gray.
3. **Final frame.** Hard labels from `argmax` of the propagated weights, beside the ground truth for all 2000 points, including the labels that were held out.

With this many seeds the 2-NN graph is covered quickly: unlabeled points that share a component with a seed are reached in roughly the first 10 epochs. Later frames are the fixed point of the same iteration, which is what running all 200 epochs produces.

## Algorithm

Descriptors are the PCA features `V1`–`V28` plus `Amount` (the `Time` stamp is left out). They are standardized and L2-normalized, matching the unit-norm embeddings in the paper.

The graph follows equation (9): for each point, the `k = 2` nearest neighbors get affinity `[cosine]_+^3`, then the affinity is symmetrized and degree-normalized. Labels then diffuse with the Zhou et al. iteration the paper cites:

```
Z ← α S Z + (1 − α) Y,    α = 0.99
```

`Y` is one-hot on the 200 seeds and zero elsewhere. At convergence this matches the paper’s closed form up to the positive scale `(1 − α)`, so class ratios and `argmax` agree with equation (6).

## Data

`python main.py` first tries the requested Hub dataset `mlegrad/msbd5013-creditcard-fraud`. That repository is not publicly readable from this environment (the Hub returns 401). The script then loads `jyunyilin/credit-card-fraud-detection`, a public copy of the same ULB credit-card fraud table: 284,807 transactions, columns `Time`, `V1`–`V28`, `Amount`, and `Class`.

The demo keeps a deterministic subset of 2,000 rows with exactly 10 fraud cases (`numpy` seed 0).

## Run

```bash
pip install -r requirements.txt
python main.py
```

The first run downloads the table (about 150 MB) into the Hugging Face cache. Outputs:

| File | Contents |
| --- | --- |
| `outputs/label_propagation.mp4` | 200 frames at 10 fps, then the final comparison is held for 3 seconds |
| `outputs/label_propagation.gif` | same movie as a GIF |
| `outputs/keyframes/` | opening, early epochs, and the final comparison |
| `outputs/summary.json` | graph stats and held-out accuracy |

`ffmpeg` is required for the mp4 and gif.
