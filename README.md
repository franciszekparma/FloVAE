# Flower VAE

> A from-scratch PyTorch Variational Autoencoder that reconstructs and generates 128×128 flower images from a 256-dimensional latent space.

<p align="center">
  <img src="samples/fig_1.png" width="128" />
  <img src="samples/fig_2.png" width="128" />
  <img src="samples/fig_3.png" width="128" />
  <img src="samples/fig_4.png" width="128" />
  <img src="samples/fig_5.png" width="128" />
</p>

<p align="center"><sub><i>Selected reconstructions from the trained VAE.</i></sub></p>

<p align="center">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" />
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img alt="Apple Silicon" src="https://img.shields.io/badge/Trained_on-M4_Max-555555?logo=apple&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## Why a VAE?

A vanilla autoencoder learns to compress an image into a code and reconstruct it — but nothing constrains where those codes land. Sample a random point and decode it: noise.

A **Variational Autoencoder** treats the encoder as a probabilistic mapping. Each image yields a Gaussian `N(μ, σ²)`, and a KL term pulls every per-image posterior toward the prior `N(0, I)`. The latent space becomes continuous: nearby points decode to perceptually similar images, and pure samples from `N(0, I)` land on the data manifold.

Sampling `z ~ N(μ, σ²)` is non-differentiable, so the **reparameterization trick** moves the randomness outside the graph:

```
z = μ + σ · ε,    ε ~ N(0, I)
```

The training objective:

```
L  =  MSE(x̂, x)                       ← reconstruction
   +  β · KL( N(μ, σ²) ‖ N(0, I) )    ← latent regularization
   +  λ · ‖ φ(x̂) − φ(x) ‖²            ← perceptual loss (VGG16, relu4_3)
```

> **Why VGG perceptual loss?** Pixel MSE rewards hedging — averaging over plausible reconstructions yields blur. Matching VGG features forces the decoder to commit to specific edges, petal patterns, and color transitions.

---

## Architecture

### Encoder · `(3, 128, 128) → μ, log σ² ∈ ℝ²⁵⁶`

```
img (3, 128, 128)
  ↓ DoubleDownConv ×4    →  (256, 8, 8)
  ↓ refine (Conv → GN → GELU → Conv)
  ↓ ┌─ hid_2_mean    : Conv → AdaptiveAvgPool(4,4) → MLP → μ
    └─ hid_2_logvar  : Conv → AdaptiveAvgPool(4,4) → MLP → log σ²
```

Each `DoubleDownConv` = `Conv → GN → GELU → Conv → GN → GELU → MaxPool(2)`.

### Decoder · `z ∈ ℝ²⁵⁶ → (3, 128, 128)`

```
z (256)
  ↓ Linear → Unflatten(128, 8, 8) → Conv → GN → GELU
  ↓ DoubleUpConv ×4      →  (3, 128, 128)
  ↓ refine RGB
```

Each `DoubleUpConv` = `Conv → GN → GELU → Conv → GN → GELU → Upsample(bilinear, ×2)`.

### Design choices

| Choice | Why |
|---|---|
| **GroupNorm** instead of BatchNorm | No batch coupling, no train/eval drift — important when per-image posteriors matter. |
| **Bilinear upsample + Conv** instead of `ConvTranspose2d` | Avoids checkerboard artifacts (Odena et al., 2016). |
| **Two independent heads** for μ and log σ² | Decouples *where the code is* from *how confident the encoder is*. |

---

## Loss

```python
rec_loss = F.mse_loss(y_preds, y, reduction='sum') / batch_size
kl_loss  = beta * (-0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp())) / batch_size
p_loss   = vgg_weight * F.mse_loss(vgg(y_preds), vgg(y))
```

`KL_BETA = 1.0` recovers standard β-VAE. `VGG_LOSS_WEIGHT = 10000` rescales the perceptual term — feature-MSE at relu4_3 sits around 10⁻¹–10⁻², while pixel-MSE on 128×128 RGB sits in the thousands.

---

## Project Structure

```
.
├── code/
│   ├── model.py        # Encoder, Decoder, VAE with reparameterization
│   ├── train.py        # Training loop, VGG perceptual loss, checkpointing
│   ├── config.py       # All hyperparameters and paths
│   └── vis_outs.py     # Load checkpoint, reconstruct and display samples
├── docs/TRAINING.md    # Detailed training and resuming notes
├── samples/            # Example reconstructions
├── checkpoints/        # Model weights — not tracked
├── data/               # Oxford 102 Flower images — not tracked
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/franciszekparma/flovae.git
cd flovae
pip install -r requirements.txt
```

**Dependencies:** `torch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`, `tqdm`

### 1 · Prepare data

Download the [Oxford 102 Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) images and drop every `.jpg` directly under `data/`.

### 2 · Train

```bash
python code/train.py
```

A checkpoint lands in `checkpoints/vae_epoch_{N}.pth` after every epoch. `checkpoints/best_model.pth` is overwritten whenever total loss improves.

To **resume**, set in `code/config.py`:

```python
LOAD_WEIGHTS = True
RESUME_EPOCH = 128
```

### 3 · Generate / reconstruct

```bash
python code/vis_outs.py
```

Loads `checkpoints/vae_epoch_{DISP_EPOCH}.pth`, encodes a random sample, draws `z` via reparameterization, decodes, denormalizes, and shows each reconstruction.

> All commands run from the repo root.

---

## Pretrained Weights

You don't need to train from scratch — epoch-256 weights are available:

> **Google Drive:** https://drive.google.com/file/d/1d1xBw9PyHieS6IWYMfGlUz4XLphP0Qau/view?usp=sharing

Drop the file in `checkpoints/` as `vae_epoch_256.pth`, set `DISP_EPOCH = 256` in `code/config.py`, and run `python code/vis_outs.py`.

---

## Dataset

**Oxford 102 Category Flower Dataset** (Nilsback & Zisserman, 2008) — 8,189 images across 102 categories.

- Resized to 140×140, **center-cropped** to 128×128
- Normalized with ImageNet stats so VGG sees its training distribution
- Augmentation: HFlip, saturation boost, ColorJitter

---

## Hyperparameters

All configurable in [`code/config.py`](code/config.py).

<table>
<tr><th align="left">Training</th><th></th><th align="left">Model</th><th></th></tr>
<tr><td>Batch size</td><td><code>64</code></td><td>Latent dim <code>z</code></td><td><code>256</code></td></tr>
<tr><td>Epochs</td><td><code>256</code></td><td>Encoder/Decoder stages</td><td><code>4</code></td></tr>
<tr><td>Learning rate</td><td><code>5e-5</code></td><td>Channels</td><td><code>3 → 32 → 64 → 128 → 256</code></td></tr>
<tr><td>Optimizer</td><td><code>AdamW</code></td><td>GroupNorm groups</td><td><code>8</code></td></tr>
<tr><td>KL β</td><td><code>1.0</code></td><td>Dropout (latent heads)</td><td><code>0.3</code></td></tr>
<tr><td>VGG weight</td><td><code>10000</code></td><td>VGG cutoff</td><td>layers 0–22 (relu4_3)</td></tr>
<tr><td>Image size</td><td>128 × 128</td><td></td><td></td></tr>
</table>

---

## Training Setup

Trained on a **MacBook Pro (M4 Max)** using PyTorch's MPS backend. The model auto-selects `cuda` if available and falls back to `mps` otherwise — see `DEVICE` in `code/config.py`.

---

## References

- Kingma & Welling (2013). [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- Simonyan & Zisserman (2014). [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).
- Nilsback & Zisserman (2008). [Automated Flower Classification over a Large Number of Classes](https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/).
- Johnson, Alahi & Fei-Fei (2016). [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

---

<p align="center"><sub>MIT &copy; franciszekparma</sub></p>
