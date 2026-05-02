# Flower VAE

A from-scratch PyTorch implementation of a Variational Autoencoder that learns to reconstruct and generate 128x128 flower images from a 256-dimensional latent space.

<p align="center">
  <img src="samples/fig_1.png" width="128" />
  <img src="samples/fig_2.png" width="128" />
  <img src="samples/fig_3.png" width="128" />
  <img src="samples/fig_4.png" width="128" />
  <img src="samples/fig_5.png" width="128" />
</p>

<p align="center"><i>Selected reconstructions from the trained VAE.</i></p>

---

## Why a VAE?

A vanilla autoencoder learns to compress an image into a code and reconstruct it. Nothing constrains where those codes land in latent space, so the encoder is free to scatter them into a discrete, jagged manifold full of holes. Sample a random point and decode it and you get noise — there is no reason a point between two training codes corresponds to anything meaningful.

A **Variational Autoencoder** fixes this by treating the encoder as a probabilistic mapping. Instead of producing one code per image, it outputs the parameters of a Gaussian — a mean μ and a log-variance log σ² — and a KL divergence term pulls every per-image Gaussian toward the standard normal prior **N(0, I)**. The latent space becomes continuous and densely populated: nearby points decode into perceptually similar images, and pure samples from N(0, I) fall on the data manifold.

The catch is that sampling z ~ N(μ, σ²) is non-differentiable. The **reparameterization trick** rewrites the sample as a deterministic function of μ, σ, and an external noise source ε:

```
z = μ + σ · ε,    ε ~ N(0, I)
```

Now the randomness lives outside the computation graph, and gradients flow cleanly through μ and σ back into the encoder.

The full training objective combines three terms:

```
L  =  MSE(x̂, x)                    ← reconstruction
   +  β · KL( N(μ, σ²)  ‖  N(0, I) )    ← latent regularization
   +  λ · ‖ φ(x̂) - φ(x) ‖²              ← perceptual loss
```

where φ is the VGG16 feature extractor up to relu4_3 (layers 0–22).

**Why add VGG perceptual loss on top of pixel MSE?** Pixel-wise MSE has a well-known failure mode: averaging over all plausible reconstructions of a slightly uncertain region gives the *blurry* average. The model is rewarded for hedging. VGG features, by contrast, encode texture and structure — matching them in feature space forces the decoder to commit to specific edges, petal patterns, and color transitions instead of smearing them out.

---

## How It Works

### The Encoder

Takes a 3×128×128 image and produces the parameters of a 256-dimensional Gaussian. Spatial resolution is halved at every stage by `MaxPool2d(2)`; channels grow to compensate.

```
img (3, 128, 128)
    ↓  DoubleDownConv(3 → 32)            → (32,  64, 64)
    ↓  DoubleDownConv(32 → 64)           → (64,  32, 32)
    ↓  DoubleDownConv(64 → 128)          → (128, 16, 16)
    ↓  DoubleDownConv(128 → 256)         → (256,  8,  8)
    ↓  last_encoder_layer (refine 256ch) → (256,  8,  8)
    ↓  hid_2_mean    : Conv → AdaptiveAvgPool(4,4) → Flatten → Linear → μ      ∈ R^256
    ↓  hid_2_logvar  : Conv → AdaptiveAvgPool(4,4) → Flatten → Linear → log σ² ∈ R^256
```

Each `DoubleDownConv` block is:

```
Conv3×3 → GroupNorm → GELU  →  Conv3×3 → GroupNorm → GELU  →  MaxPool2d(2)
```

The two heads `hid_2_mean` and `hid_2_logvar` share the same shape (a 1×1 conv to 128 channels, an `AdaptiveAvgPool2d((4,4))` that collapses the spatial grid, and a small MLP with `LayerNorm` + dropout) but have independent weights, so the network can decouple "where the code is" from "how confident I am in it."

### The Decoder

Takes a 256-d sample z and progressively upsamples it back to a 3×128×128 image.

```
z ∈ R^256
    ↓  Linear → Unflatten to (128, 8, 8) → Conv → GroupNorm → GELU
    ↓  DoubleUpConv(256 → 128)              → (128, 16, 16)
    ↓  DoubleUpConv(128 → 64)               → (64,  32, 32)
    ↓  DoubleUpConv(64  → 32)               → (32,  64, 64)
    ↓  DoubleUpConv(32  → 3,  last=True)    → (3,  128, 128)
    ↓  last_decoder_layer (refine 3ch RGB)
```

Each `DoubleUpConv` block is:

```
Conv3×3 → GroupNorm → GELU  →  Conv3×3 → GroupNorm → GELU  →  Upsample(scale=2, bilinear)
```

A few choices in the encoder/decoder are deliberate:

**GroupNorm instead of BatchNorm.** BatchNorm normalizes activations using statistics computed across the batch dimension. When the batch is small or the per-sample statistics are highly variable, those running statistics become noisy and unstable, and they leak information between samples in a way that is especially problematic for VAEs (where the per-sample posterior matters). GroupNorm splits channels into groups and normalizes each group within a single sample — no batch coupling, no train/eval drift.

**Bilinear upsampling + Conv instead of ConvTranspose2d.** `ConvTranspose2d` with stride 2 and a kernel that is not a multiple of the stride creates uneven overlap between adjacent kernel windows, which produces visible **checkerboard artifacts** in the output (Odena et al., 2016). Decoupling the upsample from the learned filter — bilinear upsampling first, then a regular `Conv2d` — eliminates the overlap problem entirely while letting the conv learn whatever refinement it needs.

---

## The Loss Function

The total loss is a sum of three components, all averaged over the batch:

**1. Reconstruction (pixel MSE).** Sum of squared pixel errors, divided by batch size. Drives the decoder to reproduce the input.

```python
rec_loss = F.mse_loss(y_preds, y, reduction='sum') / batch_size
```

**2. KL divergence.** Closed-form KL between the per-image Gaussian and the standard normal prior. Pulls the aggregate posterior toward N(0, I) so the latent space stays sample-able.

```python
kl_loss = beta * (-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))) / batch_size
```

`KL_BETA = 1.0` recovers the standard β-VAE formulation. Lower β biases toward sharper reconstructions at the cost of a less regular latent space; higher β biases toward a cleaner prior at the cost of mode collapse and blur.

**3. Perceptual loss (VGG16 features).** MSE in the feature space of a frozen ImageNet-pretrained VGG16, taken up to relu4_3 (layers 0–22). Multiplied by a large weight because feature-space activations are numerically much smaller than pixel-space differences.

```python
class VGGLoss(nn.Module):
    def __init__(self, weight, device):
        super().__init__()
        self.weight = weight
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.selected_layers = nn.Sequential(*list(vgg.children())[:23])
        for p in self.selected_layers.parameters():
            p.requires_grad_(False)

    def forward(self, x, y):
        return self.weight * F.mse_loss(self.selected_layers(x), self.selected_layers(y))
```

`VGG_LOSS_WEIGHT = 10000` is not arbitrary. Pixel-MSE on 128×128 RGB images sits in the thousands when summed; VGG feature-MSE at relu4_3 sits in the 10⁻¹–10⁻² range. The weight rescales the perceptual term so it actually contends with the reconstruction term during gradient updates instead of getting steamrolled.

---

## Project Structure

```
.
├── model.py            # Encoder, Decoder, VAE with reparameterization
├── train.py            # Training loop, VGG perceptual loss, checkpointing
├── config.py           # All hyperparameters and paths
├── vis_outs.py         # Load checkpoint, reconstruct and display samples
├── docs/
│   └── TRAINING.md     # Detailed training and resuming notes
├── samples/            # Example reconstructions from the trained model
├── checkpoints/        # Model weights — not tracked (except best_model.pth)
├── data/               # Oxford 102 Flower images — not tracked
├── requirements.txt
└── README.md
```

---

## Dataset

Trained on the **Oxford 102 Category Flower Dataset** (Nilsback & Zisserman, 2008) — 8,189 images across 102 flower categories.

- Images are resized to 140×140 and **center-cropped** to 128×128.
- Normalized with ImageNet statistics (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`) so VGG's feature extractor sees inputs in the distribution it was trained on.
- Light augmentation during training: random horizontal flip, saturation boost, color jitter.

The dataset is **not** included in this repo. Download the raw images from:

> https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

and unpack them so that flat image files (`.jpg`) live directly under `data/`.

---

## Getting Started

```bash
git clone https://github.com/franciszekparma/flovae.git
cd flovae
pip install -r requirements.txt
```

**Dependencies:** `torch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`, `tqdm`

### Prepare data

Download the Oxford 102 Flower images and place every `.jpg` file directly under `data/`. The dataset class will pick them up automatically.

### Train

```bash
python train.py
```

A checkpoint is written to `checkpoints/vae_epoch_{N}.pth` after every epoch, and `checkpoints/best_model.pth` is overwritten whenever the total loss improves.

To **resume** from a saved checkpoint, set the following in `config.py`:

```python
LOAD_WEIGHTS = True
RESUME_EPOCH = 128   # the epoch number you want to start from
```

The trainer will load `checkpoints/vae_epoch_128.pth` (model + optimizer state) and continue from epoch 129.

### Generate / reconstruct

```bash
python vis_outs.py
```

`vis_outs.py` loads `checkpoints/vae_epoch_{DISP_EPOCH}.pth`, encodes a random sample of real images, draws z via the reparameterization trick, decodes, denormalizes, and shows each reconstruction with `matplotlib`.

---

## Pretrained Weights

You don't need to train from scratch. The trained weights at epoch 256 are available here:

> **Google Drive:** https://drive.google.com/file/d/1d1xBw9PyHieS6IWYMfGlUz4XLphP0Qau/view?usp=sharing

Download the file, place it in `checkpoints/` as `vae_epoch_256.pth`, set `DISP_EPOCH = 256` in `config.py`, and run:

```bash
python vis_outs.py
```

The `checkpoints/` folder is otherwise not tracked in this repo, so this is the only way to get the trained weights without retraining.

---

## Hyperparameters

Everything lives in [`config.py`](config.py).

**Training**
| | |
|---|---|
| Batch size | 64 |
| Epochs | 256 |
| Learning rate | 5e-5 |
| Optimizer | AdamW |
| KL β | 1.0 |
| VGG perceptual weight | 10000 |
| Image size | 128 × 128 (resize 140 → center crop 128) |
| Augmentation | HFlip(0.5), saturation×1.3, ColorJitter(0.15, 0.1, 0.1, 0.05) |

**Model**
| | |
|---|---|
| Latent dim (z) | 256 |
| Encoder stages | 4 (DoubleDownConv) |
| Decoder stages | 4 (DoubleUpConv) |
| Channel progression | 3 → 32 → 64 → 128 → 256 |
| GroupNorm groups | 8 |
| Dropout (latent heads) | 0.3 |
| VGG cutoff | layers 0–22 (up to relu4_3) |

---

## References

- Kingma & Welling (2013). [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- Simonyan & Zisserman (2014). [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).
- Nilsback & Zisserman (2008). [Automated Flower Classification over a Large Number of Classes](https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/).
- Johnson, Alahi & Fei-Fei (2016). [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

---

## License

MIT &copy; franciszekparma
