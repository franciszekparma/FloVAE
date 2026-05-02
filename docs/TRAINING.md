# Training Notes

This document records the training setup used to produce the released checkpoint, what to expect from each loss term over time, and how to resume cleanly from a saved epoch.

---

## Setup

- **Hardware target.** A single CUDA GPU; an Apple-Silicon MPS fallback is available via `DEVICE = 'mps'` in `config.py`. With `BATCH_SIZE = 64` and a 128×128 input, peak GPU memory stays well under 12 GB, so most consumer cards are fine.
- **Optimizer.** `AdamW` with `lr = 5e-5` and default betas. Weight decay defaults to AdamW's standard 1e-2 — the `GroupNorm` layers and the frozen VGG mean we don't need to tune this further.
- **Schedule.** No LR scheduler. The loss surface for a VGG-weighted VAE is well-behaved enough that a constant low LR over 256 epochs reaches a sharp reconstruction without oscillation.
- **Augmentation.** `RandomHorizontalFlip(0.5)`, a fixed saturation boost (×1.3), and mild `ColorJitter(0.15, 0.1, 0.1, 0.05)`. These are deliberately gentle — heavier augmentation pushes the encoder to allocate latent capacity to nuisance variation rather than flower identity.
- **Normalization.** ImageNet mean/std. This matters: VGG16 was trained on ImageNet-normalized inputs, so feeding it [0, 1] tensors silently shifts the perceptual loss to a regime VGG was never optimized in, and the gradients become much less informative.

---

## Loss components and how they behave

The three loss terms move on very different schedules. Watching them individually (rather than just the sum) is the fastest way to diagnose training issues.

### Reconstruction loss (pixel MSE)

- **Drops fast.** In the first ~5 epochs the decoder learns gross color and shape, and pixel-MSE collapses by an order of magnitude.
- **Plateaus high.** After that initial drop, pixel-MSE stays roughly flat for the remainder of training. This is expected — beyond a certain point, lowering pixel-MSE further means averaging away high-frequency detail (the "blurry-mean" problem). The VGG term is what continues to improve sample quality from there on.

### KL divergence

- **Spikes early, then settles.** Early on, the encoder pushes μ around aggressively to fit different images, so KL grows. Within the first 20–30 epochs it stabilizes as the per-image posteriors find a stable arrangement under N(0, I).
- **Watch for posterior collapse.** If `kl_loss → 0` while reconstruction stalls high, the encoder has stopped using the latent (the decoder is producing the same image regardless of z). This is the classic VAE failure mode. Symptoms: `vis_outs.py` produces near-identical reconstructions for different inputs. With β=1.0 and the perceptual term weighted heavily this is rare, but if it happens, drop `KL_BETA` to 0.5 or 0.1 and continue.
- **Watch for KL blow-up.** The opposite failure: KL keeps growing without bound, meaning σ→0 (the encoder becomes deterministic, posteriors don't overlap, sampling is broken). With the current settings this hasn't been observed, but if the value crosses ~1000 something is wrong.

### VGG perceptual loss

- **Drops slowly and steadily.** This is the term that improves over the bulk of training. Texture, petal edge sharpness, and color fidelity all track with the perceptual loss.
- **Scaled by 10000 for a reason.** Without the rescale, pixel-MSE dominates the gradient and the perceptual term is decorative. After rescaling, all three losses sit in the same order of magnitude and contribute meaningfully to updates.

A rough log of what to expect (your numbers will differ depending on init):

| Epoch | Rec MSE | KL | VGG (×10000) |
|---:|---:|---:|---:|
| 1 | very high | low | very high |
| 10 | medium | medium | medium-high |
| 50 | low-medium | stable | medium |
| 128 | low | stable | low-medium |
| 256 | low (plateau) | stable | low (plateau) |

---

## Tips for resuming

The trainer in `train.py` supports clean resume out of the box. The mechanism:

1. Each epoch writes `checkpoints/vae_epoch_{epoch+1}.pth` containing
   - `model_state_dict`
   - `optimizer_state_dict`
   - `epoch`
   - `loss` (the total epoch loss for tracking the best model)
2. On startup, if `LOAD_WEIGHTS = True`, the trainer reads `checkpoints/vae_epoch_{RESUME_EPOCH}.pth` and starts from `epoch = RESUME_EPOCH + 1`.

Practical notes:

- **Set `RESUME_EPOCH` to the file you want to load**, not to "the next epoch I want to run." If you load `vae_epoch_128.pth`, training continues at epoch 129 automatically — no off-by-one needed.
- **Optimizer state matters.** AdamW carries first- and second-moment estimates per parameter; resuming without the optimizer state effectively warm-restarts the optimizer and the loss usually jumps for several epochs before recovering. The checkpoint format includes optimizer state for exactly this reason — don't strip it.
- **Best-model tracking resets to `inf` on resume in the current code.** That means a single noisy epoch right after resume can overwrite your saved best. If you care about preserving the best so far, copy `checkpoints/best_model.pth` aside before restarting.
- **Switching device on resume.** `torch.load` is called with `map_location=DEVICE`, so resuming a CUDA-trained checkpoint on MPS (or vice versa) Just Works. Don't worry about saving with `.cpu()` first.
- **Disk usage.** Checkpoints are written every epoch and they're not small. The `.gitignore` excludes them, but the local `checkpoints/` folder will balloon over a 256-epoch run — prune older epochs you don't need.
- **Determinism.** No global seeds are set, so two runs from the same epoch will diverge slightly. If you want reproducibility for a specific experiment, fix `torch.manual_seed(...)` and `random.seed(...)` at the top of `train.py`.

---

## Inspecting a run

Use `vis_outs.py` against a specific epoch by setting `DISP_EPOCH` in `config.py`. It loads `checkpoints/vae_epoch_{DISP_EPOCH}.pth`, encodes a random batch of real images, samples z via the reparameterization trick, and shows decoded reconstructions one by one.

Two things to look for when comparing epochs:

- **Sharpness.** Improvement here is the perceptual loss working. Compare a mid-training epoch (say 64) to the final epoch (256) — the difference in petal-edge crispness is what VGG buys you.
- **Latent diversity.** Encode several different images and see whether their reconstructions actually differ. If they collapse toward a single mean image, the latent has stopped carrying information (see "posterior collapse" above).
