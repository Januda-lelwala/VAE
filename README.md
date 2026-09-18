# Variational Autoencoder from Scratch

A PyTorch implementation of the variational autoencoder from [Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114), trained on MNIST.

## What it implements

The encoder `q(z|x)` is a diagonal Gaussian `N(mu(x), diag(sigma^2(x)))`. The decoder `p(x|z)` is a Bernoulli whose logits are a neural net of `z`. The prior is `p(z) = N(0, I)`.

Sampling uses the **reparameterization trick**

```
z = mu + sigma * eps,   eps ~ N(0, I)
```

so the draw is a deterministic function of the encoder outputs and an independent noise source, which lets gradients reach the encoder.

Training maximizes the **ELBO**

```
log p(x)  >=  E_{q(z|x)}[log p(x|z)]  -  KL(q(z|x) || p(z))
```

which here is binary cross-entropy reconstruction plus the closed-form KL of two Gaussians, averaged per image.

Two encoders/decoders are included:

- `conv` (default) — stride-2 conv / conv-transpose, 28→14→7 and back
- `mlp` — the 784-400-400 fully-connected net from the original paper

`--beta` scales the KL term (a β-VAE). `--kl-warmup-epochs` anneals that weight in linearly.

## Setup

Python 3.12 (PyTorch has no 3.14 wheels yet):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Train

```bash
python train.py
python train.py --arch mlp --latent-dim 20 --epochs 20
python train.py --latent-dim 2 --epochs 30 --kl-warmup-epochs 5   # 2D latent plane
```

MNIST is downloaded to `data/`. Each epoch writes:

- `checkpoints/vae_latest.pt` (and `vae_final.pt` at the end)
- `outputs/recon_epoch_XXX.png` — original vs reconstruction
- `outputs/samples_epoch_XXX.png` — draws from the prior
- `outputs/loss_curve.png` and `outputs/history.json`

On Apple Silicon this uses MPS automatically.

## Train on Modal (GPU)

Same training loop, run on a cloud GPU. Checkpoints and figures land in a Volume named `vae-mnist` and are copied into local `checkpoints/` and `outputs/` when the job finishes.

```bash
uv pip install modal
modal setup
modal run modal_train.py
modal run modal_train.py --arch mlp --latent-dim 20 --epochs 20
modal run modal_train.py --latent-dim 2 --epochs 30 --kl-warmup-epochs 5 --gpu L4
```

GPU jobs need a payment method on the Modal account. Default GPU is a T4; pass `--gpu L4` / `--gpu A10` / `--gpu A100` to pick another. `modal run --detach modal_train.py` keeps the job running if your laptop disconnects.

Pull artifacts from a previous run without training again:

```bash
modal run modal_train.py --download-only
```

Or with the Volume CLI:

```bash
modal volume ls vae-mnist
modal volume get vae-mnist /checkpoints ./checkpoints
modal volume get vae-mnist /outputs ./outputs
```

## Generate

```bash
python generate.py --checkpoint checkpoints/vae_final.pt
```

Writes `outputs/reconstructions.png`, `outputs/samples.png`, and `outputs/interpolations.png`. If the checkpoint was trained with `--latent-dim 2`, also writes a decoded manifold and a scatter of posterior means colored by digit.

## Latent playground

An interactive page for dragging each latent mean `μ` and std `σ`, then decoding `p(x|z)`. Load a MNIST test digit to fill `q(z|x)`, switch between the posterior mean `z = μ` and a reparameterized sample `z = μ + σ·ε`, morph between digits, or sweep one coordinate while holding the rest fixed.

```bash
python app.py
```

Then open http://127.0.0.1:8000. Optional flags: `--checkpoint`, `--port`, `--host`.

## Tests

```bash
python test_model.py
python test_app.py
```
