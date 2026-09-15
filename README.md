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

## Generate

```bash
python generate.py --checkpoint checkpoints/vae_final.pt
```

Writes `outputs/reconstructions.png`, `outputs/samples.png`, and `outputs/interpolations.png`. If the checkpoint was trained with `--latent-dim 2`, also writes a decoded manifold and a scatter of posterior means colored by digit.

## Tests

```bash
python test_model.py
```
