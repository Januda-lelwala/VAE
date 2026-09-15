"""Variational autoencoder (Kingma & Welling, 2014), implemented from scratch.

The encoder parameterizes a diagonal Gaussian q(z|x) = N(mu(x), diag(sigma^2(x))).
The decoder parameterizes a Bernoulli p(x|z) whose logits are a neural net of z.
The prior is the unit Gaussian p(z) = N(0, I).

Training maximizes the evidence lower bound (ELBO):

    log p(x) >= E_{q(z|x)}[log p(x|z)] - KL(q(z|x) || p(z))

which is equivalent to minimizing reconstruction BCE plus the closed-form KL.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Arch = Literal["mlp", "conv"]


class MLPEncoder(nn.Module):
    """Fully-connected encoder used in the original VAE paper (MNIST)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 400) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 400) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1 * 28 * 28),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).view(-1, 1, 28, 28)


class ConvEncoder(nn.Module):
    """Stride-2 conv encoder: 28x28 -> 14x14 -> 7x7, then linear to (mu, logvar)."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.features(x)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20, arch: Arch = "conv") -> None:
        super().__init__()
        if arch not in ("mlp", "conv"):
            raise ValueError(f"arch must be 'mlp' or 'conv', got {arch!r}")
        self.latent_dim = latent_dim
        self.arch = arch
        if arch == "mlp":
            self.encoder: nn.Module = MLPEncoder(latent_dim)
            self.decoder: nn.Module = MLPDecoder(latent_dim)
        else:
            self.encoder = ConvEncoder(latent_dim)
            self.decoder = ConvDecoder(latent_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (mu, logvar) of the approximate posterior q(z|x)."""
        return self.encoder(x)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Draw z ~ q(z|x) via z = mu + sigma * eps, eps ~ N(0, I).

        Writing the sample as a deterministic function of (mu, logvar, eps)
        lets gradients flow into the encoder. log-variance is used instead of
        sigma so the scale stays positive without a softplus.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: Tensor) -> Tensor:
        """Return Bernoulli logits of p(x|z), shape (B, 1, 28, 28)."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return {"recon_logits": recon_logits, "mu": mu, "logvar": logvar, "z": z}

    @torch.no_grad()
    def reconstruct(self, x: Tensor, use_mean: bool = True) -> Tensor:
        """Decode the posterior mean (or a sample) and return probabilities."""
        mu, logvar = self.encode(x)
        z = mu if use_mean else self.reparameterize(mu, logvar)
        return torch.sigmoid(self.decode(z))

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | None = None) -> Tensor:
        """Draw x ~ p(x|z) p(z) by sampling z from the unit Gaussian prior."""
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z))


def elbo_loss(
    recon_logits: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
) -> dict[str, Tensor]:
    """Average -ELBO per datapoint: reconstruction BCE + beta * KL.

    For a diagonal Gaussian posterior and unit Gaussian prior the KL has
    the closed form

        KL = -1/2 * sum_i (1 + logvar_i - mu_i^2 - exp(logvar_i)).
    """
    batch = x.size(0)
    recon = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum") / batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
    return {"loss": recon + beta * kl, "recon": recon, "kl": kl}
