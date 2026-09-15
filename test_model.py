"""Sanity checks for the VAE: shapes, closed-form KL, and gradient flow."""

from __future__ import annotations

import torch

from model import VAE, elbo_loss


def test_forward_shapes(arch: str) -> None:
    model = VAE(latent_dim=8, arch=arch)
    x = torch.rand(4, 1, 28, 28)
    out = model(x)
    assert out["recon_logits"].shape == (4, 1, 28, 28), out["recon_logits"].shape
    assert out["mu"].shape == (4, 8)
    assert out["logvar"].shape == (4, 8)
    assert out["z"].shape == (4, 8)
    assert model.reconstruct(x).shape == (4, 1, 28, 28)
    assert model.sample(5).shape == (5, 1, 28, 28)


def test_kl_zero_at_prior() -> None:
    mu = torch.zeros(3, 7)
    logvar = torch.zeros(3, 7)
    x = torch.rand(3, 1, 28, 28)
    logits = torch.zeros_like(x)
    metrics = elbo_loss(logits, x, mu, logvar, beta=1.0)
    assert torch.allclose(metrics["kl"], torch.tensor(0.0), atol=1e-6), metrics["kl"]


def test_gradients_reach_encoder_and_decoder() -> None:
    model = VAE(latent_dim=4, arch="mlp")
    x = torch.rand(2, 1, 28, 28)
    out = model(x)
    loss = elbo_loss(out["recon_logits"], x, out["mu"], out["logvar"])["loss"]
    loss.backward()
    enc_grad = model.encoder.fc_mu.weight.grad
    dec_grad = model.decoder.net[-1].weight.grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0
    assert dec_grad is not None and dec_grad.abs().sum() > 0


def test_reparameterize_is_stochastic_and_centered() -> None:
    model = VAE(latent_dim=16, arch="mlp")
    mu = torch.zeros(256, 16)
    logvar = torch.zeros(256, 16)
    z = model.reparameterize(mu, logvar)
    assert not torch.allclose(z, model.reparameterize(mu, logvar))
    assert z.mean().abs() < 0.1
    assert (z.std() - 1).abs() < 0.1


if __name__ == "__main__":
    for arch in ("mlp", "conv"):
        test_forward_shapes(arch)
        print(f"ok  shapes[{arch}]")
    test_kl_zero_at_prior()
    print("ok  kl at prior")
    test_gradients_reach_encoder_and_decoder()
    print("ok  gradients")
    test_reparameterize_is_stochastic_and_centered()
    print("ok  reparameterize")
    print("all tests passed")
