"""Checks for the latent playground helpers and HTTP API."""

from __future__ import annotations

import base64

import torch
from fastapi.testclient import TestClient

from app import app, image_to_b64_u8, kl_to_standard_normal


def test_kl_zero_at_prior() -> None:
    mu = torch.zeros(1, 7)
    logvar = torch.zeros(1, 7)
    assert abs(kl_to_standard_normal(mu, logvar)) < 1e-6


def test_image_to_b64_is_784_bytes() -> None:
    x = torch.full((1, 28, 28), 0.5)
    raw = base64.b64decode(image_to_b64_u8(x))
    assert len(raw) == 784
    assert raw[0] == 128


def test_api_decode_and_encode() -> None:
    with TestClient(app) as client:
        info = client.get("/api/info").json()
        dim = info["latent_dim"]
        assert dim >= 1
        page = client.get("/")
        assert page.status_code == 200
        assert b"Latent Playground" in page.content

        decoded = client.post(
            "/api/decode",
            json={"mu": [0.0] * dim, "sigma": [1.0] * dim, "mode": "mean"},
        )
        assert decoded.status_code == 200, decoded.text
        body = decoded.json()
        assert len(base64.b64decode(body["pixels"])) == 784
        assert len(body["z"]) == dim
        assert abs(body["kl"]) < 1e-4

        encoded = client.post("/api/encode", json={"digit": 7})
        assert encoded.status_code == 200, encoded.text
        enc = encoded.json()
        assert enc["label"] == 7
        assert len(enc["mu"]) == dim
        assert len(enc["sigma"]) == dim
        assert all(s >= 0 for s in enc["sigma"])
        assert len(base64.b64decode(enc["original"])) == 784
        assert len(base64.b64decode(enc["recon"])) == 784

        sampled = client.post(
            "/api/samples",
            json={"mu": enc["mu"], "sigma": enc["sigma"], "mode": "sample", "n": 4},
        )
        assert sampled.status_code == 200, sampled.text
        assert len(sampled.json()["images"]) == 4

        bad = client.post("/api/decode", json={"mu": [0.0], "sigma": [1.0, 2.0]})
        assert bad.status_code == 422


if __name__ == "__main__":
    test_kl_zero_at_prior()
    print("ok  kl at prior")
    test_image_to_b64_is_784_bytes()
    print("ok  png bytes")
    test_api_decode_and_encode()
    print("ok  api")
    print("all playground tests passed")
