"""Interactive latent playground for a trained MNIST VAE.

Serves a small web UI where you set the mean μ and std σ of q(z),
optionally sample z = μ + σ·ε, and decode p(x|z) live.
"""

from __future__ import annotations

import argparse
import base64
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from torch import Tensor
from torchvision import datasets, transforms

from model import VAE
from utils import get_device

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DEFAULT_CKPT = ROOT / "checkpoints" / "vae_final.pt"
DEFAULT_DATA = ROOT / "data"

settings = {
    "checkpoint": DEFAULT_CKPT,
    "data_dir": DEFAULT_DATA,
}


@dataclass
class Runtime:
    model: VAE
    dataset: datasets.MNIST
    device: torch.device
    checkpoint: Path
    epoch: int
    arch: str


rt: Runtime | None = None


def load_runtime(checkpoint: Path, data_dir: Path) -> Runtime:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}\ntrain first with: python train.py"
        )
    device = get_device()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = VAE(latent_dim=ckpt["latent_dim"], arch=ckpt["arch"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    with torch.no_grad():
        model.decode(torch.zeros(1, model.latent_dim, device=device))
    dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    return Runtime(
        model=model,
        dataset=dataset,
        device=device,
        checkpoint=checkpoint,
        epoch=int(ckpt.get("epoch", 0)),
        arch=str(ckpt["arch"]),
    )


def get_rt() -> Runtime:
    if rt is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return rt


def image_to_b64_u8(image: Tensor) -> str:
    """Flatten a (1,28,28) or (28,28) tensor in [0, 1] to base64-encoded 784 uint8s."""
    arr = (image.detach().cpu().clamp(0, 1).reshape(-1).float() * 255.0).round().to(torch.uint8)
    return base64.b64encode(arr.numpy().tobytes()).decode("ascii")


def kl_to_standard_normal(mu: Tensor, logvar: Tensor) -> float:
    return float((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum().item())


def as_row(values: list[float], dim: int, name: str, device: torch.device) -> Tensor:
    if len(values) != dim:
        raise HTTPException(status_code=400, detail=f"{name} must have length {dim}")
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, dim)


@torch.no_grad()
def decode_from_q(
    model: VAE,
    mu: Tensor,
    sigma: Tensor,
    mode: Literal["mean", "sample"],
    eps: Tensor | None,
) -> dict:
    sigma = sigma.clamp(min=0)
    if mode == "mean":
        z = mu
        if eps is None:
            eps = torch.zeros_like(mu)
    else:
        if eps is None:
            eps = torch.randn_like(mu)
        z = mu + sigma * eps
    probs = torch.sigmoid(model.decode(z))
    logvar = (sigma.clamp(min=1e-8).pow(2)).log()
    return {
        "pixels": image_to_b64_u8(probs[0]),
        "z": z.squeeze(0).cpu().tolist(),
        "eps": eps.squeeze(0).cpu().tolist(),
        "kl": kl_to_standard_normal(mu, logvar),
    }


@torch.no_grad()
def encode_index(runtime: Runtime, index: int) -> dict:
    n = len(runtime.dataset)
    if not 0 <= index < n:
        raise HTTPException(status_code=400, detail=f"index must be in 0..{n - 1}")
    img, label = runtime.dataset[index]
    x = img.unsqueeze(0).to(runtime.device)
    mu, logvar = runtime.model.encode(x)
    sigma = torch.exp(0.5 * logvar)
    recon = torch.sigmoid(runtime.model.decode(mu))
    return {
        "index": index,
        "label": int(label),
        "mu": mu.squeeze(0).cpu().tolist(),
        "sigma": sigma.squeeze(0).cpu().tolist(),
        "original": image_to_b64_u8(img),
        "recon": image_to_b64_u8(recon[0]),
        "kl": kl_to_standard_normal(mu, logvar),
    }


def pick_digit_index(runtime: Runtime, digit: int) -> int:
    if not 0 <= digit <= 9:
        raise HTTPException(status_code=400, detail="digit must be 0..9")
    hits = (runtime.dataset.targets == digit).nonzero(as_tuple=False).view(-1)
    if hits.numel() == 0:
        raise HTTPException(status_code=404, detail=f"no test image with digit {digit}")
    return int(hits[int(torch.randint(0, hits.numel(), (1,)).item())].item())


class DecodeIn(BaseModel):
    mu: list[float]
    sigma: list[float]
    mode: Literal["mean", "sample"] = "mean"
    eps: list[float] | None = None

    @model_validator(mode="after")
    def matching_lengths(self) -> DecodeIn:
        if len(self.mu) != len(self.sigma):
            raise ValueError("mu and sigma must have the same length")
        if self.eps is not None and len(self.eps) != len(self.mu):
            raise ValueError("eps must have the same length as mu")
        if any(s < 0 for s in self.sigma):
            raise ValueError("sigma must be >= 0")
        return self


class EncodeIn(BaseModel):
    index: int | None = None
    digit: int | None = Field(default=None, ge=0, le=9)


class SamplesIn(DecodeIn):
    n: int = Field(default=8, ge=1, le=16)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rt
    rt = load_runtime(Path(settings["checkpoint"]), Path(settings["data_dir"]))
    yield
    rt = None


app = FastAPI(title="VAE Latent Playground", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/info")
def info() -> dict:
    runtime = get_rt()
    return {
        "latent_dim": runtime.model.latent_dim,
        "arch": runtime.arch,
        "epoch": runtime.epoch,
        "device": str(runtime.device),
        "n_test": len(runtime.dataset),
        "checkpoint": str(runtime.checkpoint.name),
        "mu_min": -5.0,
        "mu_max": 5.0,
        "sigma_min": 0.0,
        "sigma_max": 3.0,
    }


@app.post("/api/decode")
def decode(body: DecodeIn) -> dict:
    runtime = get_rt()
    dim = runtime.model.latent_dim
    mu = as_row(body.mu, dim, "mu", runtime.device)
    sigma = as_row(body.sigma, dim, "sigma", runtime.device)
    eps = None if body.eps is None else as_row(body.eps, dim, "eps", runtime.device)
    return decode_from_q(runtime.model, mu, sigma, body.mode, eps)


@app.post("/api/encode")
def encode(body: EncodeIn) -> dict:
    runtime = get_rt()
    if body.index is not None:
        index = body.index
    elif body.digit is not None:
        index = pick_digit_index(runtime, body.digit)
    else:
        index = random.randrange(len(runtime.dataset))
    return encode_index(runtime, index)


@app.post("/api/samples")
def samples(body: SamplesIn) -> dict:
    runtime = get_rt()
    dim = runtime.model.latent_dim
    mu = as_row(body.mu, dim, "mu", runtime.device)
    sigma = as_row(body.sigma, dim, "sigma", runtime.device)
    eps = torch.randn(body.n, dim, device=runtime.device)
    z = mu.expand(body.n, -1) + sigma.clamp(min=0) * eps
    with torch.no_grad():
        probs = torch.sigmoid(runtime.model.decode(z))
    return {"images": [image_to_b64_u8(p) for p in probs]}


@app.get("/api/catalog")
def catalog(n: int = 32, offset: int = 0) -> dict:
    runtime = get_rt()
    n_test = len(runtime.dataset)
    offset = max(0, min(offset, n_test - 1))
    n = max(1, min(n, 64))
    items = []
    for i in range(offset, min(offset + n, n_test)):
        img, label = runtime.dataset[i]
        items.append({"index": i, "label": int(label), "pixels": image_to_b64_u8(img)})
    return {"items": items, "n_test": n_test}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive VAE latent playground")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings["checkpoint"] = args.checkpoint
    settings["data_dir"] = args.data_dir
    print(f"playground → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
