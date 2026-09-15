from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torchvision.utils import save_image


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_grid(images: Tensor, path: Path, nrow: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.clamp(0, 1).cpu(), path, nrow=nrow)
