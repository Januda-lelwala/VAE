"""Load a trained VAE and write reconstructions, prior samples, interpolations,
and (if latent_dim == 2) a decoded latent manifold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import VAE
from utils import get_device, save_grid

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate figures from a trained VAE")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "vae_final.pt")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--n-interp", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="MNIST test-set index to reconstruct (0..9999)",
    )
    p.add_argument(
        "--digit",
        type=int,
        default=None,
        help="Use the first MNIST test image with this label (0..9)",
    )
    p.add_argument(
        "--catalog",
        action="store_true",
        help="Write a numbered grid of test images so you can pick --index",
    )
    p.add_argument("--catalog-n", type=int, default=40, help="How many catalog images to show")
    return p.parse_args()


def load_model(path: Path, device: torch.device) -> VAE:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = VAE(latent_dim=ckpt["latent_dim"], arch=ckpt["arch"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def interpolate(model: VAE, x1: torch.Tensor, x2: torch.Tensor, steps: int) -> torch.Tensor:
    mu1, _ = model.encode(x1)
    mu2, _ = model.encode(x2)
    t = torch.linspace(0, 1, steps, device=x1.device).view(-1, 1)
    z = (1 - t) * mu1 + t * mu2
    return torch.sigmoid(model.decode(z))


@torch.no_grad()
def latent_manifold(model: VAE, n: int = 20, bound: float = 3.0) -> torch.Tensor:
    """Decode a grid over [-bound, bound]^2. Requires latent_dim == 2."""
    grid = torch.linspace(-bound, bound, n, device=next(model.parameters()).device)
    ys, xs = torch.meshgrid(grid, grid, indexing="ij")
    z = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
    return torch.sigmoid(model.decode(z))


def find_digit_index(dataset: datasets.MNIST, digit: int) -> int:
    targets = dataset.targets
    hits = (targets == digit).nonzero(as_tuple=False)
    if hits.numel() == 0:
        raise SystemExit(f"no MNIST test image with digit {digit}")
    return int(hits[0].item())


def save_catalog(dataset: datasets.MNIST, path: Path, n: int = 40, ncol: int = 8) -> None:
    n = min(n, len(dataset))
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.35, nrow * 1.55))
    axes_flat = axes.ravel() if n else []
    for i in range(n):
        img, label = dataset[i]
        axes_flat[i].imshow(img.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes_flat[i].set_title(f"{i}: {int(label)}", fontsize=8)
        axes_flat[i].axis("off")
    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")
    fig.suptitle("MNIST test set (index: digit) — pass --index N to reconstruct one")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_original_vs_recon(
    original: torch.Tensor,
    recon: torch.Tensor,
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.6))
    for ax, img, name in (
        (axes[0], original, "original"),
        (axes[1], recon, "VAE output"),
    ):
        ax.imshow(img.detach().cpu().squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def latent_scatter(model: VAE, loader: DataLoader, device: torch.device, path: Path, max_points: int = 5000) -> None:
    zs, ys = [], []
    seen = 0
    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        zs.append(mu.cpu())
        ys.append(y)
        seen += x.size(0)
        if seen >= max_points:
            break
    z = torch.cat(zs)[:max_points].numpy()
    y = torch.cat(ys)[:max_points].numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=6, alpha=0.7)
    fig.colorbar(scatter, ax=ax, ticks=range(10), label="digit")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("posterior means q(z|x)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}\ntrain first with: python train.py")

    model = load_model(args.checkpoint, device)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())

    if args.catalog:
        catalog_path = OUT_DIR / "sample_catalog.png"
        save_catalog(test_set, catalog_path, n=args.catalog_n)
        print(f"wrote {catalog_path}")

    if args.index is not None or args.digit is not None:
        if args.digit is not None and not 0 <= args.digit <= 9:
            raise SystemExit("--digit must be an integer 0..9")
        if args.index is not None:
            index = args.index
            if not 0 <= index < len(test_set):
                raise SystemExit(f"--index must be in 0..{len(test_set) - 1}")
        else:
            index = find_digit_index(test_set, args.digit)
        img, label = test_set[index]
        x = img.unsqueeze(0).to(device)
        recon = model.reconstruct(x)
        out_path = OUT_DIR / f"sample_{index:05d}_digit_{int(label)}.png"
        save_original_vs_recon(
            x,
            recon,
            out_path,
            title=f"test index {index}  (digit {int(label)})",
        )
        print(f"selected test index {index} (digit {int(label)})")
        print(f"wrote {out_path}")
        return

    if args.catalog:
        return

    loader = DataLoader(test_set, batch_size=128, shuffle=False)

    vis_x, _ = next(iter(DataLoader(test_set, batch_size=64, shuffle=False)))
    vis_x = vis_x.to(device)

    recon = model.reconstruct(vis_x)
    save_grid(torch.cat([vis_x, recon], dim=0), OUT_DIR / "reconstructions.png", nrow=8)
    save_grid(model.sample(args.n_samples, device=device), OUT_DIR / "samples.png", nrow=8)

    # One example of each digit, then interpolate 0→1, 1→2, ..., 9→0.
    by_digit: dict[int, torch.Tensor] = {}
    for img, label in test_set:
        label = int(label)
        if label not in by_digit:
            by_digit[label] = img.to(device).unsqueeze(0)
        if len(by_digit) == 10:
            break
    rows = []
    for a in range(10):
        xa = by_digit[a]
        xb = by_digit[(a + 1) % 10]
        morph = interpolate(model, xa, xb, args.n_interp)
        rows.append(torch.cat([xa, morph, xb], dim=0))
    save_grid(torch.cat(rows, dim=0), OUT_DIR / "interpolations.png", nrow=args.n_interp + 2)

    if model.latent_dim == 2:
        save_grid(latent_manifold(model), OUT_DIR / "manifold.png", nrow=20)
        latent_scatter(model, loader, device, OUT_DIR / "latent_scatter.png")

    print(f"wrote figures under {OUT_DIR}")


if __name__ == "__main__":
    main()
