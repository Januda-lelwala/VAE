"""Train a VAE on MNIST and dump reconstructions, samples, and a loss curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import VAE, elbo_loss
from utils import get_device, save_grid, seed_everything

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a variational autoencoder on MNIST")
    p.add_argument("--arch", choices=("conv", "mlp"), default="conv")
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0, help="Weight on the KL term (beta-VAE)")
    p.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=0,
        help="Linearly anneal beta from 0 to --beta over this many epochs",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def beta_at_epoch(epoch: int, beta: float, warmup: int) -> float:
    if warmup <= 0:
        return beta
    return beta * min(1.0, epoch / warmup)


def plot_history(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["loss"], label="-ELBO")
    ax.plot(history["epoch"], history["recon"], label="reconstruction")
    ax.plot(history["epoch"], history["kl"], label="KL")
    ax.set_xlabel("epoch")
    ax.set_ylabel("nats / datapoint")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def eval_epoch(model: VAE, loader: DataLoader, device: torch.device, beta: float) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        metrics = elbo_loss(out["recon_logits"], x, out["mu"], out["logvar"], beta=beta)
        b = x.size(0)
        n += b
        for k in totals:
            totals[k] += metrics[k].item() * b
    return {k: v / n for k, v in totals.items()}


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = get_device()
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = VAE(latent_dim=args.latent_dim, arch=args.arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    vis_x, _ = next(iter(test_loader))
    vis_x = vis_x[:64].to(device)

    history: dict[str, list[float]] = {
        "epoch": [],
        "loss": [],
        "recon": [],
        "kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
        "beta": [],
    }

    print(f"device={device}  arch={args.arch}  latent_dim={args.latent_dim}  params={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = beta_at_epoch(epoch, args.beta, args.kl_warmup_epochs)
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        seen = 0
        bar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        for x, _ in bar:
            x = x.to(device)
            out = model(x)
            metrics = elbo_loss(out["recon_logits"], x, out["mu"], out["logvar"], beta=beta)
            opt.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            opt.step()

            b = x.size(0)
            seen += b
            for k in running:
                running[k] += metrics[k].item() * b
            bar.set_postfix(
                loss=f"{running['loss'] / seen:.2f}",
                recon=f"{running['recon'] / seen:.2f}",
                kl=f"{running['kl'] / seen:.2f}",
                beta=f"{beta:.2f}",
            )

        train_stats = {k: v / seen for k, v in running.items()}
        val_stats = eval_epoch(model, test_loader, device, beta=args.beta)

        history["epoch"].append(epoch)
        history["beta"].append(beta)
        for k in ("loss", "recon", "kl"):
            history[k].append(train_stats[k])
            history[f"val_{k}"].append(val_stats[k])

        print(
            f"epoch {epoch:03d}  "
            f"train -ELBO {train_stats['loss']:.2f}  recon {train_stats['recon']:.2f}  kl {train_stats['kl']:.2f}  |  "
            f"val -ELBO {val_stats['loss']:.2f}  recon {val_stats['recon']:.2f}  kl {val_stats['kl']:.2f}  "
            f"beta {beta:.2f}"
        )

        model.eval()
        recon = model.reconstruct(vis_x)
        comparison = torch.cat([vis_x, recon], dim=0)
        save_grid(comparison, out_dir / f"recon_epoch_{epoch:03d}.png", nrow=8)
        save_grid(model.sample(64, device=device), out_dir / f"samples_epoch_{epoch:03d}.png", nrow=8)

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "arch": args.arch,
            "latent_dim": args.latent_dim,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "history": history,
        }
        torch.save(ckpt, ckpt_dir / "vae_latest.pt")
        if epoch == args.epochs:
            torch.save(ckpt, ckpt_dir / "vae_final.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, out_dir / "loss_curve.png")
    final_ckpt = ckpt_dir / "vae_final.pt"
    print(f"saved checkpoint to {final_ckpt}")
    print(f"wrote figures under {out_dir}")
    return final_ckpt


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
