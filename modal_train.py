"""Train the VAE on a Modal GPU.

Checkpoints and figures are stored on a persistent Volume (`vae-mnist`) and
downloaded into local `checkpoints/` and `outputs/` when the run finishes.

    modal setup                          # once: create an account / token
    modal run modal_train.py             # default conv VAE, 20 epochs, T4
    modal run modal_train.py --epochs 5 --gpu L4
    modal run modal_train.py --download-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

APP_NAME = "vae-mnist"
VOLUME_NAME = "vae-mnist"
VOL_MOUNT = Path("/vol")
ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "torchvision", "matplotlib", "tqdm")
    .add_local_file(ROOT / "model.py", remote_path="/root/model.py")
    .add_local_file(ROOT / "utils.py", remote_path="/root/utils.py")
    .add_local_file(ROOT / "train.py", remote_path="/root/train.py")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

try:
    from modal.volume import FileEntryType
except ImportError:  # pragma: no cover
    FileEntryType = None


def _is_file(entry) -> bool:
    kind = getattr(entry, "type", None)
    if FileEntryType is not None and kind is not None:
        return kind == FileEntryType.FILE
    if kind is None:
        return not str(entry.path).endswith("/")
    return int(kind) == 1


def download_volume_dir(remote_dir: str, local_dir: Path) -> list[Path]:
    """Copy files under `remote_dir` on the Volume into `local_dir`."""
    remote_dir = "/" + remote_dir.strip("/")
    local_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        entries = volume.listdir(remote_dir, recursive=True)
    except Exception as exc:
        print(f"nothing at {remote_dir}: {exc}")
        return written

    prefix = remote_dir.strip("/")
    for entry in entries:
        if not _is_file(entry):
            continue
        remote_path = str(entry.path)
        rel = remote_path.replace("\\", "/").lstrip("/")
        if rel.startswith(prefix + "/"):
            rel = rel[len(prefix) + 1 :]
        elif rel == prefix:
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = volume.read_file(remote_path)
        if not isinstance(payload, (bytes, bytearray)):
            payload = b"".join(payload)
        dest.write_bytes(payload)
        written.append(dest)
        print(f"  {dest}  ({len(payload):,} bytes)")
    return written


@app.function(
    gpu="T4",
    timeout=2 * 60 * 60,
    volumes={str(VOL_MOUNT): volume},
)
def train_remote(
    arch: str = "conv",
    latent_dim: int = 20,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    beta: float = 1.0,
    kl_warmup_epochs: int = 0,
    seed: int = 0,
    num_workers: int = 4,
) -> dict[str, str]:
    import torch

    from train import train

    print(
        f"cuda={torch.cuda.is_available()}  "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
    )

    args = argparse.Namespace(
        arch=arch,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta=beta,
        kl_warmup_epochs=kl_warmup_epochs,
        seed=seed,
        num_workers=num_workers,
        data_dir=VOL_MOUNT / "data",
        ckpt_dir=VOL_MOUNT / "checkpoints",
        out_dir=VOL_MOUNT / "outputs",
    )
    ckpt = train(args)
    volume.commit()
    return {
        "checkpoint": str(ckpt),
        "outputs": str(args.out_dir),
    }


@app.local_entrypoint()
def main(
    arch: str = "conv",
    latent_dim: int = 20,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    beta: float = 1.0,
    kl_warmup_epochs: int = 0,
    seed: int = 0,
    gpu: str = "T4",
    download: bool = True,
    download_only: bool = False,
) -> None:
    if not download_only:
        print(f"training on Modal gpu={gpu}  arch={arch}  latent_dim={latent_dim}  epochs={epochs}")
        # with_options replaces volumes rather than merging them, so re-attach the Volume.
        result = train_remote.with_options(
            gpu=gpu,
            timeout=2 * 60 * 60,
            volumes={str(VOL_MOUNT): volume},
        ).remote(
            arch=arch,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            beta=beta,
            kl_warmup_epochs=kl_warmup_epochs,
            seed=seed,
        )
        print(f"remote wrote {result['checkpoint']}")

    if download or download_only:
        print(f"downloading from Volume {VOLUME_NAME!r} ...")
        n_ckpt = download_volume_dir("checkpoints", ROOT / "checkpoints")
        n_out = download_volume_dir("outputs", ROOT / "outputs")
        print(f"downloaded {len(n_ckpt)} checkpoint file(s), {len(n_out)} output file(s)")
        print("generate locally with:  python generate.py --checkpoint checkpoints/vae_final.pt")
