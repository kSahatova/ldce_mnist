"""
Train a class-conditional Latent Diffusion Model (LDM) on MNIST.

Requires a pre-trained VAE checkpoint produced by train_vae_mnist.py.
The model architecture is defined in mnist_ldce/ldm_mnist.yaml.

Usage (from the repository root):
    # Train (VAE must already exist at ./checkpoints/mnist_vae/mnist_vae_best.ckpt)
    python train_ldm_mnist.py --output_dir ./checkpoints/mnist_ldm --epochs 200

    # Resume from a checkpoint
    python train_ldm_mnist.py --resume_ckpt mnist_ldce/checkpoints/mnist_ldm/last.ckpt

Key design choices
──────────────────
  • Same DDPMDataset / MNISTDataModule as train_ddpm.py — returns dicts with
    'image' (B, H, W, C) in [-1, 1] and 'class_label' (B,).
  • The VAE first stage is loaded and frozen automatically by LatentDiffusion
    from the ckpt_path set in ldm_mnist.yaml.
  • scale_by_std=true in the yaml: scale_factor is computed on the first
    training batch from the std of the encoded latents.
  • CFG dropout: class labels are randomly replaced with NULL_CLASS=10 at
    rate cfg_dropout, enabling classifier-free guidance at inference.
"""

import argparse
import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ldm.util import instantiate_from_config
from ldm.data.datasets import MNIST, FashionMNIST, DermaMNIST

_DATASET_MAP = {"MNIST": MNIST, "FashionMNIST": FashionMNIST, "DermaMNIST": DermaMNIST}

# ── Constants ─────────────────────────────────────────────────────────────────

# Index reserved for the unconditional / null class in the ClassEmbedder.
# Must match n_classes - 1 in ldm_mnist.yaml (10 digits → index 10).
NULL_CLASS = 10


# ── Dataset ───────────────────────────────────────────────────────────────────


class DDPMDataset(Dataset):
    """Wraps torchvision MNIST for LatentDiffusion training.

    Returns dicts:
        'image'       – (H, W, C) float32 tensor in [-1, 1]   ← b h w c layout
        'class_label' – int64 scalar (or NULL_CLASS after CFG dropout)
    """

    def __init__(
        self,
        split: str,
        dataset_name: str = "MNIST",
        image_size: int = 32,
        root: str = "./data",
        classes: list = None,
        cfg_dropout: float = 0.1,
        **dataset_kwargs,
    ):
        self.cfg_dropout = cfg_dropout
        cls = _DATASET_MAP.get(dataset_name)
        if cls is None:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Supported: {list(_DATASET_MAP)}"
            )
        # DermaMNIST accepts `undersample`; MNIST/FashionMNIST do not.
        if cls is not DermaMNIST:
            dataset_kwargs.pop("undersample", None)
        self.data = cls(
            root=root,
            split=split,
            image_size=image_size,
            classes=classes,
            **dataset_kwargs,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        image, label, _ = self.data[idx]
        image = image * 2.0 - 1.0  # [0, 1] → [-1, 1]
        image = image.permute(1, 2, 0)  # (3, H, W) → (H, W, 3)

        if self.cfg_dropout > 0 and torch.rand(1).item() < self.cfg_dropout:
            label = NULL_CLASS

        return {
            "image": image.contiguous(),
            "class_label": torch.tensor(label, dtype=torch.long),
        }


# ── DataModule ────────────────────────────────────────────────────────────────


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_size: int,
        root: str,
        batch_size: int,
        num_workers: int,
        dataset_name: str,
        filter_classes: list,
        cfg_dropout: float,
        **dataset_kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.filter_classes = filter_classes
        self.cfg_dropout = cfg_dropout
        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage=None):
        kwargs = dict(
            dataset_name=self.dataset_name,
            image_size=self.image_size,
            root=self.root,
            classes=self.filter_classes,
            **self.dataset_kwargs,
        )
        self.train_ds = DDPMDataset("train", **kwargs)
        self.val_ds = DDPMDataset("test", **kwargs)
        
        print(f"Train: {len(self.train_ds)} | Val: {len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train a class-conditional LDM on MNIST")
    p.add_argument(
        "--cfg_path",
        default="assets/configs/ldm_mnist.yaml",
        help="Path to the model architecture YAML (default: mnist_ldce/ldm_mnist.yaml)",
    )
    p.add_argument(
        "--output_dir",
        default="assests/checkpoints/mnist_ldm",
        help="Directory where checkpoints are saved",
    )
    p.add_argument("--data_root", default="./data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (overrides the value in the yaml)",
    )
    p.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="Pixel resolution fed to the VAE encoder (default: 32)",
    )
    p.add_argument(
        "--cfg_dropout",
        type=float,
        default=0.1,
        help="Fraction of labels replaced with null class during "
        "training to enable classifier-free guidance (default: 0.1)",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every_n_epochs", type=int, default=5)
    p.add_argument(
        "--resume_ckpt",
        default=None,
        help="Path to a .ckpt file to resume training from",
    )
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--precision", choices=["32", "16", "bf16"], default="32")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model architecture from: {args.cfg_path}")
    config = OmegaConf.load(args.cfg_path)
    config.model.base_learning_rate = args.lr

    model = instantiate_from_config(config.model)

    datamodule = DataModule(
        dataset_name=config.data.name,
        image_size=args.image_size,
        root=args.data_root,
        filter_classes=config.data.classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cfg_dropout=args.cfg_dropout,
        undersample=config.data.get("undersample", False),
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="ldm_{epoch:04d}",
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
        save_last=True,
        verbose=True,
    )
    best_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="ldm_best",
        monitor="val/loss_simple_ema",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    accelerator = "gpu" if args.gpus > 0 else "cpu"
    devices = args.gpus if args.gpus > 0 else 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        callbacks=[checkpoint_cb, best_cb, lr_monitor],
        default_root_dir=args.output_dir,
        log_every_n_steps=50,
        val_check_interval=1.0,
    )

    print(f"\nTraining class-conditional LDM on MNIST for {args.epochs} epochs")
    print(f"  batch_size  = {args.batch_size}")
    print(
        f"  image_size  = {args.image_size}  (VAE latent: {args.image_size // 4}×{args.image_size // 4}×4)"
    )
    print(f"  cfg_dropout = {args.cfg_dropout}")
    print(f"  output_dir  = {args.output_dir}\n")

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_ckpt)

    print(f"\nTraining complete.")
    print(f"  Best checkpoint : {best_cb.best_model_path}")
    print(f"  Last checkpoint : {checkpoint_cb.last_model_path}")
    print(
        f"\nTo use with the LDCE pipeline, update config_ldm.yaml:\n"
        f"  diffusion_model:\n"
        f'    ckpt_path: "{best_cb.best_model_path}"\n'
    )


if __name__ == "__main__":
    main()
