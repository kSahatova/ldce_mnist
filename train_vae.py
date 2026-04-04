"""
Train a KL-regularised autoencoder (VAE) on MNIST.

The architecture is defined in mnist_ldce/autoencoder_mnist_kl.yaml.
The resulting checkpoint is used as the first stage in ldm_mnist.yaml.

Usage (from the repository root):
    python train_vae.py --output_dir assets/checkpoints/fmnist_vae/multi --epochs 50

Key design choices
──────────────────
  • Returns {'image': (H, W, C) float32 in [-1, 1]} per sample — the exact
    format AutoencoderKL.get_input() expects.
  • No class labels are needed; the VAE is unconditional.
  • AutoencoderKL uses two optimisers (encoder+decoder and discriminator);
    PyTorch Lightning handles the alternation via optimizer_idx automatically.
  • learning_rate is set explicitly on the model after instantiation so that
    configure_optimizers() can read self.learning_rate regardless of PL version.
"""

import argparse
import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset

from ldm.util import instantiate_from_config
from ldm.data.datasets import MNIST, FashionMNIST, DermaMNIST

_DATASET_MAP = {
    "MNIST": MNIST,
    "FashionMNIST": FashionMNIST,
    "DermaMNIST": DermaMNIST
}

# ── Dataset ───────────────────────────────────────────────────────────────────


class VAEDataset(Dataset):
    """Dataset wrapper that returns the image dict expected by AutoencoderKL.

    Delegates loading and class-filtering to the existing MNIST / FashionMNIST
    classes in data/datasets.py.  Those classes already handle resize and
    grayscale→pseudo-RGB conversion; this wrapper just normalises to [-1, 1]
    and permutes to (H, W, 3) as AutoencoderKL.get_input() expects.

    Returns:
        {'image': (H, W, 3) float32 tensor in [-1, 1]}
    """

    def __init__(
        self,
        split: str,
        dataset_name: str = "MNIST",
        image_size: int = 32,
        root: str = "./data",
        classes: list = None,
        **dataset_kwargs,
    ):
        cls = _DATASET_MAP.get(dataset_name)
        if cls is None:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Supported: {list(_DATASET_MAP)}"
            )
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
        image = self.data[idx][0]  # (3, H, W) in [0, 1]
        image = image * 2.0 - 1.0  # [0, 1] → [-1, 1]
        image = image.permute(1, 2, 0)  # (3, H, W) → (H, W, 3)
        return {"image": image.contiguous()}


# ── DataModule ────────────────────────────────────────────────────────────────


class VAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_size: int,
        root: str,
        batch_size: int,
        num_workers: int,
        dataset_name: str = "MNIST",
        filter_classes: list = None,
        **dataset_kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.filter_classes = filter_classes
        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage=None):
        kwargs = dict(
            dataset_name=self.dataset_name,
            image_size=self.image_size,
            root=self.root,
            classes=self.filter_classes,
            **self.dataset_kwargs,
        )
        self.train_ds = VAEDataset("train", **kwargs)
        self.val_ds = VAEDataset("test", **kwargs)

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
    p = argparse.ArgumentParser(description="Train a KL-VAE on MNIST")
    p.add_argument(
        "--cfg_path",
        default="assets/configs/autoencoderkl_mnist.yaml",
        help="Path to the model architecture YAML",
    )
    p.add_argument(
        "--output_dir",
        default="assets/checkpoints/mnist_vae",
        help="Directory where checkpoints are saved",
    )
    p.add_argument("--data_root", default="./data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every_n_epochs", type=int, default=5)
    p.add_argument(
        "--resume_ckpt",
        default=None,
        help="Path to a .ckpt file to resume training from",
    )
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--precision", choices=["32", "16", "bf16"], default="32")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model architecture from: {args.cfg_path}")
    config = OmegaConf.load(args.cfg_path)

    dataset_name = OmegaConf.select(config, "data.name", default="MNIST")
    filter_classes = OmegaConf.select(config, "data.classes", default=None)
    if filter_classes is not None:
        filter_classes = list(filter_classes)

    _reserved = {"name", "classes"}
    dataset_kwargs = {
        k: v for k, v in OmegaConf.to_container(config.data, resolve=True).items()
        if k not in _reserved
    } if "data" in config else {}

    model = instantiate_from_config(config.model)

    # Set learning_rate explicitly so configure_optimizers() can read it
    # regardless of PyTorch Lightning version.
    model.learning_rate = args.lr

    if args.resume_ckpt:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(
                f"  Missing keys    ({len(missing)}): {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            print(
                f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
                f"{'...' if len(unexpected) > 5 else ''}"
            )

    datamodule = VAEDataModule(
        image_size=args.image_size,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_name=dataset_name,
        filter_classes=filter_classes,
        **dataset_kwargs,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="fmnist_vae_{epoch:04d}",
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
        save_last=True,
        verbose=True,
    )
    best_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="mnist_vae_best",
        monitor="val/rec_loss",
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

    print(f"\nTraining KL-VAE on {dataset_name} for {args.epochs} epochs")
    print(f"  num_classes  = {len(config.data.classes)}")
    print(f"  batch_size  = {args.batch_size}")
    print(f"  image_size  = {args.image_size}")
    print(f"  lr          = {args.lr}")
    print(f"  output_dir  = {args.output_dir}\n")


    trainer.fit(model, datamodule=datamodule)

    print(f"\nTraining complete.")
    print(f"  Best checkpoint : {best_cb.best_model_path}")
    print(f"  Last checkpoint : {checkpoint_cb.last_model_path}")
    # print(
    #     f"\nUpdate ldm_mnist.yaml first_stage_config.params.ckpt_path to:\n"
    #     f"  {best_cb.best_model_path}\n"
    # )


if __name__ == "__main__":
    main()
