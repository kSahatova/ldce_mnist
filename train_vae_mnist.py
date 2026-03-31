"""
Train a KL-regularised autoencoder (VAE) on MNIST.

The architecture is defined in mnist_ldce/autoencoder_mnist_kl.yaml.
The resulting checkpoint is used as the first stage in ldm_mnist.yaml.

Usage (from the repository root):
    python train_vae_mnist.py --output_dir ./checkpoints/mnist_vae --epochs 50

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
from torchvision import datasets, transforms

from ldm.util import instantiate_from_config


# ── Dataset ───────────────────────────────────────────────────────────────────

class VAEDataset(Dataset):
    """MNIST wrapper that returns the image dict expected by AutoencoderKL.

    Returns:
        {'image': (H, W, 3) float32 tensor in [-1, 1]}
    """

    def __init__(self, split: str, image_size: int = 32, root: str = './data'):
        self.data = datasets.MNIST(
            root=root, train=(split == 'train'), download=True
        )
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),          # (1, H, W) in [0, 1]
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        pil_image, _ = self.data[idx]
        image = self.transform(pil_image)   # (1, H, W) in [0, 1]
        image = image.repeat(3, 1, 1)       # (3, H, W) grayscale → pseudo-RGB
        image = image * 2.0 - 1.0          # [0, 1] → [-1, 1]
        image = image.permute(1, 2, 0)     # (3, H, W) → (H, W, 3)
        return {'image': image.contiguous()}


# ── DataModule ────────────────────────────────────────────────────────────────

class VAEDataModule(pl.LightningDataModule):
    def __init__(self, image_size: int, root: str,
                 batch_size: int, num_workers: int):
        super().__init__()
        self.image_size  = image_size
        self.root        = root
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = VAEDataset('train', self.image_size, self.root)
        self.val_ds   = VAEDataset('test',  self.image_size, self.root)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train a KL-VAE on MNIST')
    p.add_argument('--cfg_path',    default='mnist_ldce/configs/autoencoderkl_mnist.yaml',
                   help='Path to the model architecture YAML')
    p.add_argument('--output_dir',  default='./checkpoints/mnist_vae',
                   help='Directory where checkpoints are saved')
    p.add_argument('--data_root',   default='./data')
    p.add_argument('--epochs',      type=int,   default=30)
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--image_size',  type=int,   default=32)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--save_every_n_epochs', type=int, default=5)
    p.add_argument('--resume_ckpt', default=None,
                   help='Path to a .ckpt file to resume training from')
    p.add_argument('--gpus',        type=int,   default=0)
    p.add_argument('--precision',   choices=['32', '16', 'bf16'], default='32')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model architecture from: {args.cfg_path}")
    config = OmegaConf.load(args.cfg_path)

    model = instantiate_from_config(config.model)

    # Set learning_rate explicitly so configure_optimizers() can read it
    # regardless of PyTorch Lightning version.
    model.learning_rate = args.lr

    if args.resume_ckpt:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        if missing:
            print(f"  Missing keys    ({len(missing)}): {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
                  f"{'...' if len(unexpected) > 5 else ''}")

    datamodule = VAEDataModule(
        image_size=args.image_size,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='mnist_vae_{epoch:04d}',
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
        save_last=True,
        verbose=True,
    )
    best_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='mnist_vae_best',
        monitor='val/rec_loss',
        mode='min',
        save_top_k=1,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    accelerator = 'gpu' if args.gpus > 0 else 'cpu'
    devices     = args.gpus if args.gpus > 0 else 1

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

    print(f"\nTraining KL-VAE on MNIST for {args.epochs} epochs")
    print(f"  batch_size  = {args.batch_size}")
    print(f"  image_size  = {args.image_size}")
    print(f"  lr          = {args.lr}")
    print(f"  output_dir  = {args.output_dir}\n")

    trainer.fit(model, datamodule=datamodule)

    print(f"\nTraining complete.")
    print(f"  Best checkpoint : {best_cb.best_model_path}")
    print(f"  Last checkpoint : {checkpoint_cb.last_model_path}")
    print(
        f"\nUpdate ldm_mnist.yaml first_stage_config.params.ckpt_path to:\n"
        f"  {best_cb.best_model_path}\n"
    )


if __name__ == '__main__':
    main()
