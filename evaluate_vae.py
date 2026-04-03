"""
Evaluate a trained VAE checkpoint and visualise reconstructions.

Reads dataset name and classes from the same config used for training, so
it works identically for MNIST, FashionMNIST and DermaMNIST.

Usage:
    python evaluate_vae.py \
        --cfg_path  assets/configs/autoencoderkl_derma.yaml \
        --vae_ckpt  assets/checkpoints/derma_vae/last.ckpt \
        --output_dir assets/results/derma_vae \
        --n_vis 8
"""

import argparse
import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ldm.util import instantiate_from_config
from ldm.data.datasets import MNIST, FashionMNIST, DermaMNIST

_DATASET_MAP = {
    "MNIST": MNIST,
    "FashionMNIST": FashionMNIST,
    "DermaMNIST": DermaMNIST,
}

# class-name lookups for labelled datasets
_MNIST_NAMES = {i: str(i) for i in range(10)}
_FMNIST_NAMES = {
    0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal",  6: "Shirt",   7: "Sneaker", 8: "Bag",   9: "Boot",
}


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    """Wraps any dataset from _DATASET_MAP and returns
    (img_01, img_11, label) where
      img_01: (3, H, W) float32 in [0, 1]   — for metrics
      img_11: (3, H, W) float32 in [-1, 1]  — for VAE input
    """

    def __init__(self, dataset_name, split, cfg, data_root):
        cls = _DATASET_MAP[dataset_name]
        # collect any dataset-specific kwargs from the config
        reserved = {"name", "classes"}
        extra = {
            k: v for k, v in OmegaConf.to_container(cfg.data, resolve=True).items()
            if k not in reserved
        } if "data" in cfg else {}
        classes = OmegaConf.select(cfg, "data.classes", default=None)
        if classes is not None:
            classes = list(classes)
        self.inner = cls(
            root=data_root,
            split=split,
            image_size=cfg.get("image_size", 32),
            classes=classes,
            **extra,
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        img_01, label, *_ = self.inner[idx]   # (3, H, W) in [0, 1]
        return img_01, img_01 * 2.0 - 1.0, label


# ── Metrics ───────────────────────────────────────────────────────────────────

def ssim_single(a, b, window_size=7):
    """SSIM between two (C, H, W) tensors in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k, pad = window_size, window_size // 2
    mu1 = F.avg_pool2d(a.unsqueeze(0), k, stride=1, padding=pad)
    mu2 = F.avg_pool2d(b.unsqueeze(0), k, stride=1, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.avg_pool2d(a.unsqueeze(0) ** 2, k, stride=1, padding=pad) - mu1_sq
    s2 = F.avg_pool2d(b.unsqueeze(0) ** 2, k, stride=1, padding=pad) - mu2_sq
    s12 = F.avg_pool2d((a * b).unsqueeze(0), k, stride=1, padding=pad) - mu12
    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return (num / den).mean().item()


# ── Visualisation ─────────────────────────────────────────────────────────────

def _to_hwc(t):
    """(3, H, W) tensor → (H, W, 3) numpy array in [0, 1]."""
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def save_reconstruction_figure(originals, reconstructions, labels,
                                class_names, path, n_cols=8):
    """
    Save a matplotlib figure with alternating orig / recon columns.

    originals / reconstructions : list of (3, H, W) tensors in [0, 1]
    labels                      : list of int
    class_names                 : dict {int -> str}
    n_cols                      : number of image pairs per row
    """
    n = len(originals)
    n_rows = math.ceil(n / n_cols)

    # Each pair occupies 2 columns; a thin spacer column separates pairs
    # Layout per row: [orig | rec | gap | orig | rec | gap | ...]
    fig, axes = plt.subplots(
        n_rows, n_cols * 2,
        figsize=(n_cols * 2 * 1.4, n_rows * 1.6),
        gridspec_kw={"wspace": 0.05, "hspace": 0.4},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ax in axes.flat:
        ax.axis("off")

    for i, (orig, rec, label) in enumerate(zip(originals, reconstructions, labels)):
        row, col_pair = divmod(i, n_cols)
        col_orig = col_pair * 2
        col_rec  = col_pair * 2 + 1

        axes[row, col_orig].imshow(_to_hwc(orig))
        axes[row, col_rec ].imshow(_to_hwc(rec))

        # label only on the original column
        name = class_names.get(int(label), str(int(label)))
        axes[row, col_orig].set_title(name, fontsize=6, pad=2)

    # Column headers on first row
    for col_pair in range(n_cols):
        axes[0, col_pair * 2    ].set_xlabel("orig", fontsize=5)
        axes[0, col_pair * 2 + 1].set_xlabel("recon", fontsize=5)

    fig.suptitle("VAE Reconstructions  (orig | recon)", fontsize=11, y=1.01)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Core evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_vae(vae, loader, device, output_dir, n_vis, class_names):
    vae.eval()
    os.makedirs(output_dir, exist_ok=True)

    l1_vals, l2_vals, ssim_vals = [], [], []
    all_latents, all_labels = [], []
    vis_orig, vis_rec, vis_labels = [], [], []

    for img_01, img_11, labels in loader:
        img_11 = img_11.to(device)
        z = vae.encode(img_11).mode()
        rec_11 = vae.decode(z)
        rec_01 = (rec_11.clamp(-1, 1) + 1) / 2
        img_01 = img_01.to(device)

        for i in range(img_01.size(0)):
            l1_vals.append(F.l1_loss(rec_01[i], img_01[i]).item())
            l2_vals.append(F.mse_loss(rec_01[i], img_01[i]).sqrt().item())
            ssim_vals.append(ssim_single(img_01[i], rec_01[i]))

        all_latents.append(z.flatten(1).cpu())
        all_labels.append(labels)

        # Collect for visualisation (class-balanced: one sample per class seen)
        if len(vis_orig) < n_vis:
            for i in range(img_01.size(0)):
                if len(vis_orig) >= n_vis:
                    break
                vis_orig.append(img_01[i].cpu())
                vis_rec.append(rec_01[i].cpu())
                vis_labels.append(int(labels[i]))

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  VAE Reconstruction Quality  (test set)")
    print("=" * 55)
    print(f"  L1   : {np.mean(l1_vals):.5f} ± {np.std(l1_vals):.5f}")
    print(f"  RMSE : {np.mean(l2_vals):.5f} ± {np.std(l2_vals):.5f}")
    print(f"  SSIM : {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}")

    all_z = torch.cat(all_latents, dim=0)
    print(f"\n  Latent  dim  : {all_z.shape[1]} (flattened)")
    print(f"  Latent  mean : {all_z.mean():.4f}")
    print(f"  Latent  std  : {all_z.std():.4f}")
    print(f"  Latent  range: [{all_z.min():.3f}, {all_z.max():.3f}]")

    # ── Reconstruction figure ──────────────────────────────────────────────
    fig_path = os.path.join(output_dir, "vae_reconstructions.png")
    save_reconstruction_figure(
        vis_orig, vis_rec, vis_labels, class_names,
        path=fig_path, n_cols=min(n_vis, 8),
    )
    print(f"\n  Saved reconstruction figure → {fig_path}")

    # ── Per-channel latent histograms ──────────────────────────────────────
    n_ch = 4
    spatial = int(math.sqrt(all_z.shape[1] // n_ch))
    if n_ch * spatial * spatial == all_z.shape[1]:
        z_4d = all_z.reshape(-1, n_ch, spatial, spatial)
        fig, axes = plt.subplots(1, n_ch, figsize=(4 * n_ch, 3))
        for ch in range(n_ch):
            vals = z_4d[:, ch].flatten().numpy()
            axes[ch].hist(vals, bins=80, density=True, alpha=0.75, color=f"C{ch}")
            axes[ch].set_title(f"ch {ch}  μ={vals.mean():.2f} σ={vals.std():.2f}",
                               fontsize=9)
            axes[ch].set_xlabel("z")
        fig.suptitle("VAE Latent Channel Distributions", fontsize=12)
        fig.tight_layout()
        hist_path = os.path.join(output_dir, "vae_latent_histograms.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"  Saved latent histograms     → {hist_path}")

    return {"L1": np.mean(l1_vals), "RMSE": np.mean(l2_vals), "SSIM": np.mean(ssim_vals)}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained VAE checkpoint")
    p.add_argument("--cfg_path",   required=True,
                   help="Training config YAML (same one used with train_vae.py)")
    p.add_argument("--vae_ckpt",   required=True,
                   help="Path to VAE .ckpt file")
    p.add_argument("--data_root",  default="./data")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--output_dir", default="assets/results/vae_eval")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--n_vis",      type=int, default=16,
                   help="Number of orig/recon pairs to include in the figure")
    p.add_argument("--split",      default="test",
                   help="Dataset split to evaluate on")
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    cfg = OmegaConf.load(args.cfg_path)
    # expose image_size so EvalDataset can read it
    cfg_eval = OmegaConf.merge(cfg, OmegaConf.create({"image_size": args.image_size}))

    dataset_name = OmegaConf.select(cfg, "data.name", default="MNIST")

    # Build class-name lookup from config classes list (if present)
    cfg_classes = OmegaConf.select(cfg, "data.classes", default=None)
    if dataset_name == "DermaMNIST" and cfg_classes is not None:
        class_names = {i: name for i, name in enumerate(cfg_classes)}
    elif dataset_name == "FashionMNIST":
        class_names = _FMNIST_NAMES
    else:
        class_names = _MNIST_NAMES

    # Load model
    model = instantiate_from_config(cfg.model)
    sd = torch.load(args.vae_ckpt, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"Loaded VAE from {args.vae_ckpt}")

    # Build dataloader
    ds = EvalDataset(dataset_name, args.split, cfg_eval, args.data_root)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(args.num_workers > 0),
    )
    print(f"Dataset: {dataset_name}  split={args.split}  n={len(ds)}")

    with torch.no_grad():
        metrics = evaluate_vae(model, loader, device, args.output_dir,
                               args.n_vis, class_names)

    print(f"\n  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
