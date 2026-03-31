"""
Evaluate the quality of a trained VAE and/or LDM on MNIST.

Produces:
  1. VAE evaluation
     - Reconstruction loss (L1, L2, SSIM) on the test set
     - Visual grid: originals vs reconstructions
     - Latent space statistics (mean, std per channel)
     - Latent space t-SNE visualisation coloured by digit class

  2. LDM evaluation
     - Class-conditional sample grids (one row per digit)
     - Unconditional vs class-conditional comparison
     - FID score (if pytorch-fid is installed)
     - Classifier accuracy on generated samples (if classifier checkpoint exists)

Usage (from the repository root):
    # Evaluate VAE only
    python evaluate_models.py --vae_ckpt ./checkpoints/mnist_vae/mnist_vae_best.ckpt

    # Evaluate LDM only (loads VAE automatically from ldm_mnist.yaml)
    python evaluate_models.py --ldm_ckpt ./checkpoints/mnist_ldm/mnist_ldm_best.ckpt

    # Evaluate both
    python evaluate_models.py \
        --vae_ckpt ./checkpoints/mnist_vae/mnist_vae_best.ckpt \
        --ldm_ckpt ./checkpoints/mnist_ldm/mnist_ldm_best.ckpt

    # Also score generated samples with your trained classifier
    python evaluate_models.py \
        --ldm_ckpt ./checkpoints/mnist_ldm/mnist_ldm_best.ckpt \
        --clf_ckpt ./checkpoints/classifier/classifier_mnist.pth
"""

import argparse
import os
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Repo imports ──────────────────────────────────────────────────────────────
from ldm.util import instantiate_from_config

# Optional heavy imports — guarded so the script still runs without them
try:
    from sklearn.manifold import TSNE

    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths

    HAS_FID = True
except ImportError:
    HAS_FID = False


# ── Dataset ───────────────────────────────────────────────────────────────────

class MNISTEval(Dataset):
    """MNIST test set returning (image_01, image_11, label).

    image_01 : (3, H, W) in [0, 1]  — for metric computation
    image_11 : (3, H, W) in [-1, 1] — for model input (VAE / LDM expect this)
    label    : int
    """

    def __init__(self, root="./data", image_size=32, max_samples=None):
        self.data = datasets.MNIST(root=root, train=False, download=True)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.data))
        return len(self.data)

    def __getitem__(self, idx):
        pil, label = self.data[idx]
        img = self.transform(pil).repeat(3, 1, 1)  # (3, H, W) in [0, 1]
        return img, img * 2.0 - 1.0, label


# ── Metrics ───────────────────────────────────────────────────────────────────

def ssim_single(img1, img2, window_size=7):
    """Compute SSIM between two (C, H, W) tensors in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = window_size
    pad = k // 2

    # Average pooling as a simple uniform window
    mu1 = F.avg_pool2d(img1.unsqueeze(0), k, stride=1, padding=pad)
    mu2 = F.avg_pool2d(img2.unsqueeze(0), k, stride=1, padding=pad)

    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1.unsqueeze(0) ** 2, k, stride=1, padding=pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2.unsqueeze(0) ** 2, k, stride=1, padding=pad) - mu2_sq
    sigma12 = F.avg_pool2d((img1 * img2).unsqueeze(0), k, stride=1, padding=pad) - mu12

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den).mean().item()


# ── VAE evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_vae(vae, dataloader, device, output_dir):
    """Compute reconstruction metrics and produce visual diagnostics."""
    vae.eval()
    os.makedirs(output_dir, exist_ok=True)

    l1_vals, l2_vals, ssim_vals = [], [], []
    all_latents, all_labels = [], []
    originals, recons = [], []

    for img_01, img_11, labels in dataloader:
        img_11 = img_11.to(device)

        posterior = vae.encode(img_11)
        z = posterior.mode()                    # deterministic (mean)
        rec_11 = vae.decode(z)                  # [-1, 1]
        rec_01 = (rec_11.clamp(-1, 1) + 1) / 2  # [0, 1]
        img_01 = img_01.to(device)

        # Per-sample metrics
        for i in range(img_01.size(0)):
            l1_vals.append(F.l1_loss(rec_01[i], img_01[i]).item())
            l2_vals.append(F.mse_loss(rec_01[i], img_01[i]).sqrt().item())
            ssim_vals.append(ssim_single(img_01[i], rec_01[i]))

        # Collect latents for t-SNE
        all_latents.append(z.flatten(1).cpu())
        all_labels.append(labels)

        # Collect first batch for visualisation
        if len(originals) == 0:
            originals.append(img_01[:16].cpu())
            recons.append(rec_01[:16].cpu())

    # ── Numeric results ───────────────────────────────────────────────────
    metrics = {
        "L1_mean": np.mean(l1_vals),
        "L1_std": np.std(l1_vals),
        "L2_mean": np.mean(l2_vals),
        "L2_std": np.std(l2_vals),
        "SSIM_mean": np.mean(ssim_vals),
        "SSIM_std": np.std(ssim_vals),
    }

    print("\n" + "=" * 60)
    print("  VAE Reconstruction Quality  (test set)")
    print("=" * 60)
    print(f"  L1  loss   : {metrics['L1_mean']:.5f} ± {metrics['L1_std']:.5f}")
    print(f"  L2  loss   : {metrics['L2_mean']:.5f} ± {metrics['L2_std']:.5f}")
    print(f"  SSIM       : {metrics['SSIM_mean']:.4f} ± {metrics['SSIM_std']:.4f}")

    # ── Latent statistics ─────────────────────────────────────────────────
    all_z = torch.cat(all_latents, dim=0)
    z_channels = all_z.shape[1]  # flattened, but we know shape from config
    print(f"\n  Latent dim : {all_z.shape[1]}  (flattened)")
    print(f"  Latent mean: {all_z.mean().item():.4f}")
    print(f"  Latent std : {all_z.std().item():.4f}")
    print(f"  Latent min : {all_z.min().item():.4f}")
    print(f"  Latent max : {all_z.max().item():.4f}")

    # ── Reconstruction grid ───────────────────────────────────────────────
    orig = originals[0]
    rec = recons[0]
    # Interleave: orig0, rec0, orig1, rec1, ...
    pairs = torch.stack([orig, rec], dim=1).reshape(-1, 3, orig.shape[2], orig.shape[3])
    grid = make_grid(pairs, nrow=8, padding=2, pad_value=0.5)
    save_image(grid, os.path.join(output_dir, "vae_reconstructions.png"))
    print(f"\n  Saved reconstruction grid → {output_dir}/vae_reconstructions.png")

    # ── t-SNE of latent space ─────────────────────────────────────────────
    if HAS_TSNE:
        print("  Computing t-SNE (this may take a minute) ...")
        all_labels_cat = torch.cat(all_labels, dim=0).numpy()
        # Subsample if too many
        max_tsne = 3000
        if all_z.shape[0] > max_tsne:
            idx = np.random.choice(all_z.shape[0], max_tsne, replace=False)
            z_sub = all_z[idx].numpy()
            lab_sub = all_labels_cat[idx]
        else:
            z_sub = all_z.numpy()
            lab_sub = all_labels_cat

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        emb = tsne.fit_transform(z_sub)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=lab_sub, cmap="tab10",
                             s=8, alpha=0.7)
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
        cbar.set_label("Digit class")
        ax.set_title("VAE Latent Space (t-SNE)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "vae_tsne.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved t-SNE plot → {output_dir}/vae_tsne.png")
    else:
        print("  Skipping t-SNE (install scikit-learn for this)")

    # ── Per-channel latent histograms ─────────────────────────────────────
    # Reshape flattened latents back to (N, C, H, W) for per-channel stats
    # We know z_channels = 4 and spatial = 8x8 from the config
    n_ch = 4
    spatial = int(math.sqrt(all_z.shape[1] // n_ch))
    if n_ch * spatial * spatial == all_z.shape[1]:
        z_4d = all_z.reshape(-1, n_ch, spatial, spatial)
        fig, axes = plt.subplots(1, n_ch, figsize=(4 * n_ch, 3))
        for ch in range(n_ch):
            vals = z_4d[:, ch].flatten().numpy()
            axes[ch].hist(vals, bins=80, density=True, alpha=0.7, color=f"C{ch}")
            axes[ch].set_title(f"Channel {ch}\nμ={vals.mean():.2f} σ={vals.std():.2f}")
            axes[ch].set_xlabel("z value")
        fig.suptitle("VAE Latent Channel Distributions", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "vae_latent_histograms.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved latent histograms → {output_dir}/vae_latent_histograms.png")

    return metrics


# ── LDM evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ldm(ldm_model, device, output_dir, n_per_class=10,
                 ddim_steps=100, clf_ckpt=None, clf_args=None):
    """Generate class-conditional samples and evaluate them."""
    ldm_model.eval()
    os.makedirs(output_dir, exist_ok=True)

    n_classes = 10
    null_class = 10

    print("\n" + "=" * 60)
    print("  LDM Sample Quality Evaluation")
    print("=" * 60)

    # ── Class-conditional samples ─────────────────────────────────────────
    all_samples_01 = []
    all_labels = []

    for digit in range(n_classes):
        print(f"  Generating class {digit} ({n_per_class} samples) ...")

        # Build conditioning
        class_labels = torch.full((n_per_class,), digit, dtype=torch.long, device=device)
        batch_cond = {"class_label": class_labels}
        c = ldm_model.get_learned_conditioning(batch_cond)

        # Sample via DDIM
        shape = (n_per_class, ldm_model.channels,
                 ldm_model.image_size, ldm_model.image_size)

        from ldm.models.diffusion.ddim import DDIMSampler
        sampler = DDIMSampler(ldm_model)
        samples_z, _ = sampler.sample(
            S=ddim_steps,
            batch_size=n_per_class,
            shape=shape[1:],
            conditioning=c,
            verbose=False,
            eta=0.0,
        )

        # Decode to pixel space
        samples_x = ldm_model.decode_first_stage(samples_z)  # [-1, 1]
        samples_01 = (samples_x.clamp(-1, 1) + 1) / 2       # [0, 1]

        all_samples_01.append(samples_01.cpu())
        all_labels.extend([digit] * n_per_class)

    all_samples = torch.cat(all_samples_01, dim=0)  # (n_classes * n_per_class, 3, H, W)

    # ── Sample grid: one row per digit ────────────────────────────────────
    grid_samples = []
    for digit in range(n_classes):
        grid_samples.append(all_samples[digit * n_per_class: digit * n_per_class + min(n_per_class, 10)])

    grid = make_grid(torch.cat(grid_samples, dim=0), nrow=min(n_per_class, 10),
                     padding=2, pad_value=0.5)
    save_image(grid, os.path.join(output_dir, "ldm_class_conditional_samples.png"))
    print(f"\n  Saved class-conditional grid → {output_dir}/ldm_class_conditional_samples.png")

    # ── Unconditional samples for comparison ──────────────────────────────
    print("  Generating unconditional samples ...")
    uc_labels = torch.full((n_per_class,), null_class, dtype=torch.long, device=device)
    uc_batch = {"class_label": uc_labels}
    uc_c = ldm_model.get_learned_conditioning(uc_batch)

    from ldm.models.diffusion.ddim import DDIMSampler
    sampler = DDIMSampler(ldm_model)
    uc_z, _ = sampler.sample(
        S=ddim_steps,
        batch_size=n_per_class,
        shape=shape[1:],
        conditioning=uc_c,
        verbose=False,
        eta=0.0,
    )
    uc_x = ldm_model.decode_first_stage(uc_z)
    uc_01 = (uc_x.clamp(-1, 1) + 1) / 2

    grid_uc = make_grid(uc_01.cpu(), nrow=min(n_per_class, 10),
                        padding=2, pad_value=0.5)
    save_image(grid_uc, os.path.join(output_dir, "ldm_unconditional_samples.png"))
    print(f"  Saved unconditional grid → {output_dir}/ldm_unconditional_samples.png")

    # ── Per-sample quality stats ──────────────────────────────────────────
    pixel_means = all_samples.mean(dim=(1, 2, 3))
    pixel_stds = all_samples.std(dim=(1, 2, 3))
    print(f"\n  Generated pixel stats:")
    print(f"    Mean brightness : {pixel_means.mean():.4f} ± {pixel_means.std():.4f}")
    print(f"    Std  brightness : {pixel_stds.mean():.4f} ± {pixel_stds.std():.4f}")

    # ── Classifier accuracy on generated samples ──────────────────────────
    if clf_ckpt and clf_args:
        print("\n  Evaluating classifier accuracy on generated samples ...")
        from ldm.models.classifiers import CNNtorch

        clf = CNNtorch(**clf_args)
        state = torch.load(clf_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        clf.load_state_dict(state, strict=False)
        clf.to(device).eval()

        clf_input_size = 28
        in_ch = clf_args.get("input_channels", 1)

        correct = 0
        total = 0
        per_class_correct = [0] * n_classes
        per_class_total = [0] * n_classes

        for i in range(all_samples.shape[0]):
            img = all_samples[i: i + 1].to(device)
            # Resize and channel-reduce to match classifier input
            img_clf = F.interpolate(img, size=(clf_input_size, clf_input_size),
                                    mode="bilinear", align_corners=False)
            img_clf = img_clf[:, :in_ch]

            logits = clf(img_clf)
            pred = logits.argmax(dim=1).item()
            true_label = all_labels[i]

            per_class_total[true_label] += 1
            if pred == true_label:
                correct += 1
                per_class_correct[true_label] += 1
            total += 1

        overall_acc = correct / total * 100
        print(f"\n  Classifier accuracy on LDM samples: {overall_acc:.1f}%")
        print(f"  Per-class accuracy:")
        for d in range(n_classes):
            if per_class_total[d] > 0:
                acc = per_class_correct[d] / per_class_total[d] * 100
                print(f"    Digit {d}: {acc:.1f}%  ({per_class_correct[d]}/{per_class_total[d]})")

    # ── FID (if pytorch-fid is installed) ─────────────────────────────────
    if HAS_FID:
        print("\n  Computing FID ...")
        real_dir = os.path.join(output_dir, "_fid_real")
        gen_dir = os.path.join(output_dir, "_fid_gen")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)

        # Save generated images
        for i in range(all_samples.shape[0]):
            save_image(all_samples[i], os.path.join(gen_dir, f"{i:05d}.png"))

        # Save real test images
        test_ds = datasets.MNIST(root="./data", train=False, download=True)
        tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        n_real = min(len(test_ds), all_samples.shape[0])
        for i in range(n_real):
            pil, _ = test_ds[i]
            img = tf(pil).repeat(3, 1, 1)
            save_image(img, os.path.join(real_dir, f"{i:05d}.png"))

        try:
            fid_value = calculate_fid_given_paths(
                [real_dir, gen_dir], batch_size=64, device=device, dims=2048
            )
            print(f"  FID: {fid_value:.2f}")
        except Exception as e:
            print(f"  FID computation failed: {e}")

        # Clean up temp dirs
        import shutil
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(gen_dir, ignore_errors=True)
    else:
        print("\n  Skipping FID (install pytorch-fid: pip install pytorch-fid)")

    return all_samples


# ── Model loading helpers ─────────────────────────────────────────────────────

def load_vae(ckpt_path, cfg_path="mnist_ldce/autoencoder_mnist_kl.yaml", device="cpu"):
    config = OmegaConf.load(cfg_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"Loaded VAE from {ckpt_path}")
    return model


def load_ldm(ckpt_path, cfg_path="mnist_ldce/ldm_mnist.yaml", device="cpu"):
    config = OmegaConf.load(cfg_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  LDM missing keys: {len(missing)}")
    model.to(device).eval()
    print(f"Loaded LDM from {ckpt_path}")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate VAE and/or LDM quality on MNIST"
    )

    # Model checkpoints
    p.add_argument("--vae_ckpt", default=None,
                   help="Path to VAE checkpoint (.ckpt)")
    p.add_argument("--ldm_ckpt", default=None,
                   help="Path to LDM checkpoint (.ckpt)")
    p.add_argument("--clf_ckpt", default=None,
                   help="Path to classifier checkpoint (.pth) for scoring LDM samples")

    # Config files
    p.add_argument("--vae_cfg", default="mnist_ldce/configs/autoencoder_mnist_kl.yaml",
                   help="VAE architecture config YAML")
    p.add_argument("--ldm_cfg", default="mnist_ldce/configs/ldm_mnist.yaml",
                   help="LDM architecture config YAML")

    # Evaluation settings
    p.add_argument("--output_dir", default="mnist_ldce/results/eval",
                   help="Directory for output images and metrics")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Max test samples for VAE eval (None = full test set)")
    p.add_argument("--n_per_class", type=int, default=10,
                   help="Number of LDM samples to generate per digit class")
    p.add_argument("--ddim_steps", type=int, default=100,
                   help="DDIM sampling steps for LDM evaluation")
    p.add_argument("--device", default=None,
                   help="Device (auto-detected if omitted)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.vae_ckpt is None and args.ldm_ckpt is None:
        print("Error: provide at least one of --vae_ckpt or --ldm_ckpt")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── VAE evaluation ────────────────────────────────────────────────────
    if args.vae_ckpt:
        vae = load_vae(args.vae_ckpt, args.vae_cfg, device)

        dataset = MNISTEval(max_samples=args.max_samples)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

        vae_dir = os.path.join(args.output_dir, "vae")
        evaluate_vae(vae, loader, device, vae_dir)

        del vae
        torch.cuda.empty_cache() if device != "cpu" else None

    # ── LDM evaluation ────────────────────────────────────────────────────
    if args.ldm_ckpt:
        ldm_model = load_ldm(args.ldm_ckpt, args.ldm_cfg, device)

        # Classifier args from the default config
        clf_args = None
        if args.clf_ckpt:
            clf_args = dict(
                input_channels=1,
                in_conv_channels=[1, 8, 16],
                out_conv_channels=[8, 16, 32],
                conv_kernels=[5, 5, 3],
                softmax_flag=True,
                num_classes=10,
            )

        ldm_dir = os.path.join(args.output_dir, "ldm")
        evaluate_ldm(
            ldm_model, device, ldm_dir,
            n_per_class=args.n_per_class,
            ddim_steps=args.ddim_steps,
            clf_ckpt=args.clf_ckpt,
            clf_args=clf_args,
        )

    print("\n" + "=" * 60)
    print("  Evaluation complete")
    print(f"  Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()