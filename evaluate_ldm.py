"""
evaluate_ldm.py  –  config-driven LDM and VAE evaluation.

Replaces the hardcoded MNISTEval dataset and CNNtorch classifier in
evaluate_ldm_vae_models.py with the dataset and classifier specified
in the LDCE config file.

Usage (from repository root):
    # Evaluate LDM sample quality only
    python evaluate_ldm.py --config assets/configs/config_ldce_mnist.yaml


    # Override device and output directory
    python evaluate_ldm.py --config assets/configs/config_ldce_mnist.yaml \
     --device cuda:1 --output_dir assets/results/eval
"""

import argparse
import contextlib
import importlib.abc
import importlib.machinery
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ldm.util import instantiate_from_config
import ldm.models.classifiers as _clf_module
from ldm.data.datasets import MNIST, FashionMNIST, DermaMNIST
from ldm.models.diffusion.ddim import DDIMSampler
from utils.preprocessor import Normalizer

_DATASET_MAP = {"MNIST": MNIST, 
                "FashionMNIST": FashionMNIST, 
                "DermaMNIST": DermaMNIST}

UNCOND_CLASS_IDX: int | None = None  # derived from config at runtime: n_classes - 1

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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EvalDatasetAdapter(Dataset):
    """Wraps any dataset returning (image, label, idx) into (img_01, img_11, label).

    img_01 : (C, H, W) in [0, 1]   — for metric computation
    img_11 : (C, H, W) in [-1, 1]  — for model input (VAE / LDM expect this range)
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img_01, label, _ = self.dataset[idx]
        return img_01, img_01 * 2.0 - 1.0, label


def build_eval_dataset(cfg: dict) -> Dataset:
    """Build the evaluation dataset from config, wrapped as EvalDatasetAdapter."""
    data_cfg = cfg["data"]
    name = data_cfg.get("name", "MNIST")
    if name not in _DATASET_MAP:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(_DATASET_MAP)}")
    cls = _DATASET_MAP[name]

    if name in ("MNIST", "FashionMNIST"):
        base = cls(
            root=data_cfg["root"],
            split=data_cfg.get("split", "test"),
            image_size=data_cfg["image_size"],
            max_samples=data_cfg.get("max_samples"),
            classes=data_cfg.get("filter_classes"),
        )
    elif name == "DermaMNIST":
        base = cls(
            root=data_cfg["root"],
            image_size=data_cfg["image_size"],
            split=data_cfg.get("split", "test"),
            download=data_cfg.get("download", True),
            undersample=data_cfg.get("undersample", True),
            channels_first=data_cfg.get("channels_first", True),
            classes=data_cfg.get("filter_classes", ["all"]),
        )
    else:
        raise ValueError(f"Unhandled dataset name: '{name}'")

    return EvalDatasetAdapter(base)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier helpers  (mirrors ldce_runner_new.py)
# ─────────────────────────────────────────────────────────────────────────────

class EvalOnlyModule(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.eval()

    def train(self, mode: bool = True) -> "EvalOnlyModule":  # noqa: ARG002
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResizeWrapper(nn.Module):
    def __init__(self, model: nn.Module, clf_size: int, out_channels: int = 1) -> None:
        super().__init__()
        self.model = model
        self.clf_size = clf_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=(self.clf_size, self.clf_size), mode="bilinear", align_corners=False
        )
        return self.model(x[:, : self.out_channels])


@contextlib.contextmanager
def _src_stub_context():
    class _AnyObj:
        def __setstate__(self, state: dict) -> None:
            self.__dict__.update(state if isinstance(state, dict) else {})

    class _SrcStub(types.ModuleType):
        def __getattr__(self, _name: str):
            return _AnyObj

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname == "src" or fullname.startswith("src."):
                return importlib.machinery.ModuleSpec(fullname, self)

        def create_module(self, spec):
            return _SrcStub(spec.name)

        def exec_module(self, module):
            sys.modules[module.__name__] = module

    finder = _Finder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def load_classifier(args: dict, ckpt_path: str, device: torch.device) -> nn.Module:
    args = dict(args)
    cls_name = args.pop("model_class", "CNNtorch")
    cls = getattr(_clf_module, cls_name)
    lightning_used = args.pop("lightning_used", False)
    args.pop("input_channels", None)  # consumed by ResizeWrapper, not the model
    model = cls(**args)
    with _src_stub_context():
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
        if lightning_used and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def build_classifier(cfg: dict, device: torch.device) -> nn.Module:
    args = cfg["classifier_model"]["args"]
    clf_size = cfg["classifier_model"].get("input_size", cfg["data"]["image_size"])
    pipeline_size = cfg["data"]["image_size"]
    in_ch = args.get("input_channels", 1)

    model = load_classifier(args, cfg["classifier_model"]["ckpt_path"], device)

    if clf_size != pipeline_size or in_ch != 3:
        print(f"  Classifier resize: {pipeline_size}→{clf_size}, channels: 3→{in_ch}")
        model = ResizeWrapper(model, clf_size, out_channels=in_ch)

    if cfg["classifier_model"].get("mnist_normalisation", False):
        model = Normalizer(model, [0.1307] * in_ch, [0.3081] * in_ch)

    return EvalOnlyModule(model).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def ssim_single(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 7) -> float:
    """Compute SSIM between two (C, H, W) tensors in [0, 1]."""
    C1, C2 = 0.01**2, 0.03**2
    k, pad = window_size, window_size // 2

    mu1 = F.avg_pool2d(img1.unsqueeze(0), k, stride=1, padding=pad)
    mu2 = F.avg_pool2d(img2.unsqueeze(0), k, stride=1, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1.unsqueeze(0) ** 2, k, stride=1, padding=pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2.unsqueeze(0) ** 2, k, stride=1, padding=pad) - mu2_sq
    sigma12 = F.avg_pool2d((img1 * img2).unsqueeze(0), k, stride=1, padding=pad) - mu12

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den).mean().item()

# ─────────────────────────────────────────────────────────────────────────────
# LDM evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ldm(
    ldm_model: nn.Module,
    classifier: nn.Module,
    cfg: dict,
    device: torch.device,
    output_dir: str,
    uncond_class_idx: int = None,
) -> None:
    """Generate class-conditional samples and score them with the classifier."""
    ldm_model.eval()
    os.makedirs(output_dir, exist_ok=True)

    if uncond_class_idx is None:
        uncond_class_idx = ldm_model.cond_stage_model.embedding.num_embeddings - 1

    n_per_class = cfg.get("eval_n_per_class", 10)
    ddim_steps = cfg.get("eval_ddim_steps", cfg.get("ddim_steps", 100))
    ddim_eta = cfg.get("ddim_eta", 0.0)

    # ── LDM conditioning indices ──────────────────────────────────────────────
    # eval_classes must use the same label indices the LDM was trained on.
    # These come from the LDM's own config (data.classes), NOT from the LDCE
    # classifier class_map, which maps classifier output indices → dataset labels
    # and may differ from what the LDM embedding expects.
    ldm_cfg = OmegaConf.load(cfg["diffusion_model"]["cfg_path"])
    ldm_data_classes = OmegaConf.to_container(ldm_cfg.get("data", {}).get("classes", []),
                                               resolve=True)

    n_emb = ldm_model.cond_stage_model.embedding.num_embeddings  # includes null slot
    print(f"  LDM embedding size : {n_emb}  (null class idx = {uncond_class_idx})")

    if ldm_data_classes:
        # DermaMNIST uses string class names remapped to 0-based indices during training.
        # MNIST / FashionMNIST use the original integer labels directly.
        if isinstance(ldm_data_classes[0], str):
            # String labels → contiguous 0-based indices (as DermaMNIST does internally)
            eval_classes = list(range(len(ldm_data_classes)))
            ldm_idx_to_dataset_label = {i: name for i, name in enumerate(ldm_data_classes)}
        else:
            eval_classes = sorted(int(c) for c in ldm_data_classes)
            ldm_idx_to_dataset_label = {c: c for c in eval_classes}
    else:
        # Fallback: all valid non-null indices
        eval_classes = list(range(uncond_class_idx))
        ldm_idx_to_dataset_label = {i: i for i in eval_classes}

    # Safety check: catch any index that would crash the embedding lookup
    out_of_range = [c for c in eval_classes if c >= n_emb]
    if out_of_range:
        raise ValueError(
            f"eval_classes {out_of_range} exceed LDM embedding size {n_emb}. "
            f"The LDM config data.classes={ldm_data_classes} does not match the "
            f"trained checkpoint (num_embeddings={n_emb}). "
            f"Check that the correct LDM checkpoint is paired with the correct config."
        )

    # ── class_map: used only for scoring generated samples ───────────────────
    # Maps classifier output index → dataset label, so we can compare the
    # classifier's prediction against the LDM conditioning label.
    class_map_cfg = cfg["classifier_model"].get("class_map", None)
    if class_map_cfg:
        class_map = {int(k): int(v) for k, v in class_map_cfg.items()}
    else:
        n_clf = cfg["classifier_model"]["args"]["num_classes"]
        class_map = {i: i for i in range(n_clf)}

    n_classes = len(eval_classes)

    print("\n" + "=" * 60)
    print("  LDM Sample Quality Evaluation")
    print(f"  Classes: {eval_classes}  |  {n_per_class} samples each")
    print("=" * 60)

    sampler = DDIMSampler(ldm_model)

    all_samples: list[torch.Tensor] = []
    all_ldm_labels: list[int] = []

    for ldm_label in eval_classes:
        label_name = ldm_idx_to_dataset_label.get(ldm_label, ldm_label)
        print(f"  Generating class {ldm_label} ({label_name}) — {n_per_class} samples ...")
        class_tensor = torch.full(
            (n_per_class,), ldm_label, dtype=torch.long, device=device
        )
        c = ldm_model.get_learned_conditioning({ldm_model.cond_stage_key: class_tensor})
        shape = (
            n_per_class,
            ldm_model.channels,
            ldm_model.image_size,
            ldm_model.image_size,
        )

        samples_z, _ = sampler.sample(
            S=ddim_steps,
            batch_size=n_per_class,
            shape=shape[1:],
            conditioning=c,
            verbose=False,
            eta=ddim_eta,
        )
        samples_x = ldm_model.decode_first_stage(samples_z)
        samples_01 = (samples_x.clamp(-1, 1) + 1) / 2

        all_samples.append(samples_01.cpu())
        all_ldm_labels.extend([ldm_label] * n_per_class)

    all_samples_t = torch.cat(all_samples, dim=0)

    # ── Class-conditional sample grid (one row per class, labelled) ─────────
    grid_path = os.path.join(output_dir, "ldm_class_conditional_samples.png")
    img_size  = all_samples_t.shape[-1]        # spatial size in pixels
    label_col = max(48, img_size)              # width of the left label column
    pad       = 2

    fig_w = (n_per_class * img_size + (n_per_class - 1) * pad + label_col) / 100
    fig_h = (n_classes   * img_size + (n_classes   - 1) * pad)             / 100
    fig, axes = plt.subplots(n_classes, n_per_class + 1,
                             figsize=(fig_w + label_col / 100, fig_h + 0.4),
                             gridspec_kw={"width_ratios": [label_col / img_size] +
                                          [1] * n_per_class})
    if n_classes == 1:
        axes = [axes]

    for row_idx, ldm_label in enumerate(eval_classes):
        chunk = all_samples_t[row_idx * n_per_class: (row_idx + 1) * n_per_class]
        # Left cell: class label text
        axes[row_idx][0].text(0.5, 0.5, f"class\n{ldm_label}",
                              ha="center", va="center", fontsize=8,
                              transform=axes[row_idx][0].transAxes)
        axes[row_idx][0].axis("off")
        # Image cells
        for col_idx in range(n_per_class):
            img_np = chunk[col_idx].permute(1, 2, 0).numpy()
            axes[row_idx][col_idx + 1].imshow(img_np.squeeze(), cmap="gray"
                                              if img_np.shape[-1] == 1 else None)
            axes[row_idx][col_idx + 1].axis("off")

    fig.suptitle("LDM class-conditional samples (one row per class)", fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved class-conditional grid → {grid_path}")

    # ── Unconditional samples ─────────────────────────────────────────────
    print("  Generating unconditional samples ...")
    uc_labels = torch.full(
        (n_per_class,), uncond_class_idx, dtype=torch.long, device=device
    )
    uc_c = ldm_model.get_learned_conditioning({ldm_model.cond_stage_key: uc_labels})
    uc_z, _ = sampler.sample(
        S=ddim_steps,
        batch_size=n_per_class,
        shape=shape[1:],
        conditioning=uc_c,
        verbose=False,
        eta=ddim_eta,
    )
    uc_01 = (ldm_model.decode_first_stage(uc_z).clamp(-1, 1) + 1) / 2
    uc_grid_path = os.path.join(output_dir, "ldm_unconditional_samples.png")
    save_image(
        make_grid(uc_01.cpu(), nrow=n_per_class, padding=2, pad_value=0.5), uc_grid_path
    )
    print(f"  Saved unconditional grid → {uc_grid_path}")

    # ── Pixel statistics ──────────────────────────────────────────────────
    px_mean = all_samples_t.mean(dim=(1, 2, 3))
    px_std = all_samples_t.std(dim=(1, 2, 3))
    print(f"\n  Pixel brightness  mean: {px_mean.mean():.4f} ± {px_mean.std():.4f}")
    print(f"  Pixel brightness  std : {px_std.mean():.4f} ± {px_std.std():.4f}")

    # ── Classifier accuracy on generated samples ──────────────────────────
    print("\n  Scoring generated samples with classifier ...")
    correct = 0
    per_class_correct = {lbl: 0 for lbl in eval_classes}
    per_class_total = {lbl: 0 for lbl in eval_classes}

    for img, ldm_label in zip(all_samples_t, all_ldm_labels):
        logits = classifier(img.unsqueeze(0).to(device))
        clf_pred = logits.argmax(dim=1).item()  # classifier index
        # Convert classifier prediction back to dataset label via class_map
        pred_label = class_map.get(clf_pred, clf_pred)

        per_class_total[ldm_label] += 1
        if pred_label == ldm_label:
            correct += 1
            per_class_correct[ldm_label] += 1

    overall_acc = correct / len(all_ldm_labels) * 100
    print(f"\n  Overall classifier accuracy on LDM samples: {overall_acc:.1f}%")
    for lbl in eval_classes:
        tot = per_class_total[lbl]
        if tot > 0:
            acc = per_class_correct[lbl] / tot * 100
            label_name = ldm_idx_to_dataset_label.get(lbl, lbl)
            print(f"    Class {lbl} ({label_name}): {acc:.1f}%  ({per_class_correct[lbl]}/{tot})")

    # ── FID ───────────────────────────────────────────────────────────────
    if HAS_FID:
        print("\n  Computing FID ...")
        import shutil

        real_dir = os.path.join(output_dir, "_fid_real")
        gen_dir = os.path.join(output_dir, "_fid_gen")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)

        for i, img in enumerate(all_samples_t):
            save_image(img, os.path.join(gen_dir, f"{i:05d}.png"))

        # Real images from the evaluation dataset
        eval_ds = build_eval_dataset(cfg)
        n_real = min(len(eval_ds), len(all_samples_t))
        for i in range(n_real):
            img_01, _, _ = eval_ds[i]
            save_image(img_01, os.path.join(real_dir, f"{i:05d}.png"))

        try:
            fid = calculate_fid_given_paths(
                [real_dir, gen_dir], batch_size=64, device=str(device), dims=2048
            )
            print(f"  FID: {fid:.2f}")
        except Exception as exc:
            print(f"  FID computation failed: {exc}")

        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(gen_dir, ignore_errors=True)
    else:
        print("\n  Skipping FID (pip install pytorch-fid)")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_ldm(cfg: dict, device: torch.device) -> nn.Module:
    ldm_cfg = OmegaConf.load(cfg["diffusion_model"]["cfg_path"])
    model = instantiate_from_config(ldm_cfg.model)
    sd = torch.load(
        cfg["diffusion_model"]["ckpt_path"], weights_only=False, map_location="cpu"
    )
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  LDM missing keys: {len(missing)}")
    model.to(device).eval()
    print(f"Loaded LDM from {cfg['diffusion_model']['ckpt_path']}")
    return model


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config-driven LDM evaluation")
    p.add_argument(
        "--config",
        required=True,
        help="Path to LDCE config YAML (same format as ldce_runner_new.py)",
    )
    p.add_argument(
        "--output_dir", default=None, help="Override output directory from config"
    )
    p.add_argument(
        "--device", default=None, help="torch device (e.g. cuda, cuda:1, cpu)"
    )
    p.add_argument(
        "--n_per_class",
        type=int,
        default=None,
        help="Samples to generate per class (overrides eval_n_per_class in config)",
    )
    p.add_argument(
        "--ddim_steps",
        type=int,
        default=None,
        help="DDIM steps (overrides eval_ddim_steps in config)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for VAE evaluation dataloader",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.n_per_class is not None:
        cfg["eval_n_per_class"] = args.n_per_class
    if args.ddim_steps is not None:
        cfg["eval_ddim_steps"] = args.ddim_steps

    output_dir = args.output_dir or os.path.join(
        cfg.get("output_dir", "assets/results"), "eval"
    )
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Device    : {device}")
    print(f"Config    : {args.config}")
    print(f"Output    : {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)

    # ── Classifier ────────────────────────────────────────────────────────
    print("Loading classifier ...")
    classifier = build_classifier(cfg, device)

    # ── LDM ───────────────────────────────────────────────────────────────
    print("Loading LDM ...")
    ldm_model = load_ldm(cfg, device)

    # ── LDM evaluation ────────────────────────────────────────────────────
    ldm_dir = os.path.join(output_dir, "ldm")
    evaluate_ldm(ldm_model, classifier, cfg, device, ldm_dir)

    print("\n" + "=" * 60)
    print("  Evaluation complete")
    print(f"  Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
