"""
ldce_runner.py  –  cluster-ready LDCE counterfactual generation for MNIST.

For each sample the script generates a counterfactual explanation for:
  • the original input image
  • a noise-perturbed version of the same image  (if perturbation.enabled)

Usage (from repository root):
    python ldce_runner.py --config mnist_ldce/config.yaml
    python ldce_runner.py --config mnist_ldce/config.yaml --device cuda:1

Output layout  (under output_dir/):
    original/           original input images
    cf_original/        counterfactuals of originals
    perturbed/          perturbed inputs          (if perturbation.enabled)
    cf_perturbed/       counterfactuals of perturbed inputs
    metadata/           per-sample .pth dicts
    completed.txt       append-only log of finished sample IDs
"""

import argparse
import contextlib
import importlib.abc
import importlib.machinery
import os
import pathlib
import pickle
import random
import sys
import types

import numpy as np
from numpy.random import multivariate_normal, uniform
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image

from sampling_helpers import get_model, _unmap_img
from ldm.models.diffusion.cc_ddim import CCMDDIMSampler
from ldm.models.classifiers import CNNtorch
from utils.preprocessor import Normalizer
from mnist_ldce.dataset import MNISTForLDCE, DIGIT_NAMES, MNIST_CLOSEST_CLASS

UNCOND_CLASS_IDX = 10   # null class index in the DDPM's ClassEmbedder


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking  (append-only, no folder scans during the run)
# ─────────────────────────────────────────────────────────────────────────────

class CompletedTracker:
    """Tracks attempted sample IDs so they are never re-processed.

    On startup the log file is read once into a set.  During the run,
    mark_done() appends to the file and updates the set in O(1) — no
    directory scans are performed after initialisation.
    """

    def __init__(self, log_path: str) -> None:
        self._path = log_path
        self._done: set[int] = set()
        if os.path.exists(log_path):
            with open(log_path) as fh:
                self._done = {int(ln) for ln in fh if ln.strip().isdigit()}
        print(f"  CompletedTracker: {len(self._done)} samples already done")

    @property
    def done(self) -> set[int]:
        return self._done

    def is_done(self, idx: int) -> bool:
        return idx in self._done

    def mark_done(self, idx: int) -> None:
        if idx not in self._done:
            self._done.add(idx)
            with open(self._path, "a") as fh:
                fh.write(f"{idx}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Classifier components
# ─────────────────────────────────────────────────────────────────────────────

class EvalOnlyModule(nn.Module):
    """Wraps a module and ignores all calls to .train(), keeping it in eval mode.

    Replaces the monkey-patch `model.train = disabled_train` pattern.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.eval()

    def train(self, mode: bool = True) -> "EvalOnlyModule":  # noqa: ARG002
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResizeWrapper(nn.Module):
    """Bilinear resize + optional channel reduction before classification.

    The diffusion pipeline operates at pipeline_size × pipeline_size while
    the classifier was trained at clf_size × clf_size on out_channels images.
    """

    def __init__(self, model: nn.Module, clf_size: int,
                 out_channels: int = 1) -> None:
        super().__init__()
        self.model        = model
        self.clf_size     = clf_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.clf_size, self.clf_size),
                          mode="bilinear", align_corners=False)
        return self.model(x[:, :self.out_channels])


@contextlib.contextmanager
def _src_stub_context():
    """Temporarily stub 'src.*' imports so torch.load can unpickle checkpoints
    saved from a codebase where the classifier class lived under src/."""

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


def load_classifier(args: dict, ckpt_path: str,
                    device: torch.device) -> nn.Module:
    model = CNNtorch(args["input_channels"], args["num_classes"])
    with _src_stub_context():
        checkpoint = torch.load(ckpt_path, weights_only=False,
                                map_location=device)
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def build_classifier(cfg: dict, device: torch.device) -> nn.Module:
    """Load classifier, apply resize/channel and normalisation wrappers."""
    args          = cfg["classifier_model"]["args"]
    clf_size      = cfg["classifier_model"].get("input_size",
                                                cfg["data"]["image_size"])
    pipeline_size = cfg["data"]["image_size"]
    in_ch         = args.get("input_channels", 1)

    model = load_classifier(args, cfg["classifier_model"]["ckpt_path"], device)

    if clf_size != pipeline_size or in_ch != 3:
        print(f"  Classifier resize: {pipeline_size}→{clf_size}, "
              f"channels: 3→{in_ch}")
        model = ResizeWrapper(model, clf_size, out_channels=in_ch)

    if cfg["classifier_model"].get("mnist_normalisation", False):
        model = Normalizer(model, [0.1307] * in_ch, [0.3081] * in_ch)

    return EvalOnlyModule(model).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion model & sampler
# ─────────────────────────────────────────────────────────────────────────────

def build_diffusion_model(cfg: dict, device: torch.device) -> nn.Module:
    return get_model(
        cfg_path  = cfg["diffusion_model"]["cfg_path"],
        ckpt_path = cfg["diffusion_model"]["ckpt_path"],
    ).to(device).eval()


def build_sampler(diff_model: nn.Module, classifier: nn.Module,
                  cfg: dict) -> tuple[CCMDDIMSampler, int]:
    sampler = CCMDDIMSampler(
        diff_model, classifier,
        seg_model                   = None,
        classifier_wrapper          = cfg["classifier_model"].get(
                                          "classifier_wrapper", True),
        record_intermediate_results = cfg.get(
                                          "record_intermediate_results", False),
        verbose                     = True,
        **cfg["sampler"],
    )
    sampler.make_schedule(ddim_num_steps=cfg["ddim_steps"],
                          ddim_eta=cfg["ddim_eta"], verbose=False)
    t_enc = int(cfg["strength"] * len(sampler.ddim_timesteps))
    return sampler, t_enc


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation
# ─────────────────────────────────────────────────────────────────────────────

def perturb_sample(
    input_images: np.ndarray,
    n_samples: int = 1,
    type: str = "uniform",
    epsilon: float | None = None,
    channels_last: bool = False,
    std: float = 0.1,
    noise_seed: int | None = None,
) -> np.ndarray:
    """Generate perturbed samples around the input images.

    Args:
        input_images : numpy array of shape (B, C, H, W) or (C, H, W).
        n_samples    : number of perturbed samples to generate per image.
        type         : 'normal' or 'uniform'.
        epsilon      : noise magnitude / clip bound.
        channels_last: set True if input is (B, H, W, C); False for PyTorch
                       (B, C, H, W) convention (default).
        std          : standard deviation for normal noise.
        noise_seed   : if given, numpy RNG is reset to this seed before
                       sampling — guarantees identical base draws when called
                       repeatedly with different epsilon values.
    Returns:
        numpy array of shape (B, n_samples, C, H, W) or (B, n_samples, H, W, C).
    """
    data = input_images
    if len(input_images.shape) == 3:
        data = np.expand_dims(input_images, axis=0)

    if channels_last:
        # input layout: (B, H, W, C)
        batch_size, height, width, channels = data.shape
        result_shape = (batch_size, n_samples, height, width, channels)
    else:
        # input layout: (B, C, H, W)  ← PyTorch default
        batch_size, channels, height, width = data.shape
        result_shape = (batch_size, n_samples, channels, height, width)

    data = np.expand_dims(data, axis=1)
    data = np.tile(data, reps=[1, n_samples, 1, 1, 1])

    if noise_seed is not None:
        np.random.seed(noise_seed)

    if type == "normal":
        mean       = np.zeros(channels * height * width)
        covariance = np.eye(channels * height * width) * std ** 2
        noise      = multivariate_normal(mean, covariance,
                                         size=(batch_size, n_samples))
        noise      = noise.reshape(*result_shape)
        if epsilon is not None:
            noise = np.clip(noise, -epsilon, epsilon)

    elif type == "uniform":
        noise = uniform(-epsilon, epsilon, size=result_shape)

    else:
        raise ValueError(f"Unknown perturbation type: '{type}'")

    perturbed_data = np.clip(data + noise, a_min=0, a_max=1)
    return perturbed_data


# ─────────────────────────────────────────────────────────────────────────────
# Target class selection
# ─────────────────────────────────────────────────────────────────────────────

def get_target_classes(labels: torch.Tensor, logits: torch.Tensor,
                       method: str, fixed_target: int | None,
                       device: torch.device) -> torch.Tensor:
    if method == "fixed":
        return torch.full_like(labels, fixed_target)

    targets = []
    for j in range(len(labels)):
        lbl = labels[j].item()
        if method == "closest":
            tgt = MNIST_CLOSEST_CLASS[lbl][0]
        elif method == "random":
            tgt = random.choice([c for c in range(10) if c != lbl])
        elif method == "second_best":
            tgt = next(
                (c for c in logits[j].argsort(descending=True).tolist()
                 if c != lbl),
                MNIST_CLOSEST_CLASS[lbl][0],
            )
        else:
            raise ValueError(f"Unknown target_class_method: '{method}'")
        targets.append(tgt)
    return torch.tensor(targets, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Counterfactual generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_cf(diff_model: nn.Module, sampler: CCMDDIMSampler,
                images: torch.Tensor, target_classes: torch.Tensor,
                scale: float, t_enc: int, seed: int) -> torch.Tensor:
    """Encode → forward diffuse → guided denoise → decode.

    Args:
        images        : (B, C, H, W) in [0, 1] on the model's device.
        target_classes: (B,) integer class indices.

    Returns:
        (B, C, H, W) counterfactual images in [0, 1].
    """
    sampler.init_images = images.clone()
    B = target_classes.shape[0]

    init_latent = diff_model.get_first_stage_encoding(
        diff_model.encode_first_stage(_unmap_img(images))
    )

    with torch.no_grad(), diff_model.ema_scope():
        uc = diff_model.get_learned_conditioning({
            diff_model.cond_stage_key: torch.full(
                (B,), UNCOND_CLASS_IDX, dtype=torch.long,
                device=diff_model.device,
            )
        })
        c = diff_model.get_learned_conditioning({
            diff_model.cond_stage_key: target_classes.to(diff_model.device)
        })

        torch.manual_seed(seed)
        z_enc = sampler.stochastic_encode(
            init_latent,
            torch.full((B,), t_enc, device=init_latent.device),
            noise=torch.randn_like(init_latent),
        )

        torch.manual_seed(seed)
        out = sampler.decode(
            z_enc, c, t_enc,
            unconditional_guidance_scale = scale,
            unconditional_conditioning   = uc,
            y                            = target_classes.to(diff_model.device),
            latent_t_0                   = False,
        )

    return torch.clamp(
        (diff_model.decode_first_stage(out["x_dec"]) + 1.0) / 2.0, 0.0, 1.0
    )


# ─────────────────────────────────────────────────────────────────────────────
# Result saving
# ─────────────────────────────────────────────────────────────────────────────

def _cf_metrics(src: torch.Tensor, cf: torch.Tensor,
                softmax: torch.Tensor, tgt: int) -> dict:
    diff = src - cf
    return {
        "image_cf":      cf,
        "pred":          DIGIT_NAMES[softmax.argmax().item()],
        "confid":        softmax.max().item(),
        "tgt_confid":    softmax[tgt].item(),
        "l1":            torch.norm(diff, p=1).item(),
        "l2":            torch.norm(diff, p=2).item(),
    }


def save_sample(out_dir: str, uidx: int,
                src: torch.Tensor, cf_orig: torch.Tensor,
                label: int, tgt: int,
                softmax_in: torch.Tensor,
                softmax_cf_orig: torch.Tensor,
                cf_pert: torch.Tensor | None = None,
                perturbed: torch.Tensor | None = None,
                softmax_perturbed: torch.Tensor | None = None,
                softmax_cf_pert: torch.Tensor | None = None,
                save_only_successful: bool = True) -> dict | None:
    """Build a record for one sample.

    Returns a dict with the original image, its counterfactual explanation,
    source class, and target class, or None if the sample is skipped.
    The caller is responsible for accumulating records and writing the .pkl.
    """
    orig_success = softmax_cf_orig.argmax().item() == tgt
    if save_only_successful and not orig_success:
        return None

    record = {
        "unique_id": uidx,
        "source":    DIGIT_NAMES[label],
        "target":    DIGIT_NAMES[tgt],
        "image":     src.cpu(),
        "image_cf":  cf_orig.cpu(),
    }

    if cf_pert is not None:
        pert_success = softmax_cf_pert.argmax().item() == tgt
        if not save_only_successful or pert_success:
            record["image_perturbed"]    = perturbed.cpu()
            record["image_cf_perturbed"] = cf_pert.cpu()

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Classifier evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(classifier: nn.Module, cfg: dict,
                        device: torch.device) -> None:
    clf_size = cfg["classifier_model"].get("input_size",
                                           cfg["data"]["image_size"])
    split    = cfg["data"]["split"]
    dataset  = MNISTForLDCE(root=cfg["data"]["root"], split=split,
                             image_size=clf_size)
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
    )

    correct = {c: 0 for c in range(10)}
    total   = {c: 0 for c in range(10)}

    with torch.inference_mode():
        for images, labels, _ in loader:
            preds = classifier(images.to(device)).argmax(dim=1).cpu()
            for pred, lbl in zip(preds.tolist(), labels.tolist()):
                total[lbl]   += 1
                correct[lbl] += int(pred == lbl)

    n_correct = sum(correct.values())
    n_total   = sum(total.values())
    accuracy  = n_correct / n_total if n_total else 0.0

    print(f"\nClassifier  —  MNIST {split}  ({n_total} images)")
    print(f"  Overall: {accuracy * 100:.2f}%  ({n_correct}/{n_total})")
    for c in range(10):
        acc = correct[c] / total[c] if total[c] else 0.0
        print(f"    {DIGIT_NAMES[c]:>7} ({c}):  {acc * 100:6.2f}%  "
              f"{'█' * int(acc * 20)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def prepare_output_dirs(out_dir: str) -> None:
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)


def load_results_pkl(pkl_path: str) -> list[dict]:
    """Load existing results pkl, or return an empty list if none exists."""
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)
    return []


def save_results_pkl(pkl_path: str, records: list[dict]) -> None:
    with open(pkl_path, "wb") as fh:
        pickle.dump(records, fh)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: dict, device: torch.device) -> None:
    seed = cfg.get("seed", 0)
    set_seed(seed)

    out_dir        = cfg["output_dir"]
    pert_cfg       = cfg.get("perturbation", {})
    with_pert      = pert_cfg.get("enabled", False)
    save_only_succ = cfg.get("save_only_successful", True)

    prepare_output_dirs(out_dir)
    tracker  = CompletedTracker(os.path.join(out_dir, "completed.txt"))
    pkl_path = os.path.join(out_dir, "results.pkl")
    records  = load_results_pkl(pkl_path)
    print(f"  results.pkl: {len(records)} records already saved")

    # ── Models ────────────────────────────────────────────────────────────────
    print("Loading classifier …")
    classifier = build_classifier(cfg, device)

    print("Loading diffusion model …")
    diff_model = build_diffusion_model(cfg, device)

    if cfg.get("evaluate_classifier", False):
        evaluate_classifier(classifier, cfg, device)

    sampler, t_enc = build_sampler(diff_model, classifier, cfg)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = MNISTForLDCE(
        root           = cfg["data"]["root"],
        split          = cfg["data"]["split"],
        image_size     = cfg["data"]["image_size"],
        max_samples    = cfg["data"].get("max_samples"),
        filter_classes = cfg["data"].get("filter_classes"),
        skip_ids       = tracker.done,
    )
    print(f"Dataset: {len(dataset)} samples remaining  "
          f"({len(tracker.done)} already completed)\n")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = cfg["data"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["data"].get("num_workers", 4),
        pin_memory  = device.type == "cuda",
    )

    target_method = cfg.get("target_class_method", "closest")
    fixed_target  = cfg.get("target_class")
    batch_size    = cfg["data"]["batch_size"]
    total_saved   = 0

    # ── Generation loop ───────────────────────────────────────────────────────
    for i, (images, labels, unique_ids) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            logits_in = classifier(images)

        tgt_classes = get_target_classes(
            labels, logits_in, target_method, fixed_target, device
        )

        for j in range(len(labels)):
            print(f"  [{i * batch_size + j:05d}]  "
                  f"{DIGIT_NAMES[labels[j].item()]} → "
                  f"{DIGIT_NAMES[tgt_classes[j].item()]}")

        # ── Original counterfactuals ──────────────────────────────────────────
        cf_orig = generate_cf(
            diff_model, sampler, images, tgt_classes,
            cfg["scale"], t_enc, seed,
        )

        with torch.inference_mode():
            logits_cf_orig = classifier(cf_orig)

        # ── Perturbed counterfactuals ─────────────────────────────────────────
        cf_pert          = None
        logits_perturbed = None
        logits_cf_pert   = None
        perturbed        = None

        if with_pert:
            # perturb_sample works on numpy (B, C, H, W); n_samples=1 so we
            # squeeze the samples axis to get back (B, C, H, W).
            images_np = images.cpu().numpy()
            perturbed_np = perturb_sample(
                images_np,
                n_samples    = 1,
                type         = pert_cfg.get("type", "uniform"),
                epsilon      = pert_cfg.get("magnitude", 0.1),
                channels_last= False,
                std          = pert_cfg.get("std", 0.1),
                noise_seed   = pert_cfg.get("noise_seed"),
            )                                          # (B, 1, C, H, W)
            perturbed = torch.from_numpy(
                perturbed_np[:, 0]                     # (B, C, H, W)
            ).to(device)
            cf_pert   = generate_cf(
                diff_model, sampler, perturbed, tgt_classes,
                cfg["scale"], t_enc, seed,
            )
            with torch.inference_mode():
                logits_perturbed = classifier(perturbed)
                logits_cf_pert   = classifier(cf_pert)

        # ── Save & track ──────────────────────────────────────────────────────
        sm_in       = logits_in.softmax(dim=1).cpu()
        sm_cf_orig  = logits_cf_orig.softmax(dim=1).cpu()
        sm_pert     = logits_perturbed.softmax(dim=1).cpu() if logits_perturbed is not None else None
        sm_cf_pert  = logits_cf_pert.softmax(dim=1).cpu()  if logits_cf_pert   is not None else None

        batch_saved = 0
        for j in range(len(labels)):
            uidx = unique_ids[j].item()
            tgt  = tgt_classes[j].item()

            record = save_sample(
                out_dir,
                uidx,
                src              = images[j].cpu(),
                cf_orig          = cf_orig[j].cpu(),
                label            = labels[j].item(),
                tgt              = tgt,
                softmax_in       = sm_in[j],
                softmax_cf_orig  = sm_cf_orig[j],
                cf_pert          = cf_pert[j].cpu()          if cf_pert          is not None else None,
                perturbed        = perturbed[j].cpu()        if perturbed        is not None else None,
                softmax_perturbed= sm_pert[j]                if sm_pert          is not None else None,
                softmax_cf_pert  = sm_cf_pert[j]             if sm_cf_pert       is not None else None,
                save_only_successful=save_only_succ,
            )
            tracker.mark_done(uidx)
            if record is not None:
                records.append(record)
                batch_saved += 1

        save_results_pkl(pkl_path, records)
        total_saved += batch_saved
        print(f"  Saved {batch_saved}/{len(labels)} this batch  "
              f"(total: {total_saved}, pkl: {len(records)} records)")
        print(f"  orig  preds: {logits_in.argmax(1).tolist()}  →  "
              f"{logits_cf_orig.argmax(1).tolist()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LDCE counterfactual generation for MNIST"
    )
    p.add_argument("--config", required=True,
                   help="Path to config.yaml")
    p.add_argument("--device", default=None,
                   help="torch device (e.g. cuda, cuda:1, cpu). "
                        "Defaults to cuda if available.")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")
    print(f"Config : {args.config}\n")
    main(cfg, device)
