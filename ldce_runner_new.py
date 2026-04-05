"""
ldce_runner.py  –  cluster-ready LDCE counterfactual generation.

For each sample the script generates a counterfactual explanation for:
  • the original input image
  • a noise-perturbed version of the same image  (if perturbation.enabled)

Usage (from repository root):
    python ldce_runner.py --config mnist_ldce/config.yaml --device cuda:1

# TODO: update the output
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
import easydict

import numpy as np
from numpy.random import multivariate_normal, uniform
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image

from sampling_helpers import get_model, _unmap_img
from ldm.models.diffusion.cc_ddim import CCMDDIMSampler
import ldm.models.classifiers as _clf_module
from utils.preprocessor import Normalizer
from ldm.data.datasets import MNIST, FashionMNIST, DermaMNIST
from ldm.data.utils import DIGIT_NAMES, MNIST_CLOSEST_CLASS

UNCOND_CLASS_IDX = 10  # null class index in the DDPM's ClassEmbedder
_DATASET_MAP = {"MNIST": MNIST, "FashionMNIST": FashionMNIST, "DermaMNIST": DermaMNIST}


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


def load_classifier(args: dict, ckpt_path: str, device: torch.device) -> nn.Module:
    args = dict(args)  # don't mutate the caller's dict
    cls_name = args.pop("model_class", "CNNtorch")
    cls = getattr(_clf_module, cls_name)
    lightning_used = args.pop("lightning_used", False)
    
    model = cls(**args)
    with _src_stub_context():
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
        
        if lightning_used and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def build_classifier(cfg: dict, device: torch.device) -> nn.Module:
    """Load classifier, apply resize/channel and normalisation wrappers."""
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
# Diffusion model & sampler
# ─────────────────────────────────────────────────────────────────────────────


def build_diffusion_model(cfg: dict, device: torch.device) -> nn.Module:
    return (
        get_model(
            cfg_path=cfg["diffusion_model"]["cfg_path"],
            ckpt_path=cfg["diffusion_model"]["ckpt_path"],
        )
        .to(device)
        .eval()
    )


def build_sampler(
    diff_model: nn.Module, classifier: nn.Module, cfg: dict
) -> tuple[CCMDDIMSampler, int]:
    sampler = CCMDDIMSampler(
        diff_model,
        classifier,
        seg_model=None,
        classifier_wrapper=cfg["classifier_model"].get("classifier_wrapper", True),
        record_intermediate_results=cfg.get("record_intermediate_results", False),
        verbose=True,
        **cfg["sampler"],
    )
    sampler.make_schedule(
        ddim_num_steps=cfg["ddim_steps"], ddim_eta=cfg["ddim_eta"], verbose=False
    )
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
        mean = np.zeros(channels * height * width)
        covariance = np.eye(channels * height * width) * std**2
        noise = multivariate_normal(mean, covariance, size=(batch_size, n_samples))
        noise = noise.reshape(*result_shape)
        if epsilon is not None:
            noise = np.clip(noise, -epsilon, epsilon)

    elif type == "uniform":
        noise = uniform(-epsilon, epsilon, size=result_shape)

    else:
        raise ValueError(f"Unknown perturbation type: '{type}'")

    perturbed_data = np.clip(data + noise, a_min=0, a_max=1)
    return perturbed_data


# ─────────────────────────────────────────────────────────────────────────────
# Counterfactual generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_cf(
    diff_model: nn.Module,
    sampler: CCMDDIMSampler,
    images: torch.Tensor,
    target_classes: torch.Tensor,
    scale: float,
    t_enc: int,
    seed: int,
    inv_class_map: dict,
) -> torch.Tensor:
    """Encode → forward diffuse → guided denoise → decode.

    Returns:
        (B, C, H, W) counterfactual images in [0, 1].
    """
    sampler.init_images = images.clone()
    B = target_classes.shape[0]
    if inv_class_map is not None:
        y_clf = torch.tensor(
            [inv_class_map[t.item()] for t in target_classes],
            dtype=torch.long,
            device=diff_model.device,
        )
    else:
        y_clf = target_classes.to(diff_model.device)

    init_latent = diff_model.get_first_stage_encoding(
        diff_model.encode_first_stage(_unmap_img(images))
    )

    with torch.no_grad(), diff_model.ema_scope():
        uc = diff_model.get_learned_conditioning(
            {
                diff_model.cond_stage_key: torch.full(
                    (B,),
                    UNCOND_CLASS_IDX,
                    dtype=torch.long,
                    device=diff_model.device,
                )
            }
        )
        c = diff_model.get_learned_conditioning(
            {diff_model.cond_stage_key: target_classes.to(diff_model.device)}
        )

        torch.manual_seed(seed)
        z_enc = sampler.stochastic_encode(
            init_latent,
            torch.full((B,), t_enc, device=init_latent.device),
            noise=torch.randn_like(init_latent),
        )

        torch.manual_seed(seed)

        out = sampler.decode(
            z_enc,
            c,
            t_enc,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            y=y_clf,
            latent_t_0=False,
        )

    return torch.clamp(
        (diff_model.decode_first_stage(out["x_dec"]) + 1.0) / 2.0, 0.0, 1.0
    )


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
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


def build_dataset(cfg: dict, skip_ids: set) -> torch.utils.data.Dataset:
    """Instantiate the dataset specified by cfg['data']['name'].

    Supported names (case-sensitive): 'MNIST', 'FashionMNIST', 'DermaMNIST'.
    All parameters are read from cfg['data'].
    """
    data_cfg = cfg["data"]
    name = data_cfg.get("name", "MNIST")
    if name not in _DATASET_MAP:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(_DATASET_MAP)}")
    cls = _DATASET_MAP[name]

    if name in ("MNIST", "FashionMNIST"):
        return cls(
            root=data_cfg["root"],
            split=data_cfg["split"],
            image_size=data_cfg["image_size"],
            max_samples=data_cfg.get("max_samples"),
            classes=data_cfg.get("filter_classes"),
            skip_ids=skip_ids,
        )

    if name == "DermaMNIST":
        return cls(
            root=data_cfg["root"],
            image_size=data_cfg["image_size"],
            split=data_cfg["split"],
            download=data_cfg.get("download", True),
            undersample=data_cfg.get("undersample", True),
            channels_first=data_cfg.get("channels_first", True),
            classes=data_cfg.get("filter_classes", ["all"]),
        )

    raise ValueError(f"Unhandled dataset name: '{name}'")  # unreachable


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(cfg: dict, device: torch.device) -> None:
    seed = cfg.get("seed", 10)
    set_seed(seed)

    out_dir = cfg["output_dir"]
    pert_cfg = cfg.get("perturbation", {})
    with_pert = pert_cfg.get("enabled", False)
    save_only_succ = cfg.get("save_only_successful", True)

    # TODO : change saving strategy to the more optimal one
    prepare_output_dirs(out_dir)
    tracker = CompletedTracker(os.path.join(out_dir, "completed.txt"))

    pkl_path_orig = os.path.join(out_dir, "results_original.pkl")
    pkl_path_pert = os.path.join(out_dir, "results_perturbed.pkl")
    records_orig = load_results_pkl(pkl_path_orig)
    records_pert = load_results_pkl(pkl_path_pert)
    print(f"  results_original.pkl : {len(records_orig)} records already saved")
    print(f"  results_perturbed.pkl: {len(records_pert)} records already saved")

    class_map = cfg["classifier_model"].get("class_map", None)
    if class_map:
        class_map = {int(k): int(v) for k, v in class_map.items()}
        inv_class_map = {v: k for k, v in class_map.items()}
    else:
        # multiclass: identity mapping
        class_map = {i: i for i in range(10)}
        inv_class_map = class_map

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = build_dataset(cfg, tracker.done)
    print(
        f"Dataset ({cfg['data'].get('name', 'MNIST')}): "
        f"{len(dataset)} samples remaining  "
        f"({len(tracker.done)} already completed)\n"
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    # target_method = cfg.get("target_class_method", "closest")
    fixed_target = cfg.get("target_class")
    total_orig = 0
    total_pert = 0

    # ── Models ────────────────────────────────────────────────────────────────
    print("Loading classifier …")
    classifier = build_classifier(cfg, device)

    print("Loading diffusion model …")
    diff_model = build_diffusion_model(cfg, device)

    # if cfg.get("evaluate_classifier", False):
    #     evaluate_classifier(classifier, cfg, device)

    sampler, t_enc = build_sampler(diff_model, classifier, cfg)

    # ── Generation loop ───────────────────────────────────────────────────────
    for i, (images, labels, unique_ids) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            logits_in = classifier(images)

        tgt_classes = torch.full_like(labels, fixed_target).to(device)

        # ── Original counterfactuals ──────────────────────────────────────────
        cf_orig = generate_cf(
            diff_model,
            sampler,
            images,
            tgt_classes,
            cfg["scale"],
            t_enc,
            seed,
            inv_class_map,
        )

        with torch.inference_mode():
            logits_cf_orig = classifier(cf_orig)

        sm_in = logits_in.softmax(dim=1).cpu()
        sm_cf_orig = logits_cf_orig.softmax(dim=1).cpu()

        # ── Collect original records ──────────────────────────────────────────
        batch_orig = []
        for j in range(len(labels)):
            uidx = unique_ids[j].item()
            tgt = tgt_classes[j].item()
            label = labels[j].item()

            orig_success = sm_cf_orig[j].argmax().item() == tgt
            tracker.mark_done(uidx)
            if save_only_succ and not orig_success:
                continue

            batch_orig.append(
                {
                    "unique_id": uidx,
                    "source": DIGIT_NAMES[label],
                    "target": DIGIT_NAMES[tgt],
                    "image": images[j].cpu(),
                    "image_cf": cf_orig[j].cpu(),
                }
            )

        # ── Perturbed counterfactuals (one record per sample × epsilon) ───────
        batch_pert = []
        if with_pert:
            epsilon_list = pert_cfg.get("magnitude", [0.1])
            pert_type = pert_cfg.get("type", "uniform")
            images_np = images.cpu().numpy()

            for epsilon in epsilon_list:
                print(
                    f"Perturbing original images with the noise bounded by eps={epsilon}"
                )
                # perturb_sample returns (B, n_samples, C, H, W); n_samples=1
                perturbed_np = perturb_sample(
                    images_np,
                    n_samples=1,
                    type=pert_type,
                    epsilon=epsilon,
                    channels_last=False,
                    std=pert_cfg.get("std", 0.1),
                    noise_seed=pert_cfg.get("noise_seed"),
                )  # → (B, 1, C, H, W)
                perturbed = torch.from_numpy(
                    perturbed_np[:, 0].astype(np.float32)  # → (B, C, H, W) float32
                ).to(device)

                cf_pert = generate_cf(
                    diff_model,
                    sampler,
                    perturbed,
                    tgt_classes,
                    cfg["scale"],
                    t_enc,
                    seed,
                    inv_class_map,
                )
                with torch.inference_mode():
                    logits_cf_pert = classifier(cf_pert)

                sm_cf_pert = logits_cf_pert.softmax(dim=1).cpu()

                for j in range(len(labels)):
                    tgt = tgt_classes[j].item()
                    label = labels[j].item()

                    pert_success = sm_cf_pert[j].argmax().item() == tgt
                    if save_only_succ and not pert_success:
                        continue

                    batch_pert.append(
                        {
                            "unique_id": unique_ids[j].item(),
                            "epsilon": epsilon,
                            "source": DIGIT_NAMES[label],
                            "target": DIGIT_NAMES[tgt],
                            "image_perturbed": perturbed[j].cpu(),
                            "image_cf_perturbed": cf_pert[j].cpu(),
                        }
                    )

        # ── Persist & report ──────────────────────────────────────────────────
        records_orig.extend(batch_orig)
        records_pert.extend(batch_pert)
        save_results_pkl(pkl_path_orig, records_orig)
        save_results_pkl(pkl_path_pert, records_pert)

        total_orig += len(batch_orig)
        total_pert += len(batch_pert)
        print(
            f"  Batch {i}: +{len(batch_orig)} orig, +{len(batch_pert)} pert  "
            f"(running totals — orig: {total_orig}, pert: {total_pert})"
        )
        print(
            f"  orig preds: {logits_in.argmax(1).tolist()}  →  "
            f"{logits_cf_orig.argmax(1).tolist()}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LDCE counterfactual generation for MNIST")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument(
        "--device",
        default=None,
        help="torch device (e.g. cuda, cuda:1, cpu). Defaults to cuda if available.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # args = easydict.EasyDict({'config': 'ldce/mnist_ldce/configs/mnist/config_ldce_mnist.yaml',
    #         'device': 'cuda'})
    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")
    print(f"Config : {args.config}\n")
    main(cfg, device)
