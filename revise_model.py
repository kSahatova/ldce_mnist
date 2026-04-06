from typing import Dict, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn

from tqdm import tqdm

# from src.models.vae import VAE


def merge_default_parameters(hyperparams: Optional[Dict], default: Dict) -> Dict:
    """
    Checks if the input parameter hyperparams contains every necessary key and if not, uses default values or
    raises a ValueError if no default value is given.

    Parameters
    ----------
    hyperparams: dict
        Hyperparameter as passed to the recourse method.
    default: dict
        Dictionary with every necessary key and default value.
        If key has no default value and hyperparams has no value for it, raise a ValueError

    Returns
    -------
    dict
        Dictionary with every necessary key.
    """
    if hyperparams is None:
        return default

    keys = default.keys()
    dict_output = dict()

    for key in keys:
        if isinstance(default[key], dict):
            hyperparams[key] = (
                dict() if key not in hyperparams.keys() else hyperparams[key]
            )
            sub_dict = merge_default_parameters(hyperparams[key], default[key])
            dict_output[key] = sub_dict
            continue
        if key not in hyperparams.keys():
            default_val = default[key]
            if default_val is None:
                # None value for key depicts that user has to pass this value in hyperparams
                raise ValueError(
                    "For {} is no default value defined, please pass this key and its value in hyperparams".format(
                        key
                    )
                )
            elif isinstance(default_val, str) and default_val == "_optional_":
                # _optional_ depicts that value for this key is optional and therefore None
                default_val = None
            dict_output[key] = default_val
        else:
            if hyperparams[key] is None:
                raise ValueError("For {} in hyperparams is a value needed".format(key))
            dict_output[key] = hyperparams[key]

    return dict_output


class Revise:
    _DEFAULT_HYPERPARAMS = {
        "lambda": 0.01,  # 0.5 - default
        "optimizer": "adam",
        "lr": 0.5,
        "max_iter": 1000,
        "target_class_ind": [
            [0.0, 1.0]
        ],  # probabilities of the class 0 (digit 1) and the class 1 (digit 8)
    }

    def __init__(self, mlmodel, vae: nn.Module, hyperparams: Dict = None) -> None:
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        self._mlmodel = mlmodel
        self._lambda = self._params["lambda"]
        self._optimizer_name = self._params["optimizer"]
        self._lr = self._params["lr"]
        self._max_iter = self._params["max_iter"]
        self._target_class = self._params["target_class_ind"]
        self.vae = vae
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_counterfactuals(
        self, factuals: torch.Tensor, verbose: bool
    ) -> torch.Tensor:
        return self._counterfactual_optimization(factuals, verbose)

    def _counterfactual_optimization(self, factuals: torch.Tensor, verbose: bool):
        self._mlmodel.eval()
        self._mlmodel = self._mlmodel.to(self.device)
        self.vae = self.vae.to(self.device)

        N = factuals.shape[0]
        factuals = factuals.to(self.device)

        target = torch.FloatTensor(self._target_class).to(self.device)  # [1, num_classes]
        target_prediction = torch.argmax(target)
        target_batch = target.expand(N, -1)  # [N, num_classes]

        # Encode all instances in one forward pass
        z_mu, _ = self.vae.encoder(factuals)
        z = z_mu.clone().detach().requires_grad_(True)

        if self._optimizer_name == "Adam":
            optim = torch.optim.Adam([z], self._lr)
        else:
            optim = torch.optim.RMSprop([z], self._lr)

        # Per-instance candidate tracking
        candidate_counterfactuals = [[] for _ in range(N)]
        candidate_distances = [[] for _ in range(N)]

        for _ in tqdm(range(self._max_iter)):
            cf = self.vae.decoder(z)                          # [N, C, H, W]
            output = self._mlmodel(cf)                        # [N, num_classes]
            predicted = torch.argmax(output, dim=1)           # [N]

            loss_per_instance = self._compute_loss(cf, factuals, target_batch)  # [N]

            valid_mask = (predicted == target_prediction)
            if valid_mask.any():
                cf_np = cf.cpu().detach().numpy()
                loss_np = loss_per_instance.cpu().detach().numpy()
                for i in range(N):
                    if valid_mask[i]:
                        candidate_counterfactuals[i].append(cf_np[i])
                        candidate_distances[i].append(loss_np[i])

            loss_per_instance.sum().backward()
            optim.step()
            optim.zero_grad()

        # Choose the nearest counterfactual per instance
        list_cfs = []
        for i in range(N):
            if candidate_counterfactuals[i]:
                if verbose:
                    print(f"Counterfactual found for instance {i}!")
                array_distances = np.array(candidate_distances[i])
                index = np.argmin(array_distances)
                list_cfs.append(candidate_counterfactuals[i][index])
            else:
                if verbose:
                    print(f"No counterfactual found for instance {i}")
                list_cfs.append([])
        return list_cfs

    def _compute_loss(self, cf_initialize, query_instance, target):
        """Compute per-instance loss. Returns shape [N]."""
        loss_function = nn.BCEWithLogitsLoss(reduction="none")
        output = self._mlmodel(cf_initialize)

        # classification loss: average over classes per instance -> [N]
        loss1 = loss_function(output, target).mean(dim=1)

        # distance loss: L1 norm per instance -> [N]
        N = cf_initialize.shape[0]
        loss2 = (cf_initialize - query_instance).abs().view(N, -1).sum(dim=1)

        return loss1 + self._lambda * loss2
