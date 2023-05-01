from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor


def aggregate_losses(
        losses: Tensor,
        reduce_channels: bool = True,
        reduce_samples: bool = True,
        normalize: bool = True,
) -> Tensor:

    batch_size, num_channels, height, width = losses.shape
    num_elements = height * width

    losses = torch.reshape(losses, shape=(batch_size, num_channels, -1))
    losses = torch.sum(losses, dim=-1)

    if normalize:
        losses = losses / num_elements

    if reduce_channels:
        losses = torch.sum(losses, dim=1)

    if reduce_samples:
        losses = torch.mean(losses, dim=0)

    return losses


def calculate_likelihood_entropy(
        likelihoods: Tensor,
        reduce_channels: bool = True,
        reduce_samples: bool = True,
        normalize: bool = True,
) -> Tensor:

    return aggregate_losses(-torch.log2(likelihoods), reduce_channels, reduce_samples, normalize)


def calculate_reconstruction_loss(predictions: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
    rmse = functional.mse_loss(predictions, targets, reduction='none')
    rmse = torch.reshape(rmse, shape=(rmse.shape[0], -1))
    rmse = torch.mean(rmse, dim=1)
    rmse = torch.sqrt(rmse)

    rmse_scaled = 255 * torch.mean(rmse)
    psnr = torch.mean(20 * torch.log10(1 / rmse))

    return rmse_scaled, psnr


def bj_delta(rate_1: np.ndarray, psnr_1: np.ndarray, rate_2: np.ndarray, psnr_2: np.ndarray, mode: int = 0) -> float:
    log_rate_1 = np.log(rate_1)
    log_rate_2 = np.log(rate_2)

    # find integral
    if mode == 0:
        # least squares polynomial fit
        p1 = np.polyfit(log_rate_1, psnr_1, 3)
        p2 = np.polyfit(log_rate_2, psnr_2, 3)

        # integration interval
        min_int = max(min(log_rate_1), min(log_rate_2))
        max_int = min(max(log_rate_1), max(log_rate_2))

        # indefinite integral of both polynomial curves
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        # evaluates both poly curves at the limits of the integration interval
        # to find the area
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

        # find avg diff between the areas to obtain the final measure
        avg_diff = (int2 - int1) / (max_int - min_int)

    else:
        # rate method: sames as previous one but with inverse order
        p1 = np.polyfit(psnr_1, log_rate_1, 3)
        p2 = np.polyfit(psnr_2, log_rate_2, 3)

        # integration interval
        min_int = max(min(psnr_1), min(psnr_2))
        max_int = min(max(psnr_1), max(psnr_2))

        # indefinite interval of both polynomial curves
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        # evaluates both poly curves at the limits of the integration interval
        # to find the area
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

        # find avg diff between the areas to obtain the final measure
        avg_exp_diff = (int2 - int1) / (max_int - min_int)
        avg_diff = (np.exp(avg_exp_diff) - 1) * 100

    return avg_diff
