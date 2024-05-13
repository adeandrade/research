from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from numpy.typing import ArrayLike
from torch import Size, Tensor


def round_toward_zero(inputs: Tensor) -> Tensor:
    return torch.copysign(torch.ceil(torch.abs(inputs) - 0.5), inputs)


def quantize(inputs: Tensor) -> Tensor:
    return torch.detach(round_toward_zero(inputs) - inputs) + inputs


def mse_loss(x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    return torch.sum(torch.masked_fill(torch.square(x - y), torch.logical_not(mask), value=0)) / torch.sum(mask)


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


def disentangle_tuple(tensor: Tensor) -> tuple[Tensor, Tensor]:
    return tensor[:, ::2], tensor[:, 1::2]


def to_numpy(tensor):
    return tensor.numpy(force=True)


def calculate_cdf_indices(scales: Tensor, table: Tensor, lower_bound_value: float) -> Tensor:
    scales = torch.maximum(scales, torch.tensor(lower_bound_value))

    original_shape = scales.shape

    indices = torch.reshape(scales, shape=(-1,))
    indices = indices[:, None] <= table[:-1][None, :]
    indices = torch.sum(indices, dim=1, dtype=torch.int32)
    indices = torch.full_like(indices, len(table) - 1, dtype=torch.int32) - indices
    indices = torch.reshape(indices, original_shape)

    return indices


def calculate_reconstruction_loss(
    predictions: Tensor,
    targets: Tensor,
    masks: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    rmse = functional.mse_loss(predictions, targets, reduction='none')
    rmse = rmse * masks[:, None, :, :] if masks is not None else rmse
    rmse = torch.reshape(rmse, shape=(rmse.shape[0], -1))
    rmse = torch.mean(rmse, dim=1)
    rmse = torch.sqrt(rmse)

    rmse_scaled = 255 * torch.mean(rmse)
    psnr = torch.mean(20 * torch.log10(1 / rmse))

    return rmse_scaled, psnr


def bj_delta(rate_1: ArrayLike, psnr_1: ArrayLike, rate_2: ArrayLike, psnr_2: ArrayLike, mode: int = 1) -> float:
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


def calculate_bpp(entropy_channels: Tensor, num_pixels: Tensor | float) -> Tensor:
    dimensions = list(range(1, entropy_channels.ndim))

    bpp = torch.sum(entropy_channels, dimensions)
    bpp = bpp / num_pixels
    bpp = torch.mean(bpp)

    return bpp


def calculate_bpp_fixed(entropy_channels: Tensor, sizes: Size) -> Tensor:
    num_pixels = np.prod(sizes[2:]).item()

    bpp = calculate_bpp(entropy_channels, num_pixels)

    return bpp


def calculate_num_pixels(sizes: Sequence[tuple[int, int]], dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor([height * width for height, width in sizes], dtype=dtype, device=device)


def calculate_bpp_dynamic(entropy_channels: Tensor, sizes: Sequence[tuple[int, int]]) -> Tensor:
    batch_size, *_ = entropy_channels.shape
    assert batch_size == len(sizes)

    num_pixels = calculate_num_pixels(sizes, entropy_channels.dtype, entropy_channels.device)
    bpp = calculate_bpp(entropy_channels, num_pixels)

    return bpp


def calculate_likelihood_bpp_fixed(likelihoods: Tensor, sizes: Size) -> Tensor:
    bits = calculate_likelihood_entropy(likelihoods, reduce_channels=False, reduce_samples=False, normalize=False)

    bpp = calculate_bpp_fixed(bits, sizes)

    return bpp


def calculate_likelihood_bpp_dynamic(likelihoods: Tensor, sizes: Sequence[tuple[int, int]]) -> Tensor:
    bits = calculate_likelihood_entropy(likelihoods, reduce_channels=False, reduce_samples=False, normalize=False)

    bpp = calculate_bpp_dynamic(bits, sizes)

    return bpp
