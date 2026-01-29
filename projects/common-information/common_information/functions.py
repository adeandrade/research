import math

import numpy as np
import scipy.interpolate
import torch
import torch.nn.functional as functional
from torch import Size, Tensor


def round_toward_zero(inputs: Tensor) -> Tensor:
    return torch.copysign(torch.ceil(torch.abs(inputs) - 0.5), inputs)


def round_away_zero(inputs: Tensor) -> Tensor:
    return torch.copysign(torch.floor(torch.abs(inputs) + 0.5), inputs)


def quantize(inputs: Tensor) -> Tensor:
    return torch.detach(round_toward_zero(inputs) - inputs) + inputs


def safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    return torch.where(denominator == 0, numerator, numerator / denominator)


def mse_loss(x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    loss = torch.masked_fill(torch.square(x - y), torch.logical_not(mask), value=0)
    loss = torch.sum(loss, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))
    loss = torch.mean(loss)

    return loss


def aggregate_losses(
    losses: Tensor,
    reduce_channels: bool = True,
    reduce_samples: bool = True,
    normalize: bool = True,
) -> Tensor:
    if normalize:
        losses = torch.mean(losses, dim=(2, 3))
    else:
        losses = torch.sum(losses, dim=(2, 3))

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


def calculate_reconstruction_loss(
    predictions: Tensor,
    targets: Tensor,
    masks: Tensor | None = None,
    scale: float = 255.0,
) -> Tensor:
    rmse = functional.mse_loss(predictions, targets, reduction='none')
    rmse = rmse * masks[:, None, :, :] if masks is not None else rmse
    rmse = torch.mean(rmse, dim=(1, 2, 3))
    rmse = torch.sqrt(rmse)
    rmse = torch.mean(rmse)
    rmse = scale * rmse

    return rmse


def bd_rate(
    rate_1: np.ndarray,
    psnr_1: np.ndarray,
    rate_2: np.ndarray,
    psnr_2: np.ndarray,
    piecewise: bool = False,
) -> float:
    """
    Calculates the Bjontegaard Delta Rate (BD-Rate) savings percentage between two encoding methods.
    Returns the percentage of bitrate savings of encoding method 2 relative to encoding method 1.

    :param rate_1: Bitrates for encoding method 1.
    :param psnr_1: Peak Signal-to-Noise Ratios (PSNR) for encoding method 1.
    :param rate_2: Bitrates for encoding method 2.
    :param psnr_2: Peak Signal-to-Noise Ratios (PSNR) for encoding method 2.
    :param piecewise: Whether to use piecewise calculation or not. Default is False.
    """
    # take the logarithm of the bitrates
    log_rate_1 = np.log(rate_1)
    log_rate_2 = np.log(rate_2)

    # determine the integration interval
    min_int = max(min(psnr_1), min(psnr_2))
    max_int = min(max(psnr_1), max(psnr_2))

    # compute the integrals
    if piecewise:
        # use piecewise interpolation and the trapezoidal rule to compute the integrals
        samples, interval = np.linspace(min_int, max_int, num=100, retstep=True)

        v1 = scipy.interpolate.pchip_interpolate(
            np.sort(psnr_1),
            np.sort(log_rate_1),
            samples,
        )
        v2 = scipy.interpolate.pchip_interpolate(
            np.sort(psnr_2),
            np.sort(log_rate_2),
            samples,
        )

        int_1 = np.trapezoid(v1, dx=interval)
        int_2 = np.trapezoid(v2, dx=interval)

    else:
        # fit polynomials to the relationship between PSNR and log(bitrate)
        polynomial_1 = np.polyfit(psnr_1, log_rate_1, deg=3)
        polynomial_2 = np.polyfit(psnr_2, log_rate_2, deg=3)

        # use polynomial integration to compute the integrals
        p_int_1 = np.polyint(polynomial_1)
        p_int_2 = np.polyint(polynomial_2)

        int_1 = np.polyval(p_int_1, max_int) - np.polyval(p_int_1, min_int)
        int_2 = np.polyval(p_int_2, max_int) - np.polyval(p_int_2, min_int)

    # compute the average difference and return the result
    avg_diff = (int_2 - int_1) / (max_int - min_int)
    avg_diff = (np.exp(avg_diff) - 1) * 100

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


def calculate_num_pixels(sizes: list[tuple[int, int]], dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor([height * width for height, width in sizes], dtype=dtype, device=device)


def calculate_bpp_dynamic(entropy_channels: Tensor, sizes: list[tuple[int, int]]) -> Tensor:
    batch_size, *_ = entropy_channels.shape
    assert batch_size == len(sizes)

    num_pixels = calculate_num_pixels(sizes, entropy_channels.dtype, entropy_channels.device)
    bpp = calculate_bpp(entropy_channels, num_pixels)

    return bpp


def calculate_likelihood_bpp_fixed(likelihoods: Tensor, sizes: Size) -> Tensor:
    bits = calculate_likelihood_entropy(likelihoods, reduce_channels=False, reduce_samples=False, normalize=False)

    bpp = calculate_bpp_fixed(bits, sizes)

    return bpp


def calculate_likelihood_bpp_dynamic(likelihoods: Tensor, sizes: list[tuple[int, int]]) -> Tensor:
    bits = calculate_likelihood_entropy(likelihoods, reduce_channels=False, reduce_samples=False, normalize=False)

    bpp = calculate_bpp_dynamic(bits, sizes)

    return bpp


def calculate_likelihood_bpp_masked(likelihoods: Tensor, masks: Tensor, downsample_factor: int) -> Tensor:
    sizes = masks_to_sizes(masks)

    _, _, height, width = likelihoods.shape
    masks = masks[:, :height, :width]
    masks = masks.to(torch.bool)
    masks = torch.clone(masks)

    for index, (height, width) in enumerate(sizes):
        height = math.ceil(height / downsample_factor)
        width = math.ceil(width / downsample_factor)

        masks[index, height:, :] = False
        masks[index, :, width:] = False

    masks = torch.logical_not(masks)

    bpp = torch.masked_fill(likelihoods, masks[:, None, :, :], 1.0)
    bpp = calculate_likelihood_bpp_dynamic(bpp, sizes)

    return bpp


def combine_and_mask(x: Tensor, y: Tensor) -> Tensor:
    mask = (x - y) == 0

    common_combined = (x + y) / 2
    common_combined = torch.where(mask, common_combined, 0)

    return common_combined


def combine_and_mask_complex(x: Tensor, y: Tensor) -> Tensor:
    squared_error = (x - y) ** 2
    summation = x + y

    mask = torch.clamp(torch.abs(summation), min=1)
    mask = torch.log(mask)
    mask = safe_divide(mask, squared_error)
    mask = torch.detach(mask)
    mask = torch.exp(-1 * squared_error * mask)
    mask = torch.where(squared_error >= 0.25, mask, 1)

    common_combined = mask * summation / 2
    common_combined = quantize(common_combined)

    return common_combined


def combine_and_mask_simple(common_segmentation: Tensor, common_depth: Tensor, tau: float = 0.1) -> Tensor:
    mask = (common_segmentation - common_depth) ** 2 / tau
    mask = torch.exp(-mask)

    common_combined = (common_segmentation + common_depth) / 2
    common_combined = mask * common_combined
    common_combined = quantize(common_combined)

    return common_combined


def masks_to_sizes(masks) -> list[tuple[int, int]]:
    heights = torch.sum(masks[:, :, 0], dim=1)
    widths = torch.sum(masks[:, 0, :], dim=1)

    sizes = [(int(height.item()), int(width.item())) for height, width in zip(heights, widths)]

    return sizes


def tensor_to_images(images: Tensor, masks: Tensor) -> list[Tensor]:
    return [image[:, :height, :width] for image, (height, width) in zip(images, masks_to_sizes(masks))]
