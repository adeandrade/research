import torch
from torch import Tensor
from torch.nn import functional


def calculate_psnr(predictions: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    psnr = torch.round(255 * predictions)
    psnr = torch.clip(psnr, 0, 255)
    psnr = torch.clip(torch.round(255 * targets), 0, 255) - psnr
    psnr = psnr * mask if mask is not None else psnr
    psnr = torch.mean(psnr**2, dim=(1, 2, 3))
    psnr = torch.sqrt(psnr)
    psnr = 20 * torch.log10(255 / psnr)
    return torch.mean(psnr)


def calculate_rmse(predictions: Tensor, targets: Tensor, mask: Tensor | None = None, scale: float = 255.0) -> Tensor:
    rmse = functional.mse_loss(predictions, targets, reduction='none')
    rmse = rmse * mask if mask is not None else rmse
    rmse = torch.mean(rmse, dim=(1, 2, 3))
    rmse = torch.sqrt(rmse)
    rmse = torch.mean(rmse)
    return scale * rmse


def checkboard_mask(inputs: Tensor) -> tuple[Tensor, Tensor]:
    batch_size, _, height, width = inputs.shape

    mask = torch.zeros((height, width), dtype=inputs.dtype, device=inputs.device)
    mask[0::2, 1::2] = 1
    mask[1::2, 0::2] = 1
    mask = mask[None, None]

    mask_broadcasted = torch.tile(mask, dims=(batch_size, 1, 1, 1))

    inputs_with_mask = torch.masked_fill(inputs, mask.bool(), 0)
    inputs_with_mask = torch.concat((inputs_with_mask, mask_broadcasted), dim=1)

    return inputs_with_mask, mask


def round_toward_zero(inputs: Tensor) -> Tensor:
    return torch.copysign(torch.ceil(torch.abs(inputs) - 0.5), inputs)


def quantize(inputs: Tensor) -> Tensor:
    return torch.detach(round_toward_zero(inputs) - inputs) + inputs
