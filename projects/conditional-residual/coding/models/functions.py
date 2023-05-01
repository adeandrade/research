from typing import Tuple, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx


class LowerBound(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, inputs: Tensor, minimums: Tensor) -> Tensor:
        ctx.save_for_backward(inputs, minimums)

        outputs = torch.maximum(inputs, minimums)

        return outputs

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, None]:
        inputs, minimums = ctx.saved_tensors

        keep_grads_if = (inputs >= minimums) | (grad_output < 0)

        grad_output = keep_grads_if * grad_output

        return grad_output, None


class Quantize(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, inputs: Tensor) -> Tensor:
        return round_toward_zero(inputs)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_outputs: Tensor) -> Tensor:
        return grad_outputs


def lower_bound(inputs: Tensor, minimums: Tensor) -> Tensor:
    return LowerBound.apply(inputs, minimums)


def perturb(inputs: Tensor, low: float = -.5, high: float = .5) -> Tensor:
    return inputs + torch.empty_like(inputs).uniform_(low, high)


def round_toward_zero(inputs: Tensor) -> Tensor:
    return torch.copysign(torch.ceil(torch.abs(inputs) - .5), inputs)


def quantize(inputs: Tensor) -> Tensor:
    return Quantize.apply(inputs)


def perturb_or_quantize(inputs: Tensor, training: bool) -> Tensor:
    return perturb(inputs) if training else quantize(inputs)


def disentangle(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    return tensor[:, ::2, :, :], tensor[:, 1::2, :, :]


def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.numpy(force=True)


def calculate_standardized_cumulative(standard_scores: Tensor) -> Tensor:
    return .5 * torch.erfc(-standard_scores * 2 ** -.5)


def create_triangular_mask(
        shape: Sequence[int],
        padding_height: int,
        padding_width: int,
        in_group_size: int = 1,
        out_group_size: int = 1,
        in_prefix_size: int = 0,
        out_prefix_size: int = 0,
        order: int = -1,
        include_target: bool = True,
        reverse: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
) -> Tensor:

    mask = torch.zeros(shape, dtype=dtype, device=device)

    mask[:, :in_prefix_size] = 1

    for index in range(shape[0] - out_prefix_size):
        causal_channels_end = index // out_group_size * in_group_size
        causal_channels_start = causal_channels_end - order * in_group_size if order >= 0 else 0
        group_channels_end = causal_channels_end + in_group_size

        index += out_prefix_size
        causal_channels_start += in_prefix_size
        causal_channels_end += in_prefix_size
        group_channels_end += in_prefix_size

        mask[index, causal_channels_start:causal_channels_end] = 1
        mask[index, causal_channels_end:group_channels_end, :padding_height] = 1
        mask[index, causal_channels_end:group_channels_end, padding_height, :padding_width + include_target] = 1

    if reverse:
        mask[:, in_prefix_size:] = torch.flip(mask[:, in_prefix_size:], dims=(1,))

    return mask


def calculate_unit_gaussian_likelihoods(errors: Tensor, scales: Tensor) -> Tensor:
    likelihoods = calculate_standardized_cumulative((.5 - errors) / scales)
    likelihoods = likelihoods - calculate_standardized_cumulative((-.5 - errors) / scales)

    return likelihoods


def calculate_gaussian_likelihoods(
        errors: Tensor,
        scales: Tensor,
        minimum_scales: float = 1e-6,
        minimum_likelihood: float = 0.,
) -> Tensor:

    scales = lower_bound(scales, torch.tensor(minimum_scales))

    likelihoods = calculate_unit_gaussian_likelihoods(errors, scales)
    likelihoods = lower_bound(likelihoods, torch.tensor(minimum_likelihood))

    return likelihoods


def calculate_cdf_indices(scales: Tensor, table: Tensor, lower_bound_value: float) -> Tensor:
    scales = torch.maximum(scales, torch.tensor(lower_bound_value))

    original_shape = scales.shape

    indices = torch.reshape(scales, shape=(-1,))
    indices = indices[:, None] <= table[:-1][None, :]
    indices = torch.sum(indices, dim=1, dtype=torch.int32)
    indices = torch.full_like(indices, len(table) - 1, dtype=torch.int32) - indices
    indices = torch.reshape(indices, original_shape)

    return indices


def make_divisible(value: float, divisor: int, min_value: Optional[int] = None, threshold: float = .9) -> int:
    """
    This function ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor

    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)

    # make sure that round down does not go down by less than the threshold
    if new_value < threshold * value:
        new_value += divisor

    return new_value
