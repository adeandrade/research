import math
from typing import Any

import numpy as np
import scipy
import seaborn
import torch
from matplotlib import pyplot
from pandas import DataFrame
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch.nn import functional


class LowerBound(Function):
    @staticmethod
    def forward(inputs: Tensor, minimums: Tensor) -> Tensor:
        return torch.maximum(inputs, minimums)

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor], output: Tensor) -> None:
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor, None]:
        inputs, minimums = ctx.saved_tensors  # type: ignore

        keep_grads_if = (inputs >= minimums) | (grad_output < 0)

        grad_output = keep_grads_if * grad_output

        return grad_output, None


def lower_bound(inputs: Tensor, minimums: Tensor) -> Tensor:
    return LowerBound.apply(inputs, minimums)  # type: ignore


@torch.compile
def round_toward_zero(inputs: Tensor) -> Tensor:
    return torch.copysign(torch.ceil(torch.abs(inputs) - 0.5), inputs)


@torch.compile
def round_away_zero(inputs: Tensor) -> Tensor:
    return torch.copysign(torch.floor(torch.abs(inputs) + 0.5), inputs)


@torch.compile
def round_half_down(inputs: Tensor) -> Tensor:
    return torch.ceil(inputs - 0.5)


@torch.compile
def round_half_up(inputs: Tensor) -> Tensor:
    return torch.floor(inputs + 0.5)


@torch.compile
def quantize(inputs: Tensor) -> Tensor:
    return torch.detach(torch.round(inputs) - inputs) + inputs


def safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    return torch.where(denominator == 0, numerator, numerator / denominator)


def scale(inputs: Tensor, scales: Tensor) -> Tensor:
    shape = [1, -1] + [1] * (inputs.ndim - 2)

    scales = torch.reshape(scales, shape)

    return inputs * scales


def scale_quantize(inputs: Tensor, scales: Tensor) -> Tensor:
    outputs = scale(inputs, scales)
    return quantize(outputs)


def descale(inputs: Tensor, scales: Tensor) -> Tensor:
    shape = [1, -1] + [1] * (inputs.ndim - 2)

    scales = torch.reshape(scales, shape)
    scales = torch.detach(scales)

    return safe_divide(inputs, scales)


def scale_quantize_descale(inputs: Tensor, scales: Tensor) -> Tensor:
    outputs = scale_quantize(inputs, scales)
    return descale(outputs, scales)


def aggregate_losses(
    losses: Tensor,
    reduce_channels: bool = True,
    reduce_samples: bool = True,
    normalize: bool = True,
) -> Tensor:
    spatial_dimensions = list(range(2, losses.ndim))

    if normalize:
        losses = torch.mean(losses, dim=spatial_dimensions)
    else:
        losses = torch.sum(losses, dim=spatial_dimensions)

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


def calculate_bpe(entropy_channels: Tensor, num_elements: Tensor | float) -> Tensor:
    dimensions = list(range(1, entropy_channels.ndim))

    bpe = torch.sum(entropy_channels, dimensions)
    bpe = bpe / num_elements
    return torch.mean(bpe)


def calculate_standardized_cumulative(z_scores: Tensor) -> Tensor:
    return 0.5 * (1 + torch.erf(z_scores / math.sqrt(2)))


def calculate_standardized_probability(z_scores: Tensor) -> Tensor:
    return torch.exp(-0.5 * z_scores**2) * (2 * math.pi) ** -0.5


def calculate_standardized_log_probability(z_scores: Tensor) -> Tensor:
    return -0.5 * (z_scores**2 + math.log(2 * math.pi))


def calculate_normal_log_probability(inputs: Tensor, means: Tensor, stds: Tensor) -> Tensor:
    z_scores = (inputs - means) / stds

    return -0.5 * (z_scores**2 + 2 * torch.log(stds) + math.log(2 * math.pi))


def to_numpy(*tensors: Tensor) -> Any:
    outputs = [tensor.numpy(force=True) for tensor in tensors]

    if len(outputs) == 0:
        return None
    elif len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


def calculate_temperature(
    step: int,
    training: bool,
    tangent: float = 0.0015,
    offset: float = 1,
    maximum: float = 16,
) -> Tensor:
    return torch.tensor(1 / min(maximum, step * tangent + offset) if training else 0, dtype=torch.float32)


def soft_round(inputs: Tensor, temperature: Tensor | float | None = None) -> Tensor:
    """
    Differentiable approximation to `torch.round`.

    Lower temperatures correspond to closer approximations of the round function.
    For temperatures approaching infinity, this function resembles the identity.

    This function is described in Sec. 4.1 of the paper "Universally Quantized Neural Compression"
    Eirikur Agustsson & Lucas Theis https://arxiv.org/abs/2006.09952

    The temperature argument is the reciprocal of `alpha` in the paper.

    For convenience, we support `temperature = None`, which is the same as `temperature = inf`,
    which is the same as identity.

    :param inputs: Inputs to the function.
    :param temperature: Float Tensor >= 0. Controls smoothness of the approximation.

    Returns: Tensor of same shape as `inputs`.
    """
    temperature = torch.tensor(torch.inf if temperature is None else temperature, dtype=torch.float32)

    def _soft_round(x, t):
        m = torch.floor(x) + 0.5
        z = 2 * torch.tanh(0.5 / t)
        r = torch.tanh((x - m) * z) * t

        return m + r

    return torch.where(
        temperature < 1e-4,
        quantize(inputs),
        torch.where(
            temperature > 1e4,
            inputs,
            _soft_round(
                inputs,
                torch.clamp(temperature, 1e-4, 1e4),
            ),
        ),
    )


def soft_round_inverse(inputs: Tensor, temperature: Tensor | float | None = None) -> Tensor:
    """
    Inverse of `soft_round`.

    This function is described in Sec. 4.1 of the paper "Universally Quantized Neural Compression"
    Eirikur Agustsson & Lucas Theis https://arxiv.org/abs/2006.09952

    The temperature argument is the reciprocal of `alpha` in the paper.

    For convenience, we support `temperature = None`, which is the same as `temperature = inf`,
    which is the same as identity.

    :param inputs: Inputs to the function.
    :param temperature: Float Tensor >= 0. Controls smoothness of the approximation.

    Returns: Tensor of same shape as `inputs`.
    """
    temperature = torch.as_tensor(torch.inf if temperature is None else temperature)

    def _soft_round_inverse(x, t):
        m = torch.floor(x) + 0.5
        z = 2 * torch.tanh(0.5 / t)
        r = torch.arctanh((x - m) * z) * t

        return m + r

    return torch.where(
        temperature < 1e-4,
        quantize(inputs) + 0.5,
        torch.where(
            temperature > 1e4,
            inputs,
            _soft_round_inverse(
                inputs,
                torch.clamp(temperature, 1e-4, 1e4),
            ),
        ),
    )


@torch.compiler.disable()
def autocorrelate(sequence: Tensor) -> tuple[Tensor, Tensor]:
    """
    Computes batched complex-valued autocorrelation.

    :param sequence: Tensor of shape `(batch, length)`,
      where `batch` is the number of independent sequences to correlate with themselves,
      and `length` is the length of each sequence.

    Returns:
      Tensor of shape `(..., length)`. The right half of each autocorrelation sequence.
      The left half is redundant due to symmetry (even for the real part, odd for the imaginary part).
    """
    sequence = torch.view_as_complex(sequence)

    *_, length = sequence.shape

    flattened = torch.reshape(sequence, shape=(-1, length))

    batch_size, *_ = flattened.shape

    weights = torch.conj(flattened[:, None, :])
    inputs = functional.pad(flattened[None, :, :], pad=(0, length - 1))

    outputs = functional.conv1d(inputs, weights, padding=0, groups=batch_size)
    outputs = torch.reshape(outputs[0], sequence.shape)

    return outputs.real, outputs.imag


def periodic_probability(coefficients: Tensor, x: Tensor) -> Tensor:
    """
    Evaluates PDF or difference of CDFs of a periodic Fourier basis density.
    This function assumes the model is periodic with period 2.

    :param coefficients: Coefficients of Fourier series.
    :param x: Locations to evaluate the PDF.

    Returns: `p(x)`, where `p` is the PDF.
    """
    *_, num_frequencies, _ = coefficients.shape

    # autocorrelate coefficients to ensure a non-negative density.
    real, imaginary = autocorrelate(coefficients)

    # the DC coefficient is special: it is the normalizer of the density.
    dc = real[..., 0]
    ac_real = real[..., 1:]
    ac_imaginary = imaginary[..., 1:]

    pi_n = torch.pi * torch.arange(1, num_frequencies, dtype=x.dtype, device=x.device)
    pi_n_x = pi_n * x[..., None]

    # we can take the real part below because the coefficient sequence is
    # assumed to have Hermitian symmetry, so the Fourier series is always real.
    pdfs = torch.sum(ac_real * torch.cos(pi_n_x), dim=-1)
    pdfs = pdfs - torch.sum(ac_imaginary * torch.sin(pi_n_x), dim=-1)
    return pdfs / dc + 0.5


def periodic_probability_discrete(coefficients: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """
    Evaluates PDF or difference of CDFs of a periodic Fourier basis density.
    This function assumes the model is periodic with period 2.

    :param coefficients: Coefficients of Fourier series.
    :param lower: lower location of CDF.
    :param upper: upper location of CDF.

    Returns: P(upper) - P(lower)`, where `P` is the CDF.
    """
    *_, num_frequencies, _ = coefficients.shape

    # autocorrelate coefficients to ensure a non-negative density.
    real, imaginary = autocorrelate(coefficients)

    # the DC coefficient is special: it is the normalizer of the density.
    dc = real[..., 0]
    ac_real = real[..., 1:]
    ac_imaginary = imaginary[..., 1:]

    pi_n = torch.pi * torch.arange(1, num_frequencies, dtype=lower.dtype, device=lower.device)
    pi_n_lower = pi_n * lower[..., None]
    pi_n_upper = pi_n * upper[..., None]

    cos_diff = torch.cos(pi_n_upper) - torch.cos(pi_n_lower)
    sin_diff = torch.sin(pi_n_upper) - torch.sin(pi_n_lower)

    # we can take the real part below because the coefficient sequence is
    # assumed to have Hermitian symmetry, so the Fourier series is always real.
    pdfs = torch.sum(ac_real / pi_n * sin_diff, dim=-1)
    pdfs = pdfs + torch.sum(ac_imaginary / pi_n * cos_diff, dim=-1)
    return pdfs / dc + (upper - lower) / 2


def periodic_cummulative_probability(coefficients: Tensor, x: Tensor) -> Tensor:
    """
    Evaluates PDF or difference of CDFs of a periodic Fourier basis density.
    This function assumes the model is periodic with period 2.

    :param coefficients: Coefficients of Fourier series.
    :param lower: lower location of CDF.
    :param upper: upper location of CDF.

    Returns: P(upper) - P(lower)`, where `P` is the CDF.
    """
    *_, num_frequencies, _ = coefficients.shape

    # autocorrelate coefficients to ensure a non-negative density.
    real, imaginary = autocorrelate(coefficients)

    # the DC coefficient is special: it is the normalizer of the density.
    dc = real[..., 0]
    ac_real = real[..., 1:]
    ac_imaginary = imaginary[..., 1:]

    pi_n = torch.pi * torch.arange(1, num_frequencies, dtype=x.dtype, device=x.device)
    pi_n_x = pi_n * x[..., None]

    cos_diff = torch.cos(pi_n_x) - torch.cos(pi_n)

    # we can take the real part below because the coefficient sequence is
    # assumed to have Hermitian symmetry, so the Fourier series is always real.
    cdfs = torch.sum(ac_real / pi_n * torch.sin(pi_n_x), dim=-1)
    cdfs = cdfs + torch.sum(ac_imaginary / pi_n * cos_diff, dim=-1)
    return cdfs / dc + x / 2 + 0.5


def unnormalized_density_variation(coefficients: Tensor) -> Tensor:
    *_, num_frequencies, _ = coefficients.shape

    real, imaginary = autocorrelate(coefficients)

    ac_real = real[:, 1:]
    ac_imaginary = imaginary[:, 1:]

    value = ac_real**2 + ac_imaginary**2
    value = value * torch.arange(1, num_frequencies, dtype=value.dtype, device=value.device) ** 2
    value = torch.sum(value, dim=-1)
    value = torch.mean(value)
    return value * 4 * torch.pi**2


def plot_distribution(
    data: dict[str, tuple[Tensor, Tensor, Tensor]],
    path: str,
    num_columns: int = 4,
) -> None:
    num_plots = len(data)

    pyplot.clf()
    seaborn.set_theme(style='darkgrid')

    _, axs = pyplot.subplots(
        nrows=math.ceil(num_plots / num_columns),
        ncols=min(num_columns, num_plots),
        figsize=(11, 8),
        constrained_layout=True,
        squeeze=False,
    )

    for index_plot, (label, (points, pdfs, samples)) in enumerate(data.items()):
        index_row = index_plot // num_columns
        index_colum = index_plot % num_columns

        points = points.numpy(force=True)
        pdfs = pdfs.numpy(force=True)
        samples = samples.numpy(force=True)

        samples_dataframe = DataFrame({'Value': samples})

        ax = seaborn.histplot(samples_dataframe, x='Value', ax=axs[index_row][index_colum], stat='density')
        ax.plot(points, pdfs, color='red', label='PDF')
        ax.legend()
        ax.set_title(label)

    pyplot.savefig(path, dpi=300)


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
    return (np.exp(avg_diff) - 1) * 100


def split_list[T](input_list: list[T], separator: T) -> list[list[T]]:
    outer = []
    inner = []

    for element in input_list:
        if element == separator:
            if inner:
                outer.append(inner)

            inner = []

        else:
            inner.append(element)

    if inner:
        outer.append(inner)

    return outer
