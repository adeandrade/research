import math
from gzip import GzipFile

import defopt
import sfu_torch_lib.io as io
import torch
from torch import Tensor

import common_information.prepare_discrete_dataset as pdd


def caculate_distortions_joint(
    num_symbols: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    offset = (num_symbols - 1) / 2

    distortions = torch.empty((num_symbols, num_symbols, num_symbols, num_symbols), dtype=dtype, device=device)

    for a in range(num_symbols):
        for b in range(num_symbols):
            for x in range(num_symbols):
                for y in range(num_symbols):
                    distortions[a, b, x, y] = ((a - x) ** 2 + (b - y) ** 2) / (offset**2)

    return distortions


def calculate_rate(source: Tensor, target: Tensor, conditional: Tensor) -> float:
    rate = conditional / target[*[None] * target.ndim]
    rate = source * conditional * torch.log2(rate)
    rate = torch.sum(rate)
    rate = rate.item()

    return rate


def calculate_distortions_marginal(
    num_symbols: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    offset = (num_symbols - 1) / 2

    distortions = torch.arange(num_symbols, dtype=dtype, device=device)
    distortions = distortions / offset
    distortions = distortions[:, None] - distortions[None, :]
    distortions = torch.square(distortions)

    return distortions


def blahut_arimoto(pmf: Tensor, distortions: Tensor, alpha: float, num_iterations: int) -> tuple[Tensor, Tensor]:
    """
    Blahut–Arimoto algorithm.
    """
    target_dims = list(range(pmf.ndim, 2 * pmf.ndim))
    source_dims = list(range(0, pmf.ndim))

    factors = torch.exp(-1 * alpha * distortions)

    pmf_conditional, pmf_target = None, pmf

    for _ in range(num_iterations):
        pmf_conditional = pmf_target[*[None] * pmf.ndim] * factors
        pmf_conditional = pmf_conditional / torch.sum(pmf_conditional, target_dims, keepdim=True)

        pmf_target = pmf[..., *[None] * pmf.ndim] * pmf_conditional
        pmf_target = torch.sum(pmf_target, source_dims)

    assert pmf_conditional is not None

    return pmf_conditional, pmf_target


def calculate_rate_distortion_pmf(pmf: Tensor, alpha: float, num_iterations: int) -> tuple[Tensor, tuple[float, float]]:
    num_symbols_a, num_symbols_b = pmf.shape

    assert num_symbols_a == num_symbols_b

    distortions = caculate_distortions_joint(num_symbols_a, pmf.dtype, pmf.device)

    pmf_conditional, _ = blahut_arimoto(pmf, distortions, alpha, num_iterations)

    pmf_joint = pmf[:, :, None, None] * pmf_conditional

    distortion_joint = pmf_joint * distortions
    distortion_joint = torch.sum(distortion_joint)
    distortion_joint = distortion_joint.item()

    distortion_marginal_a = torch.sum(pmf_joint, dim=(1, 3))
    distortion_marginal_a = distortion_marginal_a * calculate_distortions_marginal(num_symbols_a, pmf.dtype, pmf.device)
    distortion_marginal_a = torch.sum(distortion_marginal_a)
    distortion_marginal_a = distortion_marginal_a.item()

    return pmf_joint, (distortion_joint, distortion_marginal_a)


def aggregate_tensor(dependencies: Tensor, value_marginal: float, value_joint: float, shape: tuple[int, int]) -> float:
    split_num_dimensions = shape[0] * shape[1]

    aggregated = torch.where(dependencies == -1, value_marginal, value_joint / 2)
    aggregated = torch.sum(aggregated)
    aggregated = aggregated / split_num_dimensions

    return aggregated.item()


def calculate_safe_entropy(a: Tensor, b: Tensor) -> Tensor:
    return torch.sum(torch.where(torch.logical_or(a == 0, b == 0), 0, a * torch.log2(a / b)))


def calculate_entropies(pmf: Tensor, dependencies: Tensor, shape: tuple[int, int]) -> tuple[float, float, float]:
    marginals = torch.sum(pmf, dim=(0, 1), keepdim=True) * torch.sum(pmf, dim=(2, 3), keepdim=True)

    entropy_joint = calculate_safe_entropy(pmf, marginals)
    entropy_joint = entropy_joint.item()

    joint_a = torch.sum(pmf, dim=(1, 3), keepdim=True)
    marginals_a = torch.sum(pmf, dim=(1, 2, 3), keepdim=True) * torch.sum(pmf, dim=(0, 1, 3), keepdim=True)

    entropy_marginal = calculate_safe_entropy(joint_a, marginals_a)
    entropy_marginal = entropy_marginal.item()

    entropies = pdd.calculate_entropies(dependencies, entropy_joint, entropy_marginal, shape)

    return entropies


def calculate_distortion(
    dependencies: Tensor,
    distortion_marginal_a: float,
    distortion_joint: float,
    shape: tuple[int, int],
) -> float:
    distortion = aggregate_tensor(dependencies, distortion_marginal_a, distortion_joint, shape)
    distortion = math.sqrt(distortion)

    return distortion


def calculate_rate_distortion(
    dataset_path: str,
    *,
    alphas: tuple[float, ...] = (0.1, 2.0, 5.0, 7.0, 100.0),
    num_iterations: int = 1000,
) -> None:
    dataset_path = io.localize_dataset(dataset_path)

    data = torch.load(GzipFile(dataset_path), map_location=torch.device('cuda:0'))  # type: ignore

    shape, dependencies, pmf = data['shape'], data['dependencies'], data['pmf']

    rate_distortions = [calculate_rate_distortion_pmf(pmf, alpha, num_iterations) for alpha in alphas]

    rates = [calculate_entropies(pmf, dependencies, shape) for pmf, *_ in rate_distortions]

    distortions = [
        calculate_distortion(dependencies, distortion_marginal_a, distortion_joint, shape)
        for *_, (distortion_joint, distortion_marginal_a) in rate_distortions
    ]

    rate_distortions = [list(zip(rate, distortions)) for rate in zip(*rates)]

    for label, rate_distortion in zip(('Joint', 'Marginal', 'Mutual Information'), rate_distortions):
        print(f'{label}:', rate_distortion)


def main():
    defopt.run(calculate_rate_distortion)


if __name__ == '__main__':
    main()
