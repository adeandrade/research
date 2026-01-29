import os
import random

import defopt
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal


def create_pmf_hard(num_symbols: int, rho: float) -> Tensor:
    pmf = torch.rand([num_symbols + 1, num_symbols + 1], dtype=torch.float32)
    pmf = pmf.T @ pmf

    diagonal = rho * torch.sum(pmf, dim=1)
    dominant = torch.diag_embed(diagonal)

    mask = torch.ones_like(diagonal)
    mask = 1 - torch.diag_embed(mask)

    pmf = dominant + mask * pmf
    pmf = pmf / torch.sum(pmf)

    return pmf


def create_pmf_easy(num_symbols: int, rho: float, num_samples: int = 100000) -> Tensor:
    offset = num_symbols / 2

    means = torch.tensor([0, 0], dtype=torch.float32)
    covariances = offset * torch.tensor([[1, rho], [rho, 1]], dtype=torch.float32)

    distribution = MultivariateNormal(means, covariances)

    samples = distribution.sample([num_samples])
    samples = samples + offset
    samples = torch.round(samples).to(torch.int64)
    samples = torch.clamp(samples, 0, num_symbols)

    pmf = torch.ones([num_symbols + 1, num_symbols + 1], dtype=torch.float32)

    for x, y in samples:
        pmf[x, y] += 1

    pmf = pmf / torch.sum(pmf)

    return pmf


def create_random_dependencies(num_dimensions: int, independent_probability: float) -> Tensor:
    indices = list(range(num_dimensions))
    random.shuffle(indices)

    dependencies = torch.full([num_dimensions], -1, dtype=torch.int64)

    index = 0

    while index < len(indices):
        if index < len(indices) - 1 and random.uniform(0, 1) > independent_probability:
            index_a, index_b = indices[index], indices[index + 1]

            dependencies[index_a] = index_b
            dependencies[index_b] = index_a

            index += 2

        else:
            index += 1

    return dependencies


def calculate_entropy_pmf(pmf: Tensor) -> float:
    entropy = torch.where(pmf == 0, 0, pmf * torch.log2(pmf))
    entropy = -1 * torch.sum(entropy)

    return entropy.item()


def calculate_entropies(
    dependencies: Tensor,
    entropy_joint_pmf: float,
    entropy_marginal_pmf: float,
    shape: tuple[int, int],
) -> tuple[float, float, float]:
    num_dimensions = torch.numel(dependencies)
    split_num_dimensions = shape[0] * shape[1]
    split_index = num_dimensions // 2

    entropy_joint = torch.where(dependencies == -1, entropy_marginal_pmf, entropy_joint_pmf / 2)
    entropy_joint = torch.sum(entropy_joint)
    entropy_joint = entropy_joint.item() / split_num_dimensions

    dependencies_a = dependencies[:split_index]
    dependencies_a = torch.logical_or(dependencies_a == -1, dependencies_a >= split_index)
    entropy_marginal_a = torch.where(dependencies_a, entropy_marginal_pmf, entropy_joint_pmf / 2)
    entropy_marginal_a = torch.sum(entropy_marginal_a)
    entropy_marginal_a = entropy_marginal_a.item() / split_num_dimensions

    dependencies_b = dependencies[split_index:]
    dependencies_b = dependencies_b < split_index
    entropy_marginal_b = torch.where(dependencies_b, entropy_marginal_pmf, entropy_joint_pmf / 2)
    entropy_marginal_b = torch.sum(entropy_marginal_b)
    entropy_marginal_b = entropy_marginal_b.item() / split_num_dimensions

    entropy_marginal = entropy_marginal_a + entropy_marginal_b
    mutual_information = entropy_marginal - entropy_joint

    return entropy_joint, entropy_marginal, mutual_information


def create_random_transform(num_dimensions: int, scale: float, threshold: float) -> tuple[Tensor, float]:
    transform = scale * torch.randn((num_dimensions, num_dimensions), dtype=torch.float32)
    transform = torch.clamp(transform, -threshold, threshold)

    diagonal = torch.sum(torch.abs(transform), dim=1)
    dominant = torch.diag_embed(diagonal)

    mask = torch.ones_like(diagonal)
    mask = 1 - torch.diag_embed(mask)

    transform = dominant + mask * transform
    transform = torch.matrix_exp(transform)
    transform = transform * torch.exp(-1 * torch.sum(diagonal) / num_dimensions)

    log_determinant = torch.logdet(transform)

    return transform, log_determinant.item()


def create(
    path: str,
    *,
    shape: tuple[int, int] = (64, 64),
    rho: float = 1.0,
    num_symbols: int = 8,
    independent_probability: float = 0.2,
    transform_scale: float = 0.001,
    transform_threshold: float = 0.01,
    seed: int = 110069,
    filename: str = 'discrete_tensors.pkl',
    verbose: bool = False,
) -> None:
    num_dimensions = 2 * shape[0] * shape[1]
    split_num_dimensions = shape[0] * shape[1]

    random.seed(seed)
    torch.manual_seed(seed)

    pmf = create_pmf_easy(num_symbols, rho)

    dependencies = create_random_dependencies(num_dimensions, independent_probability)

    joint_entropy, marginal_entropy, mutual_information = calculate_entropies(
        dependencies,
        calculate_entropy_pmf(pmf),
        calculate_entropy_pmf(torch.sum(pmf, dim=1)),
        shape,
    )

    transform_a, log_determinant_transform_a = create_random_transform(
        split_num_dimensions,
        transform_scale,
        transform_threshold,
    )
    transform_b, log_determinant_transform_b = create_random_transform(
        split_num_dimensions,
        transform_scale,
        transform_threshold,
    )

    data = {
        'shape': shape,
        'rho': rho,
        'num_symbols': num_symbols,
        'independent_probability': independent_probability,
        'transform_scale': transform_scale,
        'transform_threshold': transform_threshold,
        'seed': seed,
        'dependencies': dependencies,
        'pmf': pmf,
        'transform_a': transform_a,
        'transform_b': transform_b,
        'joint_entropy': joint_entropy,
        'marginal_entropy': marginal_entropy,
        'mutual_information': mutual_information,
        'log_determinant_transform_a': log_determinant_transform_a,
        'log_determinant_transform_b': log_determinant_transform_b,
    }

    torch.save(data, os.path.join(path, filename))

    if verbose:
        print(f'Data: {data}')


def main():
    defopt.run(create)


if __name__ == '__main__':
    main()
