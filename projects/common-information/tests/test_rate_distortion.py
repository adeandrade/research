import math

import pytest
import torch

import common_information.calculate_rate_distortion as rd
import common_information.prepare_discrete_dataset as pdd


def calculate_rate_distortion(distortion: float, num_symbols: int) -> float:
    threshold = 1 - 1 / num_symbols

    if distortion <= threshold:
        entropy = pdd.calculate_entropy_pmf(torch.tensor([distortion, 1 - distortion]))

        rate = math.log2(num_symbols) - entropy - distortion * math.log2(num_symbols - 1)

    else:
        rate = 0

    return rate


def test_blahut_arimoto_1d(d: int = 2):
    pmf = torch.ones(d, dtype=torch.float32) / d

    distortions = 1 - torch.eye(d, dtype=torch.float32)

    for alpha in (0.01, 0.1, 2.0, 5.0, 7.0, 100.0):
        pmf_conditional, pmf_target = rd.blahut_arimoto(pmf, distortions, alpha, 1000)

        rate = rd.calculate_rate(pmf, pmf_target, pmf_conditional)

        distortion = pmf_conditional * pmf[:, None] * distortions
        distortion = torch.sum(distortion)
        distortion = distortion.item()

        rate_expected = calculate_rate_distortion(distortion, d)

        assert pytest.approx(rate_expected, 0.01) == rate


def test_blahut_arimoto_2d(d: int = 2):
    pmf = torch.ones((d, d), dtype=torch.float32) / d**2

    distortions = 1 - torch.eye(d**2, dtype=torch.float32)
    distortions = torch.reshape(distortions, (d, d, d, d))

    for alpha in (0.01, 0.1, 2.0, 5.0, 7.0, 100.0):
        pmf_conditional, pmf_target = rd.blahut_arimoto(pmf, distortions, alpha, 1000)

        rate = rd.calculate_rate(pmf, pmf_target, pmf_conditional)

        distortion = pmf_conditional * pmf[:, :, None, None] * distortions
        distortion = torch.sum(distortion)
        distortion = distortion.item()

        rate_expected = calculate_rate_distortion(distortion, d**2)

        assert pytest.approx(rate_expected, 0.01) == rate
