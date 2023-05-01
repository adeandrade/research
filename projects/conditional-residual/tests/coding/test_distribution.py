import compressai._CXX as cxx
import numpy as np
import scipy
import torch
from compressai.entropy_models import GaussianConditional

import coding.distribution as distribution
import coding.models.functions as functions
import coding.normal as normal
from coding.normal import SCALE_MIN, TAIL_MASS


def test_pdf_to_quantized_cdf():
    pdf = np.array(
        [3.58682644e-11, 1.49155039e-02, 9.70168992e-01, 1.49155039e-02, 3.58682644e-11, 1.75577230e-27],
        dtype=np.float32,
    )

    expected = cxx.pmf_to_quantized_cdf(pdf, 16)

    actual = distribution.pmf_to_quantized_cdf(pdf, 16)
    actual = actual.tolist()

    assert actual == expected


def test_pdf_to_quantized_cdf_multiple():
    tail_mass = 1e-9
    scale_table = normal.get_scale_table()

    multiplier = -scipy.stats.norm.ppf(tail_mass / 2)
    pmf_centers = np.ceil(scale_table * multiplier).astype(np.int32)
    pmf_sizes = 2 * pmf_centers + 1
    pmf_max_size = np.max(pmf_sizes)

    samples = np.arange(pmf_max_size, dtype=np.int32)
    samples = np.abs(samples - pmf_centers[:, None])
    samples = samples.astype(np.float32)

    upper = normal.calculate_standardized_cumulative((.5 - samples) / scale_table[:, None])
    lower = normal.calculate_standardized_cumulative((-.5 - samples) / scale_table[:, None])
    pmfs = upper - lower

    for pmf in pmfs:
        expected = cxx.pmf_to_quantized_cdf(pmf, 16)
        actual = distribution.pmf_to_quantized_cdf(pmf, 16)
        actual = actual.tolist()

        assert actual == expected


def test_pdfs_to_quantized_cdfs():
    scale_table = normal.get_scale_table()

    multiplier = -scipy.stats.norm.ppf(TAIL_MASS / 2)
    pmf_centers = np.ceil(scale_table * multiplier).astype(np.int32)
    pmf_sizes = 2 * pmf_centers + 1
    pmf_max_size = np.max(pmf_sizes)

    samples = np.arange(pmf_max_size, dtype=np.int32)
    samples = np.abs(samples - pmf_centers[:, None])
    samples = samples.astype(np.float32)

    upper = normal.calculate_standardized_cumulative((.5 - samples) / scale_table[:, None])
    lower = normal.calculate_standardized_cumulative((-.5 - samples) / scale_table[:, None])
    pmfs = upper - lower

    tail_mass = 2 * lower[:, 0]

    expected = GaussianConditional(scale_table=None)._pmf_to_cdf(
        pmf=torch.tensor(pmfs),
        tail_mass=torch.tensor(tail_mass[:, None]),
        pmf_length=torch.tensor(pmf_sizes),
        max_length=pmf_max_size,
    )

    actual = distribution.pmfs_to_quantized_cdfs(pmfs, pmf_sizes, tail_mass, pmf_max_size)

    for actual_cdf, expected_cdf in zip(expected, actual):
        assert actual_cdf.tolist() == expected_cdf.tolist()


def test_calculate_scale_index():
    scale = 0.09563492983579636

    scale_table = normal.get_scale_table()

    expected = functions.calculate_cdf_indices(
        scales=torch.tensor(scale),
        table=torch.tensor(scale_table),
        lower_bound_value=SCALE_MIN,
    )

    expected = expected.item()

    scale_table[0].item()
    torch.tensor(scale_table)[0].item()

    actual = distribution.calculate_cdf_index(scale, scale_table, SCALE_MIN)

    assert actual == expected


def test_calculate_scale_indices_numpy():
    scales = np.random.rand(10000000).astype(np.float32) * 256
    scale_table = normal.get_scale_table(SCALE_MIN)

    model = GaussianConditional(scale_table=list(scale_table), scale_bound=SCALE_MIN)
    expected = model.build_indexes(torch.tensor(scales))

    actual = distribution.calculate_cdf_indices(scales, scale_table, SCALE_MIN)

    assert actual.tolist() == expected.tolist()


def test_calculate_scale_indices_torch():
    scales_numpy = np.random.rand(10000000).astype(np.float32) * 256
    scales_torch = torch.tensor(scales_numpy)

    scale_table = normal.get_scale_table()

    expected = distribution.calculate_cdf_indices(scales_numpy, scale_table, SCALE_MIN)
    actual = functions.calculate_cdf_indices(scales_torch, torch.tensor(scale_table), SCALE_MIN)

    assert actual.tolist() == expected.tolist()
