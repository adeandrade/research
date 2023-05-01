import numpy as np
import pytest
import torch

import coding.autoregression as autoregression
import coding.distribution as distribution
import coding.frameworks.grouped as grouped
import coding.models.functions as functions
import coding.quantization as quantization
from coding.models.gaussian import GaussianEntropyModel


def test_causality_mask():
    shape = (3, 5, 5)

    num_channels, _, _ = shape

    sample = np.random.randint(low=-100, high=100, size=shape)
    sample = sample + np.random.rand(*shape)
    sample -= .5
    sample = sample.astype(np.float32)
    sample = torch.tensor(sample)

    model = GaussianEntropyModel(num_channels)

    expected = model.pdf_parameters(sample[None])
    means_expected, scales_expected = functions.disentangle(expected)

    sample[2:] = 0
    sample[1:, 2:] = 0
    sample[1:, 1:, 2:] = 0

    actual = model.pdf_parameters(sample[None])
    means_actual, scales_actual = functions.disentangle(actual)

    assert means_expected[0, 1, 1, 2] == means_actual[0, 1, 1, 2]
    assert scales_expected[0, 1, 1, 2] == scales_actual[0, 1, 1, 2]


def test_causality_autoregressive():
    shape = (4, 5, 5)

    num_channels, height, width = shape

    sample = np.random.randint(low=-100, high=100, size=shape)
    sample = sample + np.random.rand(*shape)
    sample -= .5
    sample = sample.astype(np.float32)

    model = GaussianEntropyModel(num_channels)
    model.update_coder_parameters()

    weights, _, _, _, group_size, _, _, _, _ = model.coder_parameters[0]

    padding = model.kernel_size // 2
    num_groups = num_channels // group_size

    inputs = autoregression.pad_2d(sample, padding, padding)
    _, expected_outputs = autoregression.initialize_network_outputs(model.coder_parameters, height, width)

    expected_mean, expected_scale = None, None

    h = w = 0

    for group_index in range(num_groups):
        for h in range(height):
            for w in range(width):
                expected_mean, expected_scale = grouped.compute_group_parameters(
                    inputs=inputs,
                    outputs=expected_outputs,
                    network_parameters=model.coder_parameters,
                    group_index=group_index + 1,
                    h=h,
                    w=w,
                )

                if group_index == 1 and h == 1 and w == 2:
                    break

            if group_index == 1 and h == 1 and w == 2:
                break

        if group_index == 1 and h == 1 and w == 2:
            break

    sample[2:] = 0
    sample[1:, 2:] = 0
    sample[1:, 1:, 2:] = 0

    inputs = autoregression.pad_2d(sample, padding, padding)
    _, actual_outputs = autoregression.initialize_network_outputs(model.coder_parameters, height, width)

    actual_mean, actual_scale = None, None

    for group_index in range(num_groups):
        for h in range(height):
            for w in range(width):
                actual_mean, actual_scale = grouped.compute_group_parameters(
                    inputs=inputs,
                    outputs=actual_outputs,
                    network_parameters=model.coder_parameters,
                    group_index=group_index + 1,
                    h=h,
                    w=w,
                )

                if group_index == 1 and h == 1 and w == 2:
                    break

            if group_index == 1 and h == 1 and w == 2:
                break

        if group_index == 1 and h == 1 and w == 2:
            break

    assert expected_mean == actual_mean
    assert expected_scale == actual_scale

    for expected_output, actual_output in zip(expected_outputs, actual_outputs):
        num_different = np.sum(expected_output != actual_output)

        assert num_different == 0


def test_compute_location_parameters():
    shape = (8, 16, 32)

    num_channels, height, width = shape

    sample = np.random.randint(low=-100, high=100, size=shape)
    sample = sample + np.random.rand(*shape)
    sample -= .5
    sample = sample.astype(np.float32)

    model = GaussianEntropyModel(num_channels, pre_group_size=2)
    model.update_coder_parameters()

    expected_parameters = torch.tensor(sample[None])
    expected_parameters = model.pdf_parameters(expected_parameters)[0]
    expected_parameters = functions.to_numpy(expected_parameters)
    expected_means, expected_scales = autoregression.disentangle(expected_parameters)

    padding = model.coder_parameters[0][0].shape[2] // 2
    group_size = model.coder_parameters[0][4]
    num_groups = num_channels // group_size

    inputs = autoregression.pad_2d(sample, padding, padding)
    _, outputs = autoregression.initialize_network_outputs(model.coder_parameters, height, width)

    for group_index in range(num_groups):
        start_channel = group_index * group_size
        end_channel = start_channel + group_size

        for h in range(height):
            for w in range(width):
                actual_parameters = grouped.compute_group_parameters(
                    inputs=inputs,
                    outputs=outputs,
                    network_parameters=model.coder_parameters,
                    group_index=group_index + 1,
                    h=h,
                    w=w,
                )

                actual_means, actual_scales = autoregression.disentangle(actual_parameters)

                assert actual_means == pytest.approx(expected_means[start_channel:end_channel, h, w], abs=1e-4)
                assert actual_scales == pytest.approx(expected_scales[start_channel:end_channel, h, w], abs=1e-4)


def test_compute_parameters():
    shape = (4, 16, 32)
    group_size = 2

    sample = np.random.randint(low=-100, high=100, size=shape)
    sample = sample + np.random.rand(*shape)
    sample -= .5
    sample = sample.astype(np.float32)

    model = GaussianEntropyModel(shape[0], pre_group_size=group_size)
    model.update_coder_parameters()

    inputs = np.empty_like(sample)

    parameters = grouped.compute_parameters(sample, sample[:0], model.coder_parameters)

    for index, (mean, _, symbol) in enumerate(parameters):
        c, h, w = autoregression.index_to_location(index, shape, group_size)
        inputs[c, h, w] = symbol + mean

    expected_parameters = torch.tensor(inputs[None])
    expected_parameters = model.pdf_parameters(expected_parameters)[0]
    expected_parameters = functions.to_numpy(expected_parameters)
    expected_means, expected_scales = autoregression.disentangle(expected_parameters)
    expected_scales = np.abs(expected_scales)

    parameters = grouped.compute_parameters(sample, sample[:0], model.coder_parameters)

    for index, (_, scale, symbol) in enumerate(parameters):
        c, h, w = autoregression.index_to_location(index, shape, group_size)

        cdf_index = distribution.calculate_cdf_index(scale, model.scale_table, model.scale_lower_bound)

        expected_symbol = int(quantization.round_toward_zero(sample[c, h, w] - expected_means[c, h, w]))
        expected_cdf_index = distribution.calculate_cdf_index(
            scale=expected_scales[c, h, w],
            table=model.scale_table,
            lower_bound=model.scale_lower_bound,
        )

        assert abs(symbol - expected_symbol) <= 1
        assert abs(cdf_index - expected_cdf_index) <= 1
