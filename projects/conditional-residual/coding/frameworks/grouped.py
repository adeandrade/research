from typing import Tuple, Iterator

import numba
import numpy as np
from numba.typed import List

import coding.autoregression as autoregression
import coding.distribution as distributions
import coding.quantization as quantization
import coding.range as coding
from coding.autoregression import NetworkParameters, NetworkOutputs


@numba.njit
def get_output_parameters(
        outputs: NetworkOutputs,
        parameters: NetworkParameters,
        group_index: int,
        h: int,
        w: int,
) -> np.ndarray:

    _, _, _, _, _, out_group_size, _, out_prefix_size, _ = parameters[-1]

    start_channel = out_prefix_size + (group_index - 1) * out_group_size if group_index > 0 else 0
    end_channel = start_channel + out_group_size if group_index > 0 else out_prefix_size

    output_parameters = outputs[-1][start_channel:end_channel, h, w]

    return output_parameters


@numba.njit(parallel=True)
def compute_group_parameters(
        inputs: np.ndarray,
        outputs: NetworkOutputs,
        network_parameters: NetworkParameters,
        group_index: int,
        h: int,
        w: int,
) -> np.ndarray:

    for layer_index, (output, layer_parameters) in enumerate(zip(outputs, network_parameters)):
        weights, biases, scales, residual_layer_index, _, out_group_size, _, out_prefix_size, relu = layer_parameters

        _, _, kernel_height, kernel_width = weights.shape
        inputs_group = inputs[:, h:(h + kernel_height), w:(w + kernel_width)]

        if residual_layer_index >= 0:
            padding_height, padding_width = autoregression.calculate_padding(residual_layer_index, network_parameters)
            residual_height, residual_width = h + padding_height, w + padding_width
        else:
            residual_height, residual_width = -1, -1

        padding_height, padding_width = autoregression.calculate_padding(layer_index, network_parameters)
        output_height, output_width = h + padding_height, w + padding_width

        start_channel = out_prefix_size + (group_index - 1) * out_group_size if group_index > 0 else 0
        end_channel = start_channel + out_group_size if group_index > 0 else out_prefix_size

        for c in numba.prange(start_channel, end_channel):
            outputs_location = autoregression.convolve_location(inputs_group, weights[c], biases[c])

            if residual_layer_index >= 0:
                outputs_location += scales[c] * outputs[residual_layer_index][c, residual_height, residual_width]

            if relu:
                outputs_location = autoregression.leaky_relu(outputs_location)

            output[c, output_height, output_width] = outputs_location

        inputs = output

    output = get_output_parameters(outputs, network_parameters, group_index, h, w)

    return output


@numba.njit
def encode(
        sample: np.ndarray,
        conditional: np.ndarray,
        parameters: NetworkParameters,
        cdfs: np.ndarray,
        cdf_sizes: np.ndarray,
        offsets: np.ndarray,
        scale_table: np.ndarray,
        scale_lower_bound: float,
) -> np.ndarray:

    num_channels, height, width = sample.shape
    weights, _, _, _, group_size, _, prefix_size, _, _ = parameters[0]
    _, _, kernel_height, kernel_width = weights.shape

    num_groups = num_channels // group_size
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    inputs, outputs = autoregression.initialize_network_outputs(parameters, height, width)
    inputs[:prefix_size, padding_height:-padding_height, padding_width:-padding_width] = conditional

    states = List()

    for group_index in range(num_groups + 1):
        start_channel = prefix_size + (group_index - 1) * group_size
        end_channel = start_channel + group_size

        for h in range(height):
            for w in range(width):
                output = compute_group_parameters(inputs, outputs, parameters, group_index, h, w)

                if group_index > 0:
                    means, scales = autoregression.disentangle(output)

                    symbols = quantization.round_toward_zero(sample[start_channel:end_channel, h, w] - means)

                    cdf_indices = distributions.calculate_cdf_indices(scales, scale_table, scale_lower_bound)

                    symbols = symbols - offsets[cdf_indices]

                    for symbol, cdf_index in zip(symbols, cdf_indices):
                        cdf = cdfs[cdf_index][:cdf_sizes[cdf_index]]
                        states.extend(coding.encode_symbol(symbol, cdf))

                    inputs_group = symbols + offsets[cdf_indices] + means

                    inputs[start_channel:end_channel, h + padding_height, w + padding_width] = inputs_group

    codes = coding.encode_states(states)

    return codes


@numba.njit
def decode(
        codes: np.ndarray,
        conditional: np.ndarray,
        parameters: NetworkParameters,
        shape: Tuple[int, int, int],
        cdfs: np.ndarray,
        cdf_sizes: np.ndarray,
        offsets: np.ndarray,
        scale_table: np.ndarray,
        scale_lower_bound: float,
) -> np.ndarray:

    num_channels, height, width = shape
    weights, _, _, _, group_size, _, prefix_size, _, _ = parameters[0]
    _, _, kernel_height, kernel_width = weights.shape

    num_groups = num_channels // group_size
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    inputs, outputs = autoregression.initialize_network_outputs(parameters, height, width)
    inputs[:prefix_size, padding_height:-padding_height, padding_width:-padding_width] = conditional

    state, position, symbols = -1, -1, np.empty((group_size,), dtype=np.int32)

    for group_index in range(num_groups + 1):
        start_channel = prefix_size + (group_index - 1) * group_size
        end_channel = start_channel + group_size

        for h in range(height):
            for w in range(width):
                output = compute_group_parameters(inputs, outputs, parameters, group_index, h, w)

                if group_index > 0:
                    means, scales = autoregression.disentangle(output)

                    cdf_indices = distributions.calculate_cdf_indices(scales, scale_table, scale_lower_bound)

                    for index, cdf_index in enumerate(cdf_indices):
                        cdf = cdfs[cdf_index][:cdf_sizes[cdf_index]]
                        symbols[index], state, position = coding.decode_symbol(state, position, codes, cdf)

                    inputs_group = symbols + offsets[cdf_indices] + means

                    inputs[start_channel:end_channel, h + padding_height, w + padding_width] = inputs_group

    inputs = inputs[:, padding_height:-padding_height, padding_width:-padding_width]

    return inputs


@numba.njit
def compute_parameters(
        sample: np.ndarray,
        conditional: np.ndarray,
        parameters: NetworkParameters,
) -> Iterator[Tuple[float, float, int]]:

    num_channels, height, width = sample.shape
    weights, _, _, _, group_size, _, prefix_size, _, _ = parameters[0]
    _, _, kernel_height, kernel_width = weights.shape

    num_groups = num_channels // group_size
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    inputs, outputs = autoregression.initialize_network_outputs(parameters, height, width)
    inputs[:prefix_size, padding_height:-padding_height, padding_width:-padding_width] = conditional

    for group_index in range(num_groups + 1):
        start_channel = prefix_size + (group_index - 1) * group_size
        end_channel = start_channel + group_size

        for h in range(height):
            for w in range(width):
                output = compute_group_parameters(inputs, outputs, parameters, group_index, h, w)

                if group_index > 0:
                    means, scales = autoregression.disentangle(output)

                    symbols = quantization.round_toward_zero(sample[start_channel:end_channel, h, w] - means)

                    for mean, scale, symbol in zip(means, scales, symbols):
                        yield mean, scale, symbol

                    inputs[start_channel:end_channel, h + padding_height, w + padding_width] = symbols + means
