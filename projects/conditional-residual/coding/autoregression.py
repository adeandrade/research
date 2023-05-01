from typing import Tuple, Sequence, Optional

import numba
import numpy as np
from numba.typed import List


LayerParameters = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int, int, int, int, int, bool]
NetworkParameters = Sequence[LayerParameters]
NetworkOutputs = Sequence[np.ndarray]


@numba.njit
def pad_2d(inputs: np.ndarray, padding_height: int, padding_width: int) -> np.ndarray:
    num_channels, height, width = inputs.shape

    outputs = np.zeros((num_channels, height + 2 * padding_height, width + 2 * padding_width), dtype=inputs.dtype)

    outputs[:, padding_height:(padding_height + height), padding_width:(padding_width + width)] = inputs

    return outputs


@numba.njit
def ravel(inputs: np.ndarray, group_size: int) -> np.ndarray:
    num_channels, height, width = inputs.shape

    raveled = np.empty((num_channels * height * width,), dtype=inputs.dtype)

    num_groups = num_channels // group_size

    index = 0

    for group_index in range(num_groups):
        start_channel = group_index * group_size
        end_channel = start_channel + group_size

        for h in range(height):
            for w in range(width):
                for c in range(start_channel, end_channel):
                    raveled[index] = inputs[c, h, w]
                    index += 1

    return raveled


@numba.njit
def calculate_padding(layer_index: int, network_parameters: NetworkParameters) -> Tuple[int, int]:
    if layer_index + 1 < len(network_parameters):
        _, _, kernel_height, kernel_width = network_parameters[layer_index + 1][0].shape

        padding_height = kernel_height // 2
        padding_width = kernel_width // 2

    else:
        padding_height = padding_width = 0

    return padding_height, padding_width


@numba.njit
def index_to_location(index: int, shape: Tuple[int, int, int], group_size: int) -> Tuple[int, int, int]:
    num_channels, height, width = shape

    num_elements_per_group = group_size * height * width
    num_elements_per_slice = group_size * width

    group_index = index // num_elements_per_group * group_size
    index = index % num_elements_per_group

    h = index // num_elements_per_slice
    index = index % num_elements_per_slice

    w = index // group_size
    c = group_index + index % group_size

    return c, h, w


@numba.njit
def get_previous_location(c: int, h: int, w: int, height: int, width: int, padding: int) -> Tuple[int, int, int]:
    if c == 0 and h == 0 and w == 0:
        c_previous = -1
        h_previous = -1
        w_previous = -1
    elif h == 0 and w == 0:
        c_previous = c - 1
        h_previous = height - padding - 1
        w_previous = width - padding - 1
    elif w == 0:
        c_previous = c
        h_previous = h + padding - 1
        w_previous = width - padding - 1
    else:
        c_previous = c
        h_previous = h + padding
        w_previous = w + padding - 1

    return c_previous, h_previous, w_previous


@numba.njit
def disentangle(inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return inputs[::2], inputs[1::2]


@numba.njit
def convolve(inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
    _, height, width = inputs.shape
    out_channels, in_channels, length_height, length_width = weights.shape
    padding = (length_height - 1) // 2
    num_steps = height * width
    num_dimensions = in_channels * length_height * length_width

    shape = (height, width, in_channels, length_height, length_width)
    strides = inputs.strides[1:] + inputs.strides

    inputs_view = pad_2d(inputs, padding)
    inputs_view = np.lib.stride_tricks.as_strided(inputs_view, shape, strides)
    inputs_view = np.ascontiguousarray(inputs_view)
    inputs_view = np.reshape(inputs_view, (num_steps, num_dimensions))

    weights_view = np.reshape(weights, (out_channels, num_dimensions))

    outputs = np.dot(weights_view, np.transpose(inputs_view))
    outputs += np.expand_dims(biases, axis=1)
    outputs = np.reshape(outputs, (out_channels, height, width))

    return outputs


@numba.njit
def convolve_location(inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
    return np.vdot(np.ravel(weights), np.ravel(inputs)) + biases


@numba.njit
def leaky_relu(inputs: np.ndarray, negative_slope: float = 1e-2) -> np.ndarray:
    return np.maximum(inputs, np.float32(0)) + np.float32(negative_slope) * np.minimum(inputs, np.float32(0))


@numba.njit
def initialize_network_outputs(
        network: NetworkParameters,
        height: int,
        width: int,
        dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, NetworkOutputs]:

    _, num_channels, _, _ = network[0][0].shape
    padding_height, padding_width = calculate_padding(-1, network)

    inputs = np.zeros((num_channels, height + 2 * padding_height, width + 2 * padding_width), dtype=dtype)

    outputs = List()

    for layer_index, (weight, _, _, _, _, _, _, _, _) in enumerate(network):
        num_channels, _, _, _ = weight.shape
        padding_height, padding_width = calculate_padding(layer_index, network)

        output = np.zeros((num_channels, height + 2 * padding_height, width + 2 * padding_width), dtype=dtype)

        outputs.append(output)

    return inputs, outputs
