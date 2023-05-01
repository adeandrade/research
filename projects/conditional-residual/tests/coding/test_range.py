import numpy as np
import torch
from compressai.ans import RansEncoder, RansDecoder

import coding.autoregression as autoregression
import coding.conversion as conversion
import coding.distribution as distribution
import coding.frameworks.grouped as grouped
import coding.models.functions as functions
import coding.quantization as quantization
import coding.range_fixed as coding
from coding.models.gaussian import GaussianEntropyModel


def test_encoder():
    symbols = np.array([2, 100])
    indices = np.array([0, 1])
    cdfs = np.array([
        [0, 1, 3, 5, 0],
        [0, 10, 100, 110, 200],
    ])
    cdf_sizes = np.array([4, 5])
    offsets = np.array([2, 3])

    encoder = RansEncoder()
    expected = encoder.encode_with_indexes(symbols, indices, cdfs, cdf_sizes, offsets)

    actual = coding.encode_symbols(symbols, indices, cdfs, cdf_sizes, offsets)
    actual = conversion.int_array_to_bytes(actual)

    assert actual == expected


def test_encoder_tensor():
    shape = (10, 128, 16, 32)

    batch = torch.randint(low=-100, high=100, size=shape)
    batch = batch + torch.rand(shape)
    batch -= .5

    model = GaussianEntropyModel(shape[1])

    actual = model.encode(batch)

    encoder = RansEncoder()

    expected = []

    for sample in functions.to_numpy(batch):
        parameters = grouped.compute_parameters(sample, sample[:0], model.coder_parameters)

        symbols, indices = [], []

        for _, scale, symbol in parameters:
            cdf_index = distribution.calculate_cdf_index(scale, model.scale_table, model.scale_lower_bound)

            symbols.append(symbol)
            indices.append(cdf_index)

        byte_array = encoder.encode_with_indexes(
            symbols,
            indices,
            model.cdfs.tolist(),
            model.cdf_sizes.tolist(),
            model.offsets.tolist(),
        )

        expected.append(byte_array)

    assert actual == expected


def test_decoder():
    codes = b'\x00\x00l\xc1\x16\x00\x00\x00\xbc\x00\xe2\x16'
    index_first = 0
    index_second = 1
    cdfs = np.array([
        [0, 1, 3, 5, 0],
        [0, 10, 100, 110, 200],
    ])
    cdf_sizes = np.array([4, 5])
    offsets = np.array([2, 3])

    decoder = RansDecoder()
    decoder.set_stream(codes)
    expected_first = decoder.decode_stream([0], cdfs, cdf_sizes, offsets)[0]
    expected_second = decoder.decode_stream([1], cdfs, cdf_sizes, offsets)[0]

    byte_array = conversion.bytes_to_int_array(codes)
    actual_first, state, position = coding.decode_symbol(-1, -1, index_first, byte_array, cdfs, cdf_sizes, offsets)
    actual_second, _, _ = coding.decode_symbol(state, position, index_second, byte_array, cdfs, cdf_sizes, offsets)

    assert actual_first == expected_first
    assert actual_second == expected_second


def test_decoder_tensor():
    for _ in range(2):
        shape = (384, 16, 32)

        batch = torch.randint(low=-100, high=100, size=shape)
        batch = batch + torch.rand(shape)
        batch -= .5

        model = GaussianEntropyModel(shape[0])
        model.update_coder_parameters()

        byte_string = model.encode(torch.unsqueeze(batch, dim=0))[0]
        codes = conversion.bytes_to_int_array(byte_string)

        sample = functions.to_numpy(batch)
        conditional = sample[:0]
        parameters = grouped.compute_parameters(sample, conditional, model.coder_parameters)

        means, cdf_indices = [], []

        for mean, scale, symbol in parameters:
            cdf_index = distribution.calculate_cdf_index(scale, model.scale_table, model.scale_lower_bound)

            means.append(mean)
            cdf_indices.append(cdf_index)

        batch = autoregression.ravel(functions.to_numpy(batch), model.coder_parameters[0][4])

        decoder = RansDecoder()
        decoder.set_stream(byte_string)
        expected_originals = decoder.decode_stream(
            cdf_indices,
            model.cdfs.tolist(),
            model.cdf_sizes.tolist(),
            model.offsets.tolist(),
        )
        expected_originals = torch.tensor(expected_originals, dtype=torch.float32)
        expected_originals += torch.tensor(means, dtype=torch.float32)
        expected_originals = functions.round_toward_zero(expected_originals)

        state, position = -1, -1

        for sample, cdf_index, mean, expected_original in zip(batch, cdf_indices, means, expected_originals):
            symbol, state, position = coding.decode_symbol(
                state=state,
                position=position,
                cdf_index=cdf_index,
                codes=codes,
                cdfs=model.cdfs,
                cdf_sizes=model.cdf_sizes,
                offsets=model.offsets,
            )

            actual = quantization.round_toward_zero(np.float32(symbol) + mean).item()
            expected_inputs = quantization.round_toward_zero(sample).item()
            expected_original = expected_original.item()

            assert abs(actual - expected_inputs) <= 1
            assert abs(actual - expected_original) <= 1
