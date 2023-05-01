import itertools

import pytest
import sfu_torch_lib.io as io
import sfu_torch_lib.state as state
import torch
from torch.utils.data import DataLoader

import coding.models.functions as functions
import conditional_residual.processing_cityscapes as processing
from coding.models.gaussian import GaussianEntropyModel
from conditional_residual.dataset_cityscapes import Cityscapes
from conditional_residual.model_scalable import Baseline


def test_coding():
    shape = (10, 100, 16, 32)

    inputs = torch.randint(low=-100, high=100, size=shape)
    inputs = inputs + torch.rand(shape)
    inputs -= .5

    expected = functions.round_toward_zero(inputs)

    model = GaussianEntropyModel(shape[1], pre_group_size=10)

    actual = model.decode(model.encode(inputs), shape=shape[1:])
    actual = functions.round_toward_zero(actual)

    difference_max = torch.amax(torch.abs(actual - expected)).item()

    assert difference_max <= 1


@pytest.mark.skip(reason='not an unit test')
def test_coding_real():
    dataset_path = io.localize_dataset('s3://datasets/cityscapes-tiny.zip')
    transform = processing.create_input_test_transformer(means=None, scales=None)
    dataset = Cityscapes(dataset_path, transform, split='validation')
    data_loader = DataLoader(dataset, batch_size=1)

    for batch in itertools.islice(iter(data_loader), 2):
        model = state.load_model('4ab9b048770a4462b31fa7f7097a5057', Baseline, cache=True, overwrite=False)

        expected_representations = model.encoder_reconstruction(batch)
        expected_reconstructions = model.decoder(expected_representations)

        byte_strings = model.entropy_model.encode(expected_representations)
        shape = expected_representations.shape[1:]
        actual_representations = model.entropy_model.decode(byte_strings, shape)
        actual_reconstructions = model.decoder(actual_representations)

        difference_max_representations = torch.abs(actual_representations - expected_representations)
        difference_max_representations = torch.amax(difference_max_representations).item()

        actual_reconstructions = functions.round_toward_zero(255 * actual_reconstructions)
        expected_reconstructions = functions.round_toward_zero(255 * expected_reconstructions)
        difference_max_reconstructions = torch.abs(actual_reconstructions - expected_reconstructions)
        difference_max_reconstructions = torch.amax(difference_max_reconstructions).item()

        assert difference_max_representations <= 1
        assert difference_max_reconstructions <= 1
