import pytest
import torch
from sfu_torch_lib import io, state
from torch.utils.data import DataLoader

from lm_codec import functions
from lm_codec.dataset_openwebtext import OpenWebText
from lm_codec.model_lm_codec import LMCodecFourier

RUN_ID = '2236e86163a54a6684738679fe450424'
DATASET_PATH = 's3://datasets/openwebtext.tar'


@pytest.mark.skip(reason='not an unit test')
def test_cdf() -> None:
    dataset_path = io.localize_dataset(DATASET_PATH)

    dataset = OpenWebText(dataset_path)

    model = state.load_model(RUN_ID, LMCodecFourier, cache=True, overwrite=False)
    model = model.eval().cuda()

    assert isinstance(model, LMCodecFourier)

    dataloader = DataLoader(dataset, batch_size=2)

    batch, _ = next(iter(dataloader))
    batch = batch.cuda()

    *_, (representation, *_) = model.lm.forward(batch, return_blocks={model.split_index})
    parameters = model.parameters_prior[None, None]

    hyper_prior = model.analysis_prior(representation)
    hyper_prior = functions.quantize(hyper_prior)

    actual = model.model_prior.cdf(hyper_prior + 0.5, parameters)
    actual = actual - model.model_prior.cdf(hyper_prior - 0.5, parameters)

    expected = model.model_prior.cdf_symbol(hyper_prior, parameters)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skip(reason='not an unit test')
def test_coding() -> None:
    dataset_path = io.localize_dataset(DATASET_PATH)

    dataset = OpenWebText(dataset_path, block_size=3)

    model = state.load_model(RUN_ID, LMCodecFourier, cache=True, overwrite=False)
    model = model.eval().cuda()

    assert isinstance(model, LMCodecFourier)

    dataloader = DataLoader(dataset, batch_size=2)

    batch, _ = next(iter(dataloader))
    batch = batch.cuda()

    _, block_size = batch.shape

    *_, (representation_expected, *_) = model.lm.forward(batch, return_blocks={model.split_index})

    hyper_prior_expected = model.analysis_prior(representation_expected)
    hyper_prior_expected = functions.quantize(hyper_prior_expected)

    likelihoods = model.model_prior.nll_discrete(hyper_prior_expected, model.parameters_prior[None, None])
    bpt_hyper_prior_expected = functions.calculate_bpe(likelihoods, block_size)

    parameters_representation = model.synthesis_prior(hyper_prior_expected)

    likelihoods = model.model_representation.nll_discrete(representation_expected, parameters_representation)
    bpt_representation_expected = functions.calculate_bpe(likelihoods, block_size)

    (representation_actual, hyper_prior_actual), (bpt_representation_actual, bpt_hyper_prior_actual) = (
        model.forward_coded(representation_expected, quantized=True)
    )

    assert bpt_hyper_prior_actual <= bpt_hyper_prior_expected * 1.01
    assert torch.mean((hyper_prior_actual != hyper_prior_expected).float()) == 0

    assert bpt_representation_actual <= bpt_representation_expected * 1.04
    assert torch.mean((representation_actual != representation_expected).float()) == 0
