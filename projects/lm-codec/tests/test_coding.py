import pytest
import torch
from sfu_torch_lib import io, state
from torch.utils.data import DataLoader

from lm_codec import functions
from lm_codec.dataset_openwebtext import OpenWebText
from lm_codec.model_lm_codec import LMCodecHyperPrior

RUN_ID = '8c4d1b418836474eb1e15eae2b95b661'
DATASET_PATH = 's3://datasets/openwebtext.tar'


@pytest.mark.skip(reason='not an unit test')
def test_autoregressive() -> None:
    dataset_path = io.localize_dataset(DATASET_PATH)

    dataset = OpenWebText(dataset_path)

    model = state.load_model(RUN_ID, LMCodecHyperPrior, cache=True, overwrite=False)
    model = model.eval().cuda()

    assert isinstance(model, LMCodecHyperPrior)

    dataloader = DataLoader(dataset, batch_size=2)

    batch, _ = next(iter(dataloader))
    batch = batch.cuda()

    *_, (representation, *_) = model.lm.forward(batch, return_blocks={model.split_index})

    parameters = model.codec.analysis_prior(representation)
    parameters = functions.soft_round(parameters, 0)
    parameters = model.codec.synthesis_prior(parameters)

    representation_expected = model.codec.lst_representation(representation, parameters)

    representation[:, 100:] = 0
    representation_actual = model.codec.lst_representation(representation, parameters)

    torch.testing.assert_close(representation_actual[:, :101], representation_expected[:, :101])


@pytest.mark.skip(reason='not an unit test')
def test_coding() -> None:
    dataset_path = io.localize_dataset(DATASET_PATH)

    dataset = OpenWebText(dataset_path, block_size=3)

    model = state.load_model(RUN_ID, LMCodecHyperPrior, cache=True, overwrite=False)
    model = model.eval().cuda()

    assert isinstance(model, LMCodecHyperPrior)

    dataloader = DataLoader(dataset, batch_size=2)

    batch, _ = next(iter(dataloader))
    batch = batch.cuda()

    *_, (representation, *_) = model.lm.forward(batch, return_blocks={model.split_index})

    (representation_expected, hyper_prior_expected), (bpt_representation_expected, bpt_hyper_prior_expected) = (
        model.codec.forward(representation, 0, quantized=True)
    )
    (representation_actual, hyper_prior_actual), (bpt_representation_actual, bpt_hyper_prior_actual) = (
        model.codec.forward_coded(representation, quantized=True)
    )

    assert bpt_hyper_prior_actual <= bpt_hyper_prior_expected * 1.01
    assert torch.mean((hyper_prior_actual != hyper_prior_expected).float()) == 0

    assert bpt_representation_actual <= bpt_representation_expected * 1.01
    assert torch.mean((representation_actual != representation_expected).float()) == 0
