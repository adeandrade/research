from collections.abc import Callable

import defopt
import mlflow
import sfu_torch_lib.mlflow as mlflow_lib
import torch
import tqdm
from sfu_torch_lib import slack, state
from torch import Generator, Tensor
from torch.utils.data import DataLoader, RandomSampler

import lm_codec.analyze_rate_distortion as rd
import lm_codec.analyze_spectral as sp
from lm_codec.dataset_openwebtext import OpenWebText, OpenWebTextReTokenize
from lm_codec.model_lm_codec import LMCodec
from lm_codec.model_lm_codec_litgpt import LMCodecAdHoc as LMCodecAdHocLitGPT

LMType = LMCodecAdHocLitGPT | LMCodec


def calculate_rademacher(inputs: Tensor) -> float:
    batch_size, *_ = inputs.shape

    alphas = torch.rand((batch_size, 1), dtype=inputs.dtype, device=inputs.device)
    alphas = torch.round(alphas)
    alphas = 1 - 2 * alphas

    value = alphas * inputs
    value = torch.mean(value, dim=0)
    value = torch.abs(value)
    value = torch.amax(value)
    return value.item()


def calculate_covariance(inputs: Tensor, num_iterations: int = 1000) -> float:
    operator = sp.create_simple_operator(inputs)

    return sp.calculate_determinant_torch_singular(operator, num_iterations)


def calculate_expectation(function: Callable[[], float], num_iterations: int) -> float:
    expectation = 0

    for index in tqdm.tqdm(range(num_iterations)):
        value = function()

        expectation += (value - expectation) / (index + 1)

    return expectation


@torch.no_grad
def transform(batch: tuple[Tensor, Tensor], model: LMType) -> Tensor:
    inputs, _ = batch
    batch_size, *_ = inputs.shape

    *_, (representation, *_) = model.forward(inputs, return_blocks={model.split_index}, quantize=True)
    return torch.reshape(representation, (batch_size, -1))


def get_dataset(model: LMType, dataloader: DataLoader) -> Tensor:
    return torch.concatenate([transform(inputs, model) for inputs in iter(dataloader)])


@slack.notify
@mlflow_lib.install
def analyze(
    *,
    run_id_pretrained: str,
    run_id: str | None = None,
    dataset_type: str = 'openwebtext',
    num_documents: int = 1000,
    block_size: int = 512,
    num_iterations: int = 10000,
    batch_size: int = 10,
    num_steps: int = 10,
    seed: int = 110069,
) -> None:
    """
    Computes estimates of the Rademacher complexity and covariance determinant.

    :param run_id_pretrained: run ID of the training run
    :param run_id: run ID of the current run
    :param dataset_type: dataset name
    :param num_documents: number of samples
    :param block_size: context size
    :param num_iterations: number of Rademacher samples
    :param batch_size: batch size for dataset generation
    :param num_steps: number of repeated runs for error analysis
    :param seed: random seed
    """
    model = state.load_model(run_id_pretrained, cache=True, overwrite=False)
    assert isinstance(model, LMType)
    model = model.eval()

    def input_transform(batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()

        return inputs, targets

    if dataset_type == 'openwebtext':
        if hasattr(model, 'tokenizer'):
            assert isinstance(model, LMCodecAdHocLitGPT)

            dataset = OpenWebTextReTokenize(model.tokenizer, 'validation', block_size, transform=input_transform)

        else:
            dataset = OpenWebText('validation', block_size, transform=input_transform)

    else:
        raise ValueError(f'Validation dataset {dataset_type} not supported.')

    generator = Generator().manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size,
        sampler=RandomSampler(dataset, False, num_documents, generator),
    )

    dataset = get_dataset(model, dataloader)

    loss, bpt, distortion = rd.find_best_metrics(
        run_id=run_id_pretrained,
        calculate_loss=lambda loss, *_: loss,
        metric_labels=['Validation Loss', 'Validation BPT', 'Validation Distortion'],
    )

    mlflow.log_metrics({
        'Index': model.split_index,
        'Loss': loss,
        'BPT': bpt,
        'Distortion': distortion,
    })

    for step in range(num_steps):
        complexity = calculate_expectation(
            lambda: calculate_rademacher(dataset),
            num_iterations,
        )

        covariance = calculate_covariance(dataset)

        mlflow.log_metrics(
            {
                'Rademacher Complexity': complexity,
                'Covariance': covariance,
            },
            step=step,
        )


def main():
    defopt.run(analyze)


if __name__ == '__main__':
    main()
