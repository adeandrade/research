import defopt
import mlflow
import sfu_torch_lib.mlflow as mlflow_lib
import torch
from sfu_torch_lib import slack, state
from torch import Generator, Tensor
from torch.utils.data import DataLoader, RandomSampler

import lm_codec.analyze_rate_distortion as rd
import lm_codec.analyze_spectral as spectral
from lm_codec.analyze_spectral import LMType
from lm_codec.dataset_openwebtext import OpenWebText, OpenWebTextReTokenize
from lm_codec.model_lm_codec_litgpt import LMCodecAdHoc as LMCodecAdHocLitGPT


@torch.no_grad
def transform(x: Tensor, model: LMType) -> Tensor:
    return model.lm.embed(x)


@torch.compiler.set_stance('force_eager')
def analysis(x: Tensor, model: LMType) -> Tensor:
    *_, (y, *_) = model.lm.predict(x, return_blocks={model.split_index}, quantize=False)

    return y


@slack.notify
@mlflow_lib.install
def analyze(
    *,
    run_id_pretrained: str,
    run_id: str | None = None,
    dataset_type: str = 'openwebtext',
    num_documents: int = 100,
    block_size: int = 512,
    num_iterations: int = 1000,
    num_steps: int = 10,
    seed: int = 110069,
) -> None:
    """
    Compute spectral norm of the function generating the target representation.

    :param run_id_pretrained: run ID of the training run
    :param run_id: run ID of the current run
    :param dataset_type: dataset name
    :param num_documents: number of samples
    :param block_size: context size
    :param num_iterations: number of power iterations
    :param num_steps: number of repeated runs for error analysis
    :param seed: random seed
    """
    model = state.load_model(run_id_pretrained, cache=True, overwrite=False)
    assert isinstance(model, LMType)
    model = model.eval()

    def dataset_transform(batch: tuple[Tensor, Tensor]) -> Tensor:
        inputs, _ = batch
        inputs = inputs.cuda()

        return transform(inputs[None], model)[0]

    if dataset_type == 'openwebtext':
        if hasattr(model, 'tokenizer'):
            assert isinstance(model, LMCodecAdHocLitGPT)

            dataset = OpenWebTextReTokenize(model.tokenizer, 'validation', block_size, transform=dataset_transform)

        else:
            dataset = OpenWebText('validation', block_size, transform=dataset_transform)

    else:
        raise ValueError(f'Validation dataset {dataset_type} not supported.')

    generator = Generator().manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=RandomSampler(dataset, False, num_documents, generator),
    )

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
        value = spectral.calculate_expectation_v(
            lambda inputs, v: spectral.calculate_spectral_norm(
                spectral.create_jacobian_operator(
                    lambda x: analysis(x, model),
                    inputs,
                ),
                num_iterations,
                v,
            ),
            dataloader,
        )

        mlflow.log_metrics(
            {
                'Index': model.split_index,
                'Spectral Norm': value,
                'Loss': loss,
                'BPT': bpt,
                'Distortion': distortion,
            },
            step=step,
        )


def main():
    defopt.run(analyze)


if __name__ == '__main__':
    main()
