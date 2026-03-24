import defopt
import mlflow
import sfu_torch_lib.mlflow as mlflow_lib
import torch
from PIL.Image import Image
from sfu_torch_lib import io, slack, state
from torch import Generator, Tensor
from torch.utils.data import DataLoader, RandomSampler

import lm_codec.analyze_rate_distortion as rd
import lm_codec.analyze_spectral as spectral
from image_codec import processing
from image_codec.dataset_imagenet import Imagenet
from image_codec.model import ResNetAdHoc, ViTAdHoc
from lm_codec import functions

VisualModelType = ResNetAdHoc | ViTAdHoc


@torch.no_grad
def transform(x: tuple[Tensor, Tensor], model: VisualModelType) -> Tensor:
    return model(*x, {model.split_index})[2][0]


@torch.compiler.set_stance('force_eager')
def predict(x: Tensor, model: VisualModelType, mu: Tensor | None = None) -> Tensor:
    parameters = model.patchify(x) if isinstance(model, ResNetAdHoc) else x
    parameters = model.analysis_prior(parameters)
    parameters = functions.quantize(parameters)
    parameters = model.synthesis_prior(parameters)
    parameters = model.unpatchify(parameters, x.shape) if isinstance(model, ResNetAdHoc) else parameters

    likelihoods = -1 * model.model_representation.nll_discrete(x, parameters)
    return likelihoods if mu is None else likelihoods - mu * x


@slack.notify
@mlflow_lib.install
def analyze(
    *,
    run_id_pretrained: str,
    run_id: str | None = None,
    dataset_type: str = 'imagenet',
    dataset_path_imagenet: str = 's3://datasets/imagenet.zip',
    num_samples: int = 100,
    num_iterations: int = 1000,
    num_steps: int = 10,
    seed: int = 110069,
) -> None:
    """
    Compute spectral norm of the entropy model.

    :param run_id_pretrained: run ID of the training run
    :param run_id: run ID of the current run
    :param dataset_type: dataset name
    :param dataset_path_imagenet: path to the ImageNet dataset
    :param num_samples: number of samples
    :param num_iterations: number of power iterations
    :param num_steps: number of repeated runs for error analysis
    :param seed: random seed
    """
    model = state.load_model(run_id_pretrained, cache=True, overwrite=False)
    assert isinstance(model, VisualModelType)
    model = model.eval()

    dataset_transform = processing.create_detection_test_transform()

    def input_transform(batch: tuple[Image, int]) -> Tensor:
        inputs, targets = dataset_transform(batch)
        inputs = inputs.cuda()
        targets = targets.cuda()
        return transform((inputs[None], targets[None]), model)[0]

    if dataset_type == 'imagenet':
        dataset_path = io.localize_dataset(dataset_path_imagenet)

        dataset = Imagenet(dataset_path, input_transform, 'validation')

    else:
        raise ValueError(f'Validation dataset {dataset_type} not supported.')

    generator = Generator().manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=RandomSampler(dataset, False, num_samples, generator),
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
                    lambda x: predict(x, model),
                    inputs,
                ),
                num_iterations,
                v,
            ),
            dataloader,
        )

        mlflow.log_metrics(
            {
                'Spectral Norm': value,
            },
            step=step,
        )


def main():
    defopt.run(analyze)


if __name__ == '__main__':
    main()
