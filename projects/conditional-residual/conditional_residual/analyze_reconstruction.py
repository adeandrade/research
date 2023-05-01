import defopt
import matplotlib.pyplot as pyplot
import sfu_torch_lib.io as io
import sfu_torch_lib.state as state
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import conditional_residual.processing_cityscapes as processing
from conditional_residual.dataset_cityscapes import Cityscapes
from conditional_residual.model_scalable import Preview


def min_max(tensor: Tensor) -> Tensor:
    tensor_max = torch.amax(tensor)
    tensor_min = torch.amin(tensor)

    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    return tensor


def analyze(
        *,
        run_id: str,
        dataset_path: str = 's3://datasets/cityscapes.zip',
        batch_size: int = 5,
) -> None:
    """
    Prints the reconstructed images of a model.

    :param run_id: Experiment ID of this run.
    :param dataset_path: Path to dataset.
    :param batch_size: Batch size.
    """
    dataset_path = io.localize_dataset(dataset_path, overwrite=False)

    transform = processing.create_test_transformer(means=None, scales=None)
    dataset = Cityscapes(dataset_path, transform, split='validation')
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )

    model = state.load_model(run_id, Preview, cache=True, overwrite=False)

    inputs, _, _ = next(iter(dataloader))

    reconstructed = model(inputs)

    original = (inputs[0].permute((1, 2, 0)) * 255).int().numpy(force=True)
    reconstructed = (reconstructed[0].permute((1, 2, 0)).clip(0, 1) * 255).int().numpy(force=True)
    pyplot.imshow(original)
    pyplot.imshow(reconstructed)


if __name__ == '__main__':
    defopt.run(analyze)
