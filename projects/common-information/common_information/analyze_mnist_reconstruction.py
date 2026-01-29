import os

import defopt
import sfu_torch_lib.io as io
import sfu_torch_lib.state as state
import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader

import common_information.processing_mnist as processing
from common_information.dataset_mnist import MNISTColored
from common_information.model_mnist import MNISTColoredReconstruction


def create_image_grid(
    pmf_type: str,
    run_id: str,
    *,
    dataset_path: str = 's3://datasets/mnist.zip',
    format: str = 'pdf',
    path: str = f'{os.environ["PROJECT_DIR"]}/results/image-grid-{{filename}}.{{format}}',
    batch_size: int = 20,
) -> None:
    dataset_path = io.localize_dataset(dataset_path)

    transform = processing.create_test_transform()

    dataset = MNISTColored(dataset_path, pmf_type, 'validation', transform)

    model = state.load_model(run_id, MNISTColoredReconstruction)

    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    inputs, *_ = next(iter(dataloader))

    inputs = inputs.cuda()

    *_, (reconstructions_a, reconstructions_b, reconstructions_c) = model.reconstruct(inputs)

    images = torch.concatenate((inputs, reconstructions_a, reconstructions_c, reconstructions_b), dim=0)

    utils.save_image(images, path.format(filename=pmf_type, format=format), nrow=batch_size, pad_value=1)


def main():
    defopt.run(create_image_grid)


if __name__ == '__main__':
    main()
