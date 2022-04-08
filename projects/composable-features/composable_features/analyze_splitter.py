import functools
import itertools
from typing import List, Tuple, Optional

import defopt
import matplotlib.pyplot as pyplot
import mlflow
import numpy as np
import seaborn
import sfu_torch_lib.state as state
import torch
from matplotlib.ticker import MultipleLocator
from pandas import DataFrame
from torch import Tensor

import composable_features.model_splitter as model_splitter


def take_top_k(tensor: Tensor, k: int) -> List[Tuple[int, float]]:
    return list(
        (int(position), float(score))
        for score, position
        in itertools.islice(zip(*torch.sort(tensor, descending=True)), k)
    )


def sum_masked_values(mask: Tensor, values: Tensor) -> float:
    return torch.sum(torch.where(mask, values, torch.tensor(0., dtype=torch.float32))).tolist()  # type: ignore


def get_probabilities(run_id: str) -> Tuple[Tensor, Tensor, Tensor]:
    model_type = mlflow.get_run(run_id).data.tags['model']

    model = state.load_model(run_id, getattr(model_splitter, model_type))
    model.threshold = 0.

    probabilities = model.get_task_probabilities()

    return probabilities


def get_entropies(run_id: str, run_id_test: str, height: int, width: int) -> Tensor:
    num_channels = int(mlflow.get_run(run_id).data.params['num-channels'])

    entropies = [None] * num_channels

    for key, value in mlflow.get_run(run_id_test).data.metrics.items():
        if key.startswith('Test Entropy Channel'):
            index = int(key.split('Test Entropy Channel ')[1])
            entropies[index] = value / height / width

    entropies_tensor = torch.tensor(entropies, dtype=torch.float32)

    return entropies_tensor


def plot_probabilities(probabilities: Tuple[Tensor, Tensor, Tensor]) -> None:
    probabilities_reconstruction, probabilities_segmentation, probabilities_depth = probabilities

    data = torch.stack((probabilities_reconstruction, probabilities_segmentation, probabilities_depth), dim=0)
    data = data.detach().numpy()

    cmap = seaborn.diverging_palette(230, 20, as_cmap=True)
    labels = ('Reconstruction', 'Segmentation', 'Disparity')

    pyplot.figure(figsize=(10, 5))

    seaborn.heatmap(data, cmap=cmap, yticklabels=labels, center=0, linewidth=.5, cbar_kws={'label': 'Probability'})
    pyplot.xlabel('Channels')

    pyplot.tight_layout()
    pyplot.show()


def plot_entropies(entropies: Tensor, probabilities: Tuple[Tensor, Tensor, Tensor], threshold: float) -> None:
    probabilities_reconstruction, probabilities_segmentation, probabilities_depth = probabilities

    mask_reconstruction = probabilities_reconstruction > threshold
    mask_segmentation = probabilities_segmentation > threshold
    mask_depth = probabilities_depth > threshold
    mask = torch.logical_or(torch.logical_or(mask_reconstruction, mask_segmentation), mask_depth)

    data = DataFrame({
        'BPP': entropies,
        'Channels': np.arange(entropies.shape[0]),
        'Selection': ['Selected' if value else 'Unselected' for value in mask.tolist()],
    })

    pyplot.figure(figsize=(10, 5))

    plot = seaborn.barplot(data=data, x='Channels', y='BPP', hue=data.Selection.values, dodge=False)

    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
    plot.xaxis.set_major_locator(MultipleLocator(base=5))

    pyplot.tight_layout()
    pyplot.show()


def calculate_top_k(probabilities: Tuple[Tensor, Tensor, Tensor], top_k: int) -> None:
    probabilities_reconstruction, probabilities_segmentation, probabilities_depth = probabilities

    print(f'Reconstruction Probabilities: {take_top_k(probabilities_reconstruction, top_k)}')
    print(f'Segmentation Probabilities: {take_top_k(probabilities_segmentation, top_k)}')
    print(f'Depth Probabilities: {take_top_k(probabilities_depth, top_k)}')


def calculate_task_entropies(entropies: Tensor, probabilities: Tuple[Tensor, Tensor, Tensor], threshold: float) -> None:
    probabilities_reconstruction, probabilities_segmentation, probabilities_depth = probabilities

    calculate_entropy = functools.partial(sum_masked_values, values=entropies)

    mask_reconstruction = probabilities_reconstruction > threshold
    mask_segmentation = probabilities_segmentation > threshold
    mask_depth = probabilities_depth > threshold

    num_channels_reconstruction = torch.sum(mask_reconstruction)
    num_channels_segmentation = torch.sum(mask_segmentation)
    num_channels_depth = torch.sum(mask_depth)

    mask_reconstruction_segmentation = torch.logical_and(mask_reconstruction, mask_segmentation)
    mask_reconstruction_depth = torch.logical_and(mask_reconstruction, mask_depth)
    mask_segmentation_depth = torch.logical_and(mask_segmentation, mask_depth)

    entropy_reconstruction = calculate_entropy(mask_reconstruction)
    entropy_segmentation = calculate_entropy(mask_segmentation)
    entropy_depth = calculate_entropy(mask_depth)

    entropy_reconstruction_segmentation = calculate_entropy(mask_reconstruction_segmentation) / entropy_reconstruction
    entropy_reconstruction_depth = calculate_entropy(mask_reconstruction_depth) / entropy_reconstruction

    entropy_segmentation_reconstruction = calculate_entropy(mask_reconstruction_segmentation) / entropy_segmentation
    entropy_segmentation_depth = calculate_entropy(mask_segmentation_depth) / entropy_segmentation

    entropy_depth_reconstruction = calculate_entropy(mask_reconstruction_depth) / entropy_depth
    entropy_depth_segmentation = calculate_entropy(mask_segmentation_depth) / entropy_depth

    print(f'Number Channels Reconstruction {num_channels_reconstruction}')
    print(f'Number Channels Segmentation {num_channels_segmentation}')
    print(f'Number Channels Depth {num_channels_depth}')

    print(f'Entropy Reconstruction {entropy_reconstruction}')
    print(f'Entropy Segmentation {entropy_segmentation}')
    print(f'Entropy Depth {entropy_depth}')

    print(f'Entropy Segmentation/Reconstruction {entropy_reconstruction_segmentation}')
    print(f'Entropy Depth/Reconstruction {entropy_reconstruction_depth}')

    print(f'Entropy Reconstruction/Segmentation {entropy_segmentation_reconstruction}')
    print(f'Entropy Depth/Segmentation {entropy_segmentation_depth}')

    print(f'Entropy Reconstruction/Depth {entropy_depth_reconstruction}')
    print(f'Entropy Segmentation/Depth {entropy_depth_segmentation}')


def evaluate(
        *,
        run_id: str,
        run_id_test: Optional[str] = None,
        top_k: int = 50,
        height: int = 256,
        width: int = 512,
        threshold: float = .5,
) -> None:
    """
    Prints stats and plots for a splitter model.

    :param run_id: Experiment ID of an existing model
    :param run_id_test: Experiment ID of a test run.
    :param top_k: Top K probabilities to print.
    :param height: Height of the input.
    :param width: Width of the input.
    :param threshold: Threshold for the selection of a factor.
    """

    probabilities = get_probabilities(run_id)
    plot_probabilities(probabilities)
    calculate_top_k(probabilities, top_k)

    if run_id_test:
        entropies = get_entropies(run_id, run_id_test, height, width)
        plot_entropies(entropies, probabilities, threshold)
        calculate_task_entropies(entropies, probabilities, threshold)


if __name__ == '__main__':
    defopt.run(evaluate)
