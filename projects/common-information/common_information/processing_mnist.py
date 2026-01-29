from typing import Callable

import numpy as np
import sfu_torch_lib.tree as tree_lib
import torch
from sfu_torch_lib.processing import ComposeTree, ConvertImageTree, ResizeTree, ToTensorTree
from torch import Tensor
from torch.nn import Module
from torchvision.transforms.functional import InterpolationMode


class PermuteChannelTree(Module):
    def forward(self, tree):
        return tree_lib.map_tree(lambda x: torch.permute(x, (2, 0, 1)), tree)


class ScaleImageTree(Module):
    def __init__(self, means: Tensor | None = None, scales: Tensor | None = None) -> None:
        super().__init__()

        self.means = means
        self.scales = scales

    def transform(self, tensor: Tensor) -> Tensor:
        tensor -= self.means[:, None, None] if self.means is not None else 0
        tensor /= self.scales[:, None, None] if self.scales is not None else 255

        return tensor

    def forward(self, tree):
        return tree_lib.map_tree(self.transform, tree)


def create_test_transform() -> Callable:
    return ComposeTree([
        ([0], ConvertImageTree('RGB')),
        ([0], ToTensorTree(np.uint8)),  # type: ignore
        ([0], PermuteChannelTree()),
        ([0], ResizeTree((32, 32), [InterpolationMode.BICUBIC])),
        ([0], ToTensorTree(np.float32)),  # type: ignore
        ([0], ScaleImageTree()),
    ])
