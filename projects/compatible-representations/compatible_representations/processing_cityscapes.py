from typing import Tuple, Callable, Optional

import numpy as np
import sfu_torch_lib.tree as tree_lib
import torch
from sfu_torch_lib.processing import EncodeImageTree, ComposeTree, ToTensorTree, ConvertImageTree
from sfu_torch_lib.processing import RandomHorizontalFlipTree, ColorJitterTree
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import Compose


CITYSCAPES_MEANS = torch.as_tensor((123.675, 116.28, 103.53), dtype=torch.float32)
CITYSCAPES_SCALES = torch.as_tensor((58.395, 57.12, 57.375), dtype=torch.float32)


class EncodeSegmentationTree(Module):
    num_classes = 20
    ignore_index = 19
    class_map = torch.as_tensor(
        data=(
            19,
            19,
            19,
            19,
            19,
            19,
            19,
            0,
            1,
            19,
            19,
            2,
            3,
            4,
            19,
            19,
            19,
            5,
            19,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            19,
            19,
            16,
            17,
            18,
            19,
        ),
        dtype=torch.int64,
    )

    def encode_segmentation_map(self, segmentation_map: Tensor) -> Tensor:
        return self.class_map[segmentation_map]

    def forward(self, tree):
        return tree_lib.map_tree(self.encode_segmentation_map, tree)


class EncodeDepthTree(Module):
    @staticmethod
    def transform(tensor: Tensor) -> Tuple[Tensor, Tensor]:
        mask = tensor > 0

        tensor[mask] = tensor[mask] - 1
        tensor /= 32768

        return tensor, mask

    @classmethod
    def forward(cls, tree):
        return tree_lib.map_tree(cls.transform, tree)


def create_segmentation_train_transformer(
    jitter: float = 0.5,
    means: Optional[Tensor] = CITYSCAPES_MEANS,
    scales: Optional[Tensor] = CITYSCAPES_SCALES,
) -> Callable:
    return Compose([
        lambda inputs: (inputs[0], inputs[1]),
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0, 1], RandomHorizontalFlipTree()),
            ([0], ColorJitterTree(brightness=jitter, contrast=jitter, saturation=jitter)),
            ([0, 1], ToTensorTree(np.float32, np.int64)),
            ([0], EncodeImageTree(means, scales)),
            ([1], EncodeSegmentationTree()),
        ]),
    ])


def create_depth_train_transformer(
    jitter: float = 0.5,
    means: Optional[Tensor] = CITYSCAPES_MEANS,
    scales: Optional[Tensor] = CITYSCAPES_SCALES,
) -> Callable:
    return Compose([
        lambda inputs: (inputs[0], inputs[2]),
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0, 1], RandomHorizontalFlipTree()),
            ([0], ColorJitterTree(brightness=jitter, contrast=jitter, saturation=jitter)),
            ([0, 1], ToTensorTree(np.float32, np.float32)),
            ([0], EncodeImageTree(means, scales)),
            ([1], EncodeDepthTree()),
        ]),
    ])


def create_input_train_transformer(
    jitter: float = 0.5,
    means: Optional[Tensor] = CITYSCAPES_MEANS,
    scales: Optional[Tensor] = CITYSCAPES_SCALES,
) -> Callable:
    return Compose([
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0], RandomHorizontalFlipTree()),
            ([0], ColorJitterTree(brightness=jitter, contrast=jitter, saturation=jitter)),
            ([0], ToTensorTree(np.float32)),
            ([0], EncodeImageTree(means, scales)),
        ]),
        lambda inputs: inputs[0],
    ])


def create_segmentation_test_transformer(
    means: Optional[Tensor] = CITYSCAPES_MEANS,
    scales: Optional[Tensor] = CITYSCAPES_SCALES,
) -> Callable:
    return Compose([
        lambda inputs: (inputs[0], inputs[1]),
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0, 1], ToTensorTree(np.float32, np.int64)),
            ([0], EncodeImageTree(means, scales)),
            ([1], EncodeSegmentationTree()),
        ]),
    ])


def create_depth_test_transformer(
    means: Optional[Tensor] = CITYSCAPES_MEANS,
    scales: Optional[Tensor] = CITYSCAPES_SCALES,
) -> Callable:
    return Compose([
        lambda inputs: (inputs[0], inputs[2]),
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0, 1], ToTensorTree(np.float32, np.float32)),
            ([0], EncodeImageTree(means, scales)),
            ([1], EncodeDepthTree()),
        ]),
    ])


def create_input_test_transformer(
    means: Optional[Tensor] = CITYSCAPES_MEANS,
    scales: Optional[Tensor] = CITYSCAPES_SCALES,
) -> Callable:
    return Compose([
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0], ToTensorTree(np.float32)),
            ([0], EncodeImageTree(means, scales)),
        ]),
        lambda inputs: inputs[0],
    ])
