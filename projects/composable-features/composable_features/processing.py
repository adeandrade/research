from typing import Tuple, Callable, Dict, Optional

import numpy as np
import sfu_torch_lib.tree as tree_lib
import torch
from sfu_torch_lib.processing import EncodeImageTree, ComposeTree, ToTensorTree, SelectTree, ResizeTree
from sfu_torch_lib.processing import RandomResizedCropTree, RandomHorizontalFlipTree, ColorJitterTree
from sfu_torch_lib.processing import RandomRotationTree
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode


CITYSCAPES_MEANS = torch.as_tensor((0.485, 0.456, 0.406), dtype=torch.float32)
CITYSCAPES_VARIANCES = torch.as_tensor((0.229, 0.224, 0.225), dtype=torch.float32)


BatchType = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]


class EncodeSegmentationTree(Module):
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    ignore_index = len(valid_classes)

    def __init__(self) -> None:
        super().__init__()

        self.class_map = self.generate_class_map()

    @classmethod
    def generate_class_map(cls) -> Dict[int, int]:
        class_map = {}

        for void_class in cls.void_classes:
            class_map[void_class] = cls.ignore_index

        for index, valid_class in enumerate(cls.valid_classes):
            class_map[valid_class] = index

        return class_map

    def encode_segmentation_map(self, segmentation_map: Tensor) -> Tensor:
        """
        Assign class ids to class indices.
        """
        encoded = segmentation_map

        for class_id, index in self.class_map.items():
            encoded = torch.masked_fill(encoded, segmentation_map == class_id, index)

        return encoded

    def forward(self, tree: Tensor) -> Tuple[Tensor, ...]:
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


def create_backbone_transformer(
        image_size: Tuple[int, int],
        degrees: float = 10.,
        jitter: float = 0.5,
        means: Optional[Tensor] = CITYSCAPES_MEANS,
        variances: Optional[Tensor] = CITYSCAPES_VARIANCES,
) -> Callable:

    ratio = image_size[1] / image_size[0]

    return Compose([
        SelectTree([True, False, False]),
        ComposeTree([
            ([0], RandomRotationTree(degrees, (InterpolationMode.BILINEAR,))),
            ([0], RandomResizedCropTree((image_size,), (InterpolationMode.BILINEAR,), ratio=(ratio, ratio))),
            ([0], RandomHorizontalFlipTree()),
            ([0], ColorJitterTree(brightness=jitter, contrast=jitter, saturation=jitter)),
            ([0], ToTensorTree(np.float32)),
            ([0], EncodeImageTree(means, variances)),
        ]),
    ])


def create_classifier_transformer(
        data_size: Tuple[int, int],
        targets_size: Optional[Tuple[int, int]] = None,
        degrees: float = 10.,
        jitter: float = .5,
        means: Optional[Tensor] = CITYSCAPES_MEANS,
        variances: Optional[Tensor] = CITYSCAPES_VARIANCES,
) -> Callable:

    targets_size = targets_size if targets_size else data_size
    ratio = data_size[1] / data_size[0]

    return ComposeTree([
        ([0, 1, 2], RandomRotationTree(
            degrees=degrees,
            interpolations=(InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST),
        )),
        ([0, 1, 2], RandomResizedCropTree(
            sizes=(data_size, targets_size, targets_size),
            interpolations=(InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST),
            ratio=(ratio, ratio),
        )),
        ([0, 1, 2], RandomHorizontalFlipTree()),
        ([0], ColorJitterTree(brightness=jitter, contrast=jitter, saturation=jitter)),
        ([0, 1, 2], ToTensorTree(np.float32, np.int64, np.float32)),
        ([0], EncodeImageTree(means, variances)),
        ([1], EncodeSegmentationTree()),
        ([2], EncodeDepthTree()),
    ])


def create_saeed_transformer(
        data_size: Tuple[int, int],
        targets_size: Optional[Tuple[int, int]] = None,
        means: Optional[Tensor] = CITYSCAPES_MEANS,
        variances: Optional[Tensor] = CITYSCAPES_VARIANCES,
) -> Callable:

    targets_size = targets_size if targets_size else data_size

    return ComposeTree([
        ([0, 1, 2], ResizeTree(
            sizes=(data_size, targets_size, targets_size),
            interpolations=(InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST),
        )),
        ([0, 1, 2], ToTensorTree(np.float32, np.int64, np.float32)),
        ([0], EncodeImageTree(means, variances)),
        ([1], EncodeSegmentationTree()),
        ([2], EncodeDepthTree()),
    ])


def create_backbone_saeed_transformer(
        image_size: Tuple[int, int],
        means: Optional[Tensor] = CITYSCAPES_MEANS,
        variances: Optional[Tensor] = CITYSCAPES_VARIANCES,
) -> Callable:

    return Compose([
        SelectTree([True, False, False]),
        ComposeTree([
            ([0], ResizeTree((image_size,), (InterpolationMode.BILINEAR,))),
            ([0], ToTensorTree(np.float32)),
            ([0], EncodeImageTree(means, variances)),
        ]),
    ])
