from collections.abc import Callable

import numpy as np
import sfu_torch_lib.tree as tree_lib
import torch
from PIL.Image import Image
from sfu_torch_lib import processing
from sfu_torch_lib.processing import ComposeTree, ConvertImageTree, ToTensorTree
from torch import Tensor
from torch.nn import Module
from torch.utils.data import dataloader
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision.transforms import RandAugment
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import CutMix, MixUp, RandomChoice

from image_codec.dataset_imagenet import Imagenet


class ViTTransformTree(Module):
    def __init__(self) -> None:
        super().__init__()
        self.transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    def forward(self, tree):
        return tree_lib.map_tree(self.transform, tree)


class ResNetTransformTree(Module):
    def __init__(self) -> None:
        super().__init__()
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def forward(self, tree):
        return tree_lib.map_tree(self.transform, tree)


class PermuteChannelTree(Module):
    def forward(self, tree):
        return tree_lib.map_tree(lambda x: torch.permute(x, (2, 0, 1)), tree)


class RandAugmentTree(Module):
    def __init__(
        self,
        magnitude: int = 9,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.transform = RandAugment(magnitude=magnitude, interpolation=interpolation)

    def forward(self, tree):
        return tree_lib.map_tree(self.transform, tree)


def get_mixup_cutmix(mixup_alpha: float, cutmix_alpha: float, num_classes: int):
    mixup_cutmix = []

    if mixup_alpha > 0:
        mixup_cutmix.append(MixUp(alpha=mixup_alpha, num_classes=num_classes))

    if cutmix_alpha > 0:
        mixup_cutmix.append(CutMix(alpha=cutmix_alpha, num_classes=num_classes))

    return RandomChoice(mixup_cutmix)


def collate_mixup(mixup_alpha: float, cutmix_alpha: float):
    mixup_cutmix = get_mixup_cutmix(mixup_alpha, cutmix_alpha, Imagenet.num_classes)

    def function(batch):
        return mixup_cutmix(*dataloader.default_collate(batch))

    return function


@processing.with_sequence
def create_detection_train_transform() -> Callable[[Image, int], tuple[Tensor, Tensor]]:
    return ComposeTree([
        ([0], ConvertImageTree('RGB')),
        ([0, 1], ToTensorTree(np.uint8, np.int64)),
        ([0], PermuteChannelTree()),
        ([0], RandAugmentTree()),
        ([0], ViTTransformTree()),
    ])


@processing.with_sequence
def create_detection_test_transform() -> Callable[[Image, int], tuple[Tensor, Tensor]]:
    return ComposeTree([
        ([0], ConvertImageTree('RGB')),
        ([0, 1], ToTensorTree(np.uint8, np.int64)),
        ([0], PermuteChannelTree()),
        ([0], ViTTransformTree()),
    ])
