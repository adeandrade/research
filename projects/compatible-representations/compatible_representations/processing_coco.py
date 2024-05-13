import copy
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import sfu_torch_lib.tree as tree_lib
import torch
import torchvision.transforms.functional as functional
from pycocotools import mask as coco_mask
from pycocotools.mask import frPyObjects
from sfu_torch_lib.processing import (
    ComposeTree,
    ConvertImageTree,
    EncodeImageTree,
    RandomCropTree,
    RandomHorizontalFlipTree,
    ToTensorTree,
)
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import Compose

AnnotationType = Sequence[Dict[str, Any]]


class IdentityTree(Module):
    def forward(self, tree):
        return tree_lib.map_tree(lambda x: x, tree)


class FilterAndRemapCategories(Module):
    category_ids = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    num_classes = len(category_ids)

    @classmethod
    def transform(cls, height: int, width: int, annotations: AnnotationType) -> Tuple[int, int, AnnotationType]:
        annotations = [image_object for image_object in annotations if image_object['category_id'] in cls.category_ids]

        annotations = copy.deepcopy(annotations)

        for annotation in annotations:
            annotation['category_id'] = cls.category_ids.index(annotation['category_id'])

        return height, width, annotations

    @classmethod
    def forward(cls, tree):
        return list(map(lambda args: cls.transform(*args), tree))


class GenerateTargetDetection(Module):
    @staticmethod
    def transform(height: int, width: int, annotations: AnnotationType) -> Tuple[Tensor, Tensor, Tensor]:
        annotations = [image_object for image_object in annotations if image_object['iscrowd'] == 0]

        boxes = [image_object['bbox'] for image_object in annotations]

        labels = [image_object['category_id'] for image_object in annotations]
        labels = torch.tensor(labels, dtype=torch.int64)

        segmentations = [image_object['segmentation'] for image_object in annotations]
        masks = convert_polygons_to_mask(segmentations, height, width)

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        masks = masks[keep]

        return boxes, labels, masks

    @classmethod
    def forward(cls, tree):
        return list(map(lambda args: cls.transform(*args), tree))


class GenerateTargetSegmentation(Module):
    ignore_index = 255

    @classmethod
    def transform(cls, height: int, width: int, annotations: AnnotationType) -> Tensor:
        segmentations = [image_object['segmentation'] for image_object in annotations]
        labels = [image_object['category_id'] for image_object in annotations]

        if segmentations:
            masks = convert_polygons_to_mask(segmentations, height, width)
            classes = torch.as_tensor(labels, dtype=masks.dtype)

            # merge all instance masks into a single segmentation map with its corresponding categories
            target = torch.amax(masks * classes[:, None, None], dim=0)

            # discard overlapping instances
            target[torch.sum(masks, dim=0) > 1] = cls.ignore_index

        else:
            target = torch.zeros((height, width), dtype=torch.uint8)

        return target

    @classmethod
    def forward(cls, tree):
        return list(map(lambda args: cls.transform(*args), tree))


class RandomHorizontalFlipDetection(Module):
    def __init__(self, probability: float = 0.5) -> None:
        super().__init__()
        self.probability = probability

    def transform(
        self,
        image: Tensor,
        boxes: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        flip = torch.rand(1) < self.probability

        if flip:
            image = functional.hflip(image)
            mask = functional.hflip(mask)

            _, _, width = functional.get_dimensions(image)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        return image, boxes, mask

    def forward(self, args):
        return list(self.transform(*args))


def convert_polygons_to_mask(segmentations: Sequence[frPyObjects], height: int, width: int) -> Tensor:
    masks = []

    for polygons in segmentations:
        rles = frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)

        if len(mask.shape) < 3:
            mask = mask[..., None]

        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = torch.any(mask, dim=2)

        masks.append(mask)

    masks = torch.stack(masks, dim=0) if masks else torch.zeros((0, height, width), dtype=torch.uint8)

    return masks


def collate(batch):
    return tuple(zip(*batch))


def identity(batch):
    return batch


def make_divisible(x: int, divisor: int) -> int:
    return math.ceil(x / divisor) * divisor


def pad_input(tensors: Sequence[Tensor], divisor: int = 64, fill: float = 0.0) -> Tuple[Tensor, Tensor]:
    max_height = make_divisible(max(tensor.shape[-2] for tensor in tensors), divisor)
    max_width = make_divisible(max(tensor.shape[-1] for tensor in tensors), divisor)

    batch = [
        functional.pad(
            img=tensor,
            padding=[0, 0, max_width - tensor.shape[-1], max_height - tensor.shape[-2]],
            fill=fill,
        )
        for tensor in tensors
    ]
    batch = torch.stack(batch, dim=0)

    masks = (torch.ones_like(tensor[0]) for tensor in tensors)
    masks = [
        functional.pad(
            img=mask,
            padding=[0, 0, max_width - mask.shape[-1], max_height - mask.shape[-2]],
            fill=0.0,
        )
        for mask in masks
    ]
    masks = torch.stack(masks, dim=0)

    return batch, masks


def pad_segmentation(
    tensors: Sequence[Tuple[Tensor, Tensor]],
    divisor: int = 64,
    fill_image: float = 0.0,
    fill_target: int = GenerateTargetSegmentation.ignore_index,
) -> Tuple[Tensor, Tensor, Tensor]:
    images, targets = zip(*tensors)

    images, masks = pad_input(images, divisor, fill_image)

    max_height, max_width = images.shape

    targets = [
        functional.pad(
            img=tensor,
            padding=[0, 0, max_width - tensor.shape[-1], max_height - tensor.shape[-2]],
            fill=fill_target,
        )
        for tensor in targets
    ]
    targets = torch.stack(targets, dim=0)

    return images, targets, masks


def create_detection_train_transformer() -> Callable:
    return ComposeTree([
        ([0], ConvertImageTree('RGB')),
        ([1], GenerateTargetDetection()),
        ([0, [1, 0], [1, 2]], RandomHorizontalFlipDetection()),
        ([0], ToTensorTree(np.float32)),
        ([0], EncodeImageTree()),
    ])


def create_segmentation_train_transformer() -> Callable:
    return ComposeTree([
        ([0], ConvertImageTree('RGB')),
        ([1], FilterAndRemapCategories()),
        ([1], GenerateTargetSegmentation()),
        ([0, 1], RandomHorizontalFlipTree()),
        ([0, 1], ToTensorTree(np.float32, np.int64)),
        ([0], EncodeImageTree()),
    ])


def create_input_train_transformer(size: Optional[Tuple[int, int]] = None) -> Callable:
    return Compose([
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0], RandomCropTree(size) if size is not None else IdentityTree()),
            ([0], RandomHorizontalFlipTree()),
            ([0], ToTensorTree(np.float32)),
            ([0], EncodeImageTree()),
        ]),
        lambda inputs: inputs[0],
    ])


def create_detection_test_transformer() -> Callable:
    return ComposeTree([
        ([0], ConvertImageTree('RGB')),
        ([1], GenerateTargetDetection()),
        ([0], ToTensorTree(np.float32)),
        ([0], EncodeImageTree()),
    ])


def create_segmentation_test_transformer() -> Callable:
    return ComposeTree([
        ([0], ConvertImageTree('RGB')),
        ([1], FilterAndRemapCategories()),
        ([1], GenerateTargetSegmentation()),
        ([0, 1], ToTensorTree(np.float32, np.int64)),
        ([0], EncodeImageTree()),
    ])


def create_input_test_transformer() -> Callable:
    return Compose([
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0], ToTensorTree(np.float32)),
            ([0], EncodeImageTree()),
        ]),
        lambda inputs: inputs[0],
    ])
