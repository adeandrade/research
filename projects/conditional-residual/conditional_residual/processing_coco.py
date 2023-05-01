import copy
from typing import Dict, Tuple, Sequence, Callable, Any

import numpy as np
import torch
import torchvision.transforms.functional as functional
from pycocotools import mask as coco_mask
from pycocotools.mask import frPyObjects
from sfu_torch_lib.processing import ComposeTree
from sfu_torch_lib.processing import RandomHorizontalFlipTree, EncodeImageTree, ToTensorTree, ConvertImageTree
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import Compose


TargetType = Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]
BatchType = Tuple[Sequence[Tensor], Sequence[TargetType]]
AnnotationType = Sequence[Dict[str, Any]]


class FilterAndRemapCategories(Module):
    category_ids = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    num_classes = len(category_ids)

    @classmethod
    def transform(cls, height: int, width: int, annotations: AnnotationType) -> Tuple[int, int, AnnotationType]:
        annotations = [
            image_object for image_object in annotations
            if image_object['category_id'] in cls.category_ids
        ]

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


class RandomHorizontalFlipCoco(Module):
    def __init__(self, probability: float = .5) -> None:
        super().__init__()
        self.probability = probability

    def transform(
            self,
            image: Tensor,
            boxes: Tensor,
            mask: Tensor,
            segmentation: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        flip = torch.rand(1) < self.probability

        if flip:
            image = functional.hflip(image)
            mask = functional.hflip(mask)
            segmentation = functional.hflip(segmentation)

            _, _, width = functional.get_dimensions(image)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        return image, boxes, mask, segmentation

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
        mask = mask.any(dim=2)

        masks.append(mask)

    masks = torch.stack(masks, dim=0) if masks else torch.zeros((0, height, width), dtype=torch.uint8)

    return masks


def collate(batch):
    return tuple(zip(*batch))


def make_divisible(x: int, divisor: int) -> int:
    return x // divisor * divisor


def crop(tensors: Sequence[Tensor], divisor: int = 64) -> Tensor:
    min_height = make_divisible(min(tensor.shape[1] for tensor in tensors), divisor)
    min_width = make_divisible(min(tensor.shape[2] for tensor in tensors), divisor)

    batch = [functional.center_crop(tensor, output_size=(min_height, min_width)) for tensor in tensors]
    batch = torch.stack(batch, dim=0)

    return batch


def create_train_transformer() -> Callable:
    return ComposeTree([
        ([0], ConvertImageTree('RGB')),
        ([1], lambda annotations: [(annotations[0], annotations[0])]),
        ([[1, 0]], GenerateTargetDetection()),
        ([[1, 1]], FilterAndRemapCategories()),
        ([[1, 1]], GenerateTargetSegmentation()),
        ([0, [1, 0, 0], [1, 0, 2], [1, 1]], RandomHorizontalFlipCoco()),
        ([0], ToTensorTree(np.float32)),
        ([0], EncodeImageTree()),
    ])


def create_input_train_transformer() -> Callable:
    return Compose([
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0], RandomHorizontalFlipTree()),
            ([0], ToTensorTree(np.float32)),
            ([0], EncodeImageTree()),
        ]),
        lambda inputs: inputs[0],
    ])


def create_test_transformer() -> Callable:
    return ComposeTree([
        ([0], ConvertImageTree('RGB')),
        ([1], lambda annotations: [(annotations[0], annotations[0])]),
        ([[1, 0]], GenerateTargetDetection()),
        ([[1, 1]], FilterAndRemapCategories()),
        ([[1, 1]], GenerateTargetSegmentation()),
        ([0], ToTensorTree(np.float32)),
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
