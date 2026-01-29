import copy
import math
from typing import Callable, Sequence

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
from torch import Size, Tensor
from torch.nn import Module
from torchvision.transforms import Compose


class IdentityTree(Module):
    def forward(self, tree):
        return tree_lib.map_tree(lambda x: x, tree)


class FilterAndRemapCategories(Module):
    category_ids = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    num_classes = len(category_ids)

    @classmethod
    def transform(cls, height: int, width: int, annotations: list) -> tuple[int, int, list]:
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
    def transform(annotations: list, height: int, width: int) -> tuple[Tensor, Tensor, Tensor]:
        annotations = [image_object for image_object in annotations if image_object['iscrowd'] == 0]

        boxes = [image_object['bbox'] for image_object in annotations]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # guard against no boxes via resizing
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        labels = [image_object['category_id'] for image_object in annotations]
        labels = torch.tensor(labels, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]

        areas = torch.tensor([image_object['area'] for image_object in annotations])

        return boxes, labels, areas

    @classmethod
    def forward(cls, tree):
        return list(map(lambda args: cls.transform(*args), tree))


class DetectionsToMap(Module):
    @classmethod
    def forward(cls, tree):
        return list(
            map(
                lambda x: {'boxes': x[0], 'labels': x[1], 'area': x[2]},
                tree,
            )
        )


class GenerateTargetSegmentation(Module):
    ignore_index = 255

    @staticmethod
    def convert_polygons_to_mask(segmentations: Sequence[frPyObjects], height: int, width: int) -> Tensor:  # type: ignore
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

    @classmethod
    def transform(cls, annotations: list, height: int, width: int) -> Tensor:
        segmentations = [image_object['segmentation'] for image_object in annotations]
        labels = [image_object['category_id'] for image_object in annotations]

        if segmentations:
            masks = cls.convert_polygons_to_mask(segmentations, height, width)
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


class GenerateTargetKeypointing(Module):
    @staticmethod
    def transform(annotations: list, height: int, width: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        annotations = [image_object for image_object in annotations if image_object['iscrowd'] == 0]

        boxes = [image_object['bbox'] for image_object in annotations]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # guard against no boxes via resizing
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        boxes = boxes[keep]

        labels = [image_object['category_id'] for image_object in annotations]
        labels = torch.tensor(labels, dtype=torch.int64)
        labels = labels[keep]

        keypoints = [image_object['keypoints'] for image_object in annotations]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        num_keypoints = keypoints.shape[0]
        if num_keypoints > 0:
            keypoints = torch.reshape(keypoints, (num_keypoints, -1, 3))
            keypoints = keypoints[keep]
        else:
            keypoints = torch.empty((0, 0, 3), dtype=torch.float32)

        areas = torch.tensor([image_object['area'] for image_object in annotations])

        return boxes, labels, keypoints, areas

    @classmethod
    def forward(cls, tree):
        return list(map(lambda args: cls.transform(*args), tree))


class KeypointsToMap(Module):
    @classmethod
    def forward(cls, tree):
        return list(
            map(
                lambda x: {'boxes': x[0], 'labels': x[1], 'keypoints': x[2], 'area': x[3]},
                tree,
            )
        )


class RandomHorizontalFlipTargets(Module):
    def __init__(self, probability: float = 0.5) -> None:
        super().__init__()
        self.probability = probability

    @staticmethod
    def flip_keypoints(keypoints: Tensor, width: int) -> Tensor:
        if keypoints.shape[0] == 0:
            return keypoints

        indices = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

        flipped = keypoints[:, indices]
        flipped[..., 0] = width - flipped[..., 0]

        # maintain COCO convention that if visibility == 0, then x, y = 0
        flipped[flipped[..., 2] == 0] = 0

        return flipped

    def transform(
        self,
        image: Tensor,
        boxes_detection: Tensor,
        mask: Tensor,
        boxes_keypoints: Tensor,
        keypoints: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        flip = torch.rand(1) < self.probability

        if flip:
            _, _, width = functional.get_dimensions(image)

            image = functional.hflip(image)
            mask = functional.hflip(mask)

            boxes_detection[:, [0, 2]] = width - boxes_detection[:, [2, 0]]
            boxes_keypoints[:, [0, 2]] = width - boxes_keypoints[:, [2, 0]]

            keypoints = self.flip_keypoints(keypoints, width)

        return image, boxes_detection, mask, boxes_keypoints, keypoints

    def forward(self, args):
        return list(self.transform(*args))


def make_divisible(x: int, divisor: int) -> int:
    return math.ceil(x / divisor) * divisor


def pad_inputs(tensors: Sequence[Tensor], divisor: int = 64, fill: float = 0.0) -> tuple[Tensor, Tensor]:
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

    masks = [
        functional.pad(
            img=torch.ones_like(tensor[0]),
            padding=[0, 0, max_width - tensor.shape[-1], max_height - tensor.shape[-2]],
            fill=0.0,
        )
        for tensor in tensors
    ]
    masks = torch.stack(masks, dim=0)

    return batch, masks


def pad_targets(targets: Sequence[Tensor], shape: Size, fill: float) -> Tensor:
    max_height, max_width = shape

    targets = [
        functional.pad(
            img=tensor,
            padding=[0, 0, max_width - tensor.shape[-1], max_height - tensor.shape[-2]],
            fill=fill,
        )
        for tensor in targets
    ]

    tensor = torch.stack(targets, dim=0)

    return tensor


def collate(batch):
    return tuple(zip(*batch))


def identity(batch):
    return batch


def collate_detection(
    batch: Sequence[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[tuple[Tensor, Tensor], Sequence[dict[str, Tensor]]]:
    inputs, targets = collate(batch)

    inputs = pad_inputs(inputs)

    return inputs, targets


def collate_segmentation(batch: Sequence[tuple[Tensor, Tensor]]) -> tuple[tuple[Tensor, Tensor], Tensor]:
    inputs, targets = collate(batch)

    images, masks = pad_inputs(inputs)

    targets = pad_targets(targets, images.shape, GenerateTargetSegmentation.ignore_index)

    return (images, masks), targets


def collate_keypointing(
    batch: Sequence[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[tuple[Tensor, Tensor], Sequence[dict[str, Tensor]]]:
    inputs, targets = collate(batch)

    inputs = pad_inputs(inputs)

    return inputs, targets


def collate_joint(
    batch: Sequence[tuple[Tensor, dict[str, Tensor], Tensor, dict[str, Tensor]]],
) -> tuple[tuple[Tensor, Tensor], Sequence[dict[str, Tensor]], Tensor, Sequence[dict[str, Tensor]]]:
    images, targets_detection, targets_segmentation, targets_keypointing = collate(batch)

    images, masks = pad_inputs(images)

    targets_segmentation = pad_targets(targets_segmentation, images.shape[2:], GenerateTargetSegmentation.ignore_index)

    return (images, masks), targets_detection, targets_segmentation, targets_keypointing


def create_joint_train_transformer() -> Callable:
    return Compose([
        lambda x: (x[0], x[1], x[1], x[2]),
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([1], GenerateTargetDetection()),
            ([2], GenerateTargetSegmentation()),
            ([3], GenerateTargetKeypointing()),
            ([0, [1, 0], 2, [3, 0], [3, 2]], RandomHorizontalFlipTargets()),
            ([0, 2], ToTensorTree(np.float32, np.int64)),  # type: ignore
            ([0], EncodeImageTree()),
            ([1], DetectionsToMap()),
            ([3], KeypointsToMap()),
        ]),
    ])


def create_joint_test_transformer() -> Callable:
    return Compose([
        lambda x: (x[0], x[1], x[1], x[2]),
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([1], GenerateTargetDetection()),
            ([2], GenerateTargetSegmentation()),
            ([3], GenerateTargetKeypointing()),
            ([0, 2], ToTensorTree(np.float32, np.int64)),  # type: ignore
            ([0], EncodeImageTree()),
            ([1], DetectionsToMap()),
            ([3], KeypointsToMap()),
        ]),
    ])


def create_detection_train_transformer() -> Callable:
    return Compose([create_joint_train_transformer(), lambda x: (x[0], x[1])])


def create_detection_test_transformer() -> Callable:
    return Compose([create_joint_test_transformer(), lambda x: (x[0], x[1])])


def create_segmentation_train_transformer() -> Callable:
    return Compose([create_joint_train_transformer(), lambda x: (x[0], x[2])])


def create_segmentation_test_transformer() -> Callable:
    return Compose([create_joint_test_transformer(), lambda x: (x[0], x[2])])


def create_keypoint_train_transformer() -> Callable:
    return Compose([create_joint_train_transformer(), lambda x: (x[0], x[3])])


def create_keypoint_test_transformer() -> Callable:
    return Compose([create_joint_test_transformer(), lambda x: (x[0], x[3])])


def create_input_train_transformer(size: tuple[int, int] | None = None) -> Callable:
    return Compose([
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0], RandomCropTree(size) if size is not None else IdentityTree()),
            ([0], RandomHorizontalFlipTree()),
            ([0], ToTensorTree(np.float32)),  # type: ignore
            ([0], EncodeImageTree()),
        ]),
        lambda x: x[0],
    ])


def create_input_test_transformer() -> Callable:
    return Compose([
        ComposeTree([
            ([0], ConvertImageTree('RGB')),
            ([0], ToTensorTree(np.float32)),  # type: ignore
            ([0], EncodeImageTree()),
        ]),
        lambda x: x[0],
    ])
