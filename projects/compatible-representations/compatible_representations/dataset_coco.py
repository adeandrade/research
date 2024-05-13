import contextlib
import json
import os
from io import StringIO
from typing import Callable, Set, List, Optional, Sequence, Tuple, TypeVar

import PIL.Image as pillow
import sfu_torch_lib.file_fetcher as file_fetcher_lib
from PIL.Image import Image
from pycocotools.coco import COCO as COCOHelper
from sfu_torch_lib.group_sampler import ImageDataset
from torch.utils.data import Dataset

from compatible_representations.processing_coco import (
    AnnotationType,
    FilterAndRemapCategories,
    GenerateTargetSegmentation,
)


OutputType = TypeVar('OutputType')


class COCO(Dataset[OutputType], ImageDataset):
    image_channels = 3
    num_classes_detection = 91
    num_classes_segmentation = FilterAndRemapCategories.num_classes
    ignore_index = GenerateTargetSegmentation.ignore_index

    split_to_path = {'train': 'train2017', 'validation': 'val2017'}

    def __init__(
        self,
        path: str,
        transform: Callable[[Tuple[Image, Tuple[int, int, AnnotationType]]], OutputType],
        minimum_image_size: Optional[Tuple[int, int]] = None,
        split: str = 'train',
        mode: str = 'instances',
        categories: Optional[Sequence[int]] = FilterAndRemapCategories.category_ids,
        min_area: int = 1000,
    ) -> None:
        self.transform = transform
        self.minimum_image_size = minimum_image_size
        self.split = split
        self.mode = mode
        self.categories = categories
        self.min_area = min_area

        self.file_fetcher = file_fetcher_lib.get_file_fetcher(path, is_member=lambda _: False)
        self.coco = self.get_coco(path, mode, split)
        self.image_ids = self.get_image_ids(self.coco, minimum_image_size, categories, min_area)

    @classmethod
    def get_coco(cls, path: str, mode: str, split: str) -> COCOHelper:
        file_fetcher = file_fetcher_lib.get_file_fetcher(path, is_member=lambda _: False)

        coco = COCOHelper()

        with file_fetcher.open_member(f'annotations/{mode}_{cls.split_to_path[split]}.json') as file:
            coco.dataset = json.load(file)

            with contextlib.redirect_stdout(StringIO()):
                coco.createIndex()

        return coco

    @classmethod
    def has_valid_size(cls, coco: COCOHelper, image_id: int, minimum_image_size: Optional[Tuple[int, int]]) -> bool:
        if minimum_image_size is None:
            return True

        height_minimum, width_minimum = minimum_image_size

        image_info = coco.imgs[image_id]
        height, width = image_info['height'], image_info['width']

        if height < height_minimum or width < width_minimum:
            return False

        return True

    @staticmethod
    def has_only_empty_bbox(annotations: AnnotationType) -> bool:
        return all(any(size <= 1 for size in image_object['bbox'][2:]) for image_object in annotations)

    @classmethod
    def has_valid_annotations(
        cls,
        coco: COCOHelper,
        image_id: int,
        categories: Optional[Set[int]],
        min_area: int,
    ) -> bool:
        """
        If it is empty, there are no annotations.
        If all boxes have close to zero area, there are no annotations.
        If more than `min_area` pixels occupied in the image.
        """
        annotations = coco.loadAnns(coco.getAnnIds(image_id))

        if categories:
            annotations = [image_object for image_object in annotations if image_object['category_id'] in categories]

        if len(annotations) == 0:
            return False
        elif cls.has_only_empty_bbox(annotations):
            return False
        elif sum(image_object['area'] for image_object in annotations) <= min_area:
            return False
        else:
            return True

    @classmethod
    def get_image_ids(
        cls,
        coco: COCOHelper,
        minimum_image_size: Optional[Tuple[int, int]],
        categories: Optional[Sequence[int]],
        min_area: int,
    ) -> List[int]:
        image_ids = []

        category_set = set(categories) if categories is not None else None

        for image_id in sorted(coco.imgs.keys()):
            is_valid = cls.has_valid_size(coco, image_id, minimum_image_size) and cls.has_valid_annotations(
                coco, image_id, category_set, min_area
            )

            if is_valid:
                image_ids.append(image_id)

        return image_ids

    def get_height_and_width(self, index: int) -> Tuple[int, int]:
        image_id = self.image_ids[index]

        image_info = self.coco.imgs[image_id]

        height, width = image_info['height'], image_info['width']

        return height, width

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int) -> OutputType:
        image_id = self.image_ids[index]

        annotations = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        path = os.path.join(self.split_to_path[self.split], self.coco.loadImgs(image_id)[0]['file_name'])

        with self.file_fetcher.open_member(path) as image_file:
            image = pillow.open(image_file)

            outputs = self.transform((image, (image.height, image.width, annotations)))

            return outputs
