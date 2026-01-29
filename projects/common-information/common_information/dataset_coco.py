import contextlib
import json
import os
from io import StringIO
from typing import Callable, Sequence

import PIL.Image as pillow
import sfu_torch_lib.file_fetcher as file_fetcher_lib
from PIL.Image import Image
from pycocotools.coco import COCO as COCOHelper
from sfu_torch_lib.group_sampler import ImageDataset
from torch.utils.data import Dataset

from common_information.processing_coco import FilterAndRemapCategories, GenerateTargetSegmentation


class COCO[T](Dataset[T], ImageDataset):
    image_channels = 3
    num_classes_detection = 91
    num_classes_segmentation = FilterAndRemapCategories.num_classes
    ignore_index = GenerateTargetSegmentation.ignore_index
    num_classes_keypointing = 2

    split_to_path = {'train': 'train2017', 'validation': 'val2017'}

    def __init__(
        self,
        path: str,
        transform: Callable[[tuple[Image, tuple[list, int, int], tuple[list, int, int]]], T],
        split: str = 'train',
        categories: Sequence[int] | None = FilterAndRemapCategories.category_ids,
    ) -> None:
        self.transform = transform
        self.split = split
        self.categories = categories

        self.file_fetcher = file_fetcher_lib.get_file_fetcher(path, is_member=lambda _: False)
        self.coco_instances = self.get_coco(path, 'instances', split)
        self.coco_keypoints = self.get_coco(path, 'person_keypoints', split)
        self.image_ids = [image_id for image_id in sorted((self.coco_instances.imgs | self.coco_keypoints.imgs).keys())]

    @classmethod
    def get_coco(cls, path: str, mode: str, split: str) -> COCOHelper:
        file_fetcher = file_fetcher_lib.get_file_fetcher(path, is_member=lambda _: False)

        coco = COCOHelper()

        with file_fetcher.open_member(f'annotations/{mode}_{cls.split_to_path[split]}.json') as file:
            coco.dataset = json.load(file)

            with contextlib.redirect_stdout(StringIO()):
                coco.createIndex()

        return coco

    def get_height_and_width(self, index: int) -> tuple[int, int]:
        image_id = self.image_ids[index]

        image_info = (
            self.coco_instances.imgs[image_id]
            if image_id in self.coco_instances.imgs[image_id]
            else self.coco_keypoints.imgs[image_id]
        )

        height, width = image_info['height'], image_info['width']

        return height, width

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int) -> T:
        image_id = self.image_ids[index]

        path = os.path.join(self.split_to_path[self.split], self.coco_instances.loadImgs(image_id)[0]['file_name'])

        annotations_instances = self.coco_instances.loadAnns(self.coco_instances.getAnnIds(image_id))
        annotations_keypoints = self.coco_keypoints.loadAnns(self.coco_keypoints.getAnnIds(image_id))

        with self.file_fetcher.open_member(path) as image_file:
            image = pillow.open(image_file)

            outputs = self.transform((
                image,
                (annotations_instances, image.height, image.width),
                (annotations_keypoints, image.height, image.width),
            ))

            return outputs
