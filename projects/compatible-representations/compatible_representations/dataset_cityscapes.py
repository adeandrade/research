import os
from typing import Callable, Tuple, TypeVar

import PIL.Image as pillow
import sfu_torch_lib.file_fetcher as file_fetcher
from PIL.Image import Image
from torch.utils.data import Dataset

from compatible_representations.processing_cityscapes import EncodeSegmentationTree

OutputType = TypeVar('OutputType')


class Cityscapes(Dataset[OutputType]):
    num_classes = EncodeSegmentationTree.num_classes
    ignore_index = EncodeSegmentationTree.ignore_index

    image_channels = 3
    depth_input_channels = 2
    depth_output_channels = 1

    split_to_directory = {
        'train': 'train',
        'validation': 'val',
        'test': 'test',
    }

    def __init__(
        self,
        path: str,
        transform: Callable[[Tuple[Image, Image, Image]], OutputType],
        split: str = 'train',
    ) -> None:
        self.transform = transform
        self.split = split

        self.file_fetcher = file_fetcher.get_file_fetcher(path, self.is_member)

    def is_member(self, member: str) -> bool:
        prefix = os.path.join('leftImg8bit', self.split_to_directory[self.split])
        return member.startswith(prefix) and member.endswith('.png')

    def get_path(self, directory: str, index: int, suffix: str) -> str:
        image_path = self.file_fetcher[index]

        city = image_path.split(os.sep)[-2]
        prefix = os.path.basename(image_path)[:-15]

        path = os.path.join(directory, self.split_to_directory[self.split], city, f'{prefix}{suffix}')

        return path

    def __len__(self):
        return len(self.file_fetcher)

    def __getitem__(self, index: int) -> OutputType:
        image_name = self.file_fetcher[index]
        labels_name = self.get_path('gtFine', index, 'gtFine_labelIds.png')
        depth_name = self.get_path('disparity', index, 'disparity.png')

        with self.file_fetcher.open_member(image_name) as image_file, self.file_fetcher.open_member(
            labels_name
        ) as labels_file, self.file_fetcher.open_member(depth_name) as depth_file:
            image = pillow.open(image_file)
            labels = pillow.open(labels_file)
            depth = pillow.open(depth_file)

            outputs = self.transform((image, labels, depth))

            return outputs
