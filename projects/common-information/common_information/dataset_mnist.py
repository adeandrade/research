from typing import Callable
from zipfile import ZipFile

import PIL.Image as pillow
import PIL.ImageOps as image_ops
import torch
import torchvision.datasets.mnist as mnist
from PIL.Image import Image
from torch.distributions import Categorical
from torch.utils.data import Dataset


class MNISTColored[T](Dataset[T]):
    image_channels = 3
    num_classes_digit = 10
    num_classes_color = 10

    colors = [
        'red',
        'orange',
        'yellow',
        'lime',
        'green',
        'cyan',
        'blue',
        'purple',
        'magenta',
        'grey',
    ]

    pmfs = {
        'dependent': torch.eye(10, dtype=torch.float32) / 10,
        'independent': torch.ones(10, 10, dtype=torch.float32) / 100,
        'mixture': torch.tensor(
            [
                [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025],
            ],
            dtype=torch.float32,
        ),
    }

    def __init__(
        self,
        path: str,
        pmf: str,
        split: str = 'train',
        transform: Callable[[tuple[Image, int, int]], T] = lambda x: x,
    ) -> None:
        super().__init__()

        self.transform = transform

        self.pmf_conditional = self.pmfs[pmf] / torch.sum(self.pmfs[pmf], dim=1, keepdim=True)
        self.data, self.targets = self.load_data(path, split)

    def load_data(self, path: str, split: str):
        container = ZipFile(path)

        image_path = f'{"train" if split == "train" else "t10k"}-images.idx3-ubyte'
        image_file = container.extract(image_path)
        data = mnist.read_image_file(image_file)  # type: ignore

        label_path = f'{"train" if split == "train" else "t10k"}-labels.idx1-ubyte'
        label_file = container.extract(label_path)
        targets = mnist.read_label_file(label_file)  # type: ignore

        return data, targets

    def __getitem__(self, index: int) -> T:
        image, target_digit = self.data[index], int(self.targets[index])

        color_distribution = Categorical(self.pmf_conditional[target_digit])
        target_color = int(color_distribution.sample().item())

        image = image.numpy()
        image = pillow.fromarray(image, mode='L')
        image = image_ops.colorize(image, black='black', white=self.colors[target_color])

        outputs = self.transform((image, target_digit, target_color))

        return outputs

    def __len__(self) -> int:
        return len(self.data)
