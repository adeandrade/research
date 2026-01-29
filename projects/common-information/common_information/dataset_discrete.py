import itertools
from gzip import GzipFile
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.utils.data import IterableDataset


class DiscreteTensors[T](IterableDataset[T]):
    channels = 2

    def __init__(
        self,
        path: str,
        transform: Callable[[tuple[Tensor, Tensor, Tensor]], T] = lambda x: x,
    ) -> None:
        self.transform = transform

        data = torch.load(GzipFile(path), map_location=torch.device('cuda:0'))  # type: ignore

        self.shape = data['shape']
        self.num_symbols = data['num_symbols']
        self.dependencies = data['dependencies']
        self.transform_a = data['transform_a']
        self.transform_b = data['transform_b']

        self.distribution = self.get_distribution(data['pmf'])
        self.num_dimensions = 2 * self.shape[0] * self.shape[1]
        self.split_index = self.shape[0] * self.shape[1]
        self.offset = self.num_symbols / 2
        self.indices = torch.arange(self.num_dimensions, dtype=torch.int64, device=self.dependencies.device)

    @staticmethod
    def get_distribution(pmf: Tensor) -> Categorical:
        return Categorical(torch.flatten(pmf))

    def indices_1d_to_2d(self, indices: Tensor) -> Tensor:
        return torch.stack((indices // (self.num_symbols + 1), indices % (self.num_symbols + 1)))

    def set_dependencies(self, samples: Tensor) -> Tensor:
        values = torch.clone(samples[0])

        indices_a = self.indices[self.indices > self.dependencies]
        indices_b = self.dependencies[indices_a]

        values[indices_b] = samples[1, indices_a]

        values = values.to(torch.float32)
        values = (values - self.offset) / self.offset

        return values

    def sample(self) -> T:
        inputs = self.distribution.sample([self.num_dimensions])
        inputs = self.indices_1d_to_2d(inputs)
        inputs = self.set_dependencies(inputs)

        target_a = self.transform_a @ inputs[: self.split_index]
        target_a = torch.reshape(target_a, self.shape)

        target_b = self.transform_b @ inputs[self.split_index :]
        target_b = torch.reshape(target_b, self.shape)

        inputs = torch.stack((
            torch.reshape(inputs[: self.split_index], self.shape),
            torch.reshape(inputs[self.split_index :], self.shape),
        ))

        outputs = self.transform((inputs, target_a, target_b))

        return outputs

    def __iter__(self) -> Iterable[T]:
        return (function() for function in itertools.repeat(self.sample))
