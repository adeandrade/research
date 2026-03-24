import os
import random
import tempfile
from collections.abc import Callable
from random import Random
from tarfile import TarFile
from typing import ClassVar

import numpy as np
import tiktoken
import torch
from sfu_torch_lib import io
from torch import Tensor
from torch.utils.data import Dataset

from litgpt.tokenizer import Tokenizer
from lm_codec import functions


class OpenWebText[T](Dataset[T]):
    split_to_file: ClassVar[dict[str, str]] = {'train': 'train', 'validation': 'val'}
    split_to_num_documents: ClassVar[dict[str, int]] = {'train': 8009762, 'validation': 4007}

    def __init__(
        self,
        split: str,
        block_size: int = 1024,
        num_documents: int = 8013769,
        transform: Callable[[tuple[Tensor, Tensor]], T] = lambda x: x,
        path: str = 's3://datasets/openwebtext.tar',
        seed: int = 110069,
    ) -> None:
        path = io.localize_dataset(path)

        self.split = split
        self.block_size = block_size
        self.num_documents = num_documents or self.split_to_num_documents[split]
        self.transform = transform
        self.seed = seed

        self.data = self.load_data(path, split)

    @classmethod
    def load_data(cls, path: str, split: str) -> np.memmap:
        path_extracted = tempfile.gettempdir()
        member = f'{cls.split_to_file[split]}.bin'

        data = TarFile(path)
        data = data.extract(member, path_extracted)

        return np.memmap(os.path.join(path_extracted, member), dtype=np.uint16, mode='r')

    def __len__(self):
        return self.num_documents

    def __getitem__(self, index: int) -> T:
        prng = random if self.split == 'train' else Random(self.seed + index)
        index = prng.randrange(len(self.data) - self.block_size)

        inputs = torch.tensor(self.data[index : index + self.block_size], dtype=torch.int64)
        targets = torch.tensor(self.data[index + 1 : index + self.block_size + 1], dtype=torch.int64)

        return self.transform((inputs, targets))


class OpenWebTextReTokenize[T](Dataset[T]):
    split_to_file: ClassVar[dict[str, str]] = {'train': 'train', 'validation': 'val'}
    split_to_num_documents: ClassVar[dict[str, int]] = {'train': 8009762, 'validation': 4007}

    def __init__(
        self,
        tokenizer: Tokenizer,
        split: str,
        block_size: int = 1024,
        num_documents: int | None = None,
        transform: Callable[[tuple[Tensor, Tensor]], T] = lambda x: x,
        path: str = 's3://datasets/openwebtext.tar',
        seed: int = 110069,
        block_size_multiplier: float = 5.0,
    ) -> None:
        path = io.localize_dataset(path)

        self.split = split
        self.num_documents = num_documents or self.split_to_num_documents[split]
        self.transform = transform
        self.seed = seed

        self.block_size_source = round(block_size * block_size_multiplier)
        self.block_size_target = block_size

        self.tokenizer_source = tiktoken.get_encoding('gpt2')
        self.tokenizer_target = tokenizer

        self.data = OpenWebText.load_data(path, split)

    def __len__(self):
        return self.num_documents

    def retokenize(self, ids: list[int]) -> Tensor:
        texts = [
            self.tokenizer_source.decode(text_ids)
            for text_ids in functions.split_list(ids, self.tokenizer_source.eot_token)
        ]

        outputs = [self.tokenizer_target.encode(text, eos=True) for text in texts]

        return torch.concatenate(outputs)

    def __getitem__(self, index: int) -> T:
        prng = random if self.split == 'train' else Random(self.seed + index)
        index = prng.randrange(len(self.data) - self.block_size_source)

        ids = self.data[index : index + self.block_size_source]
        ids = self.retokenize(ids.tolist())

        assert len(ids) >= self.block_size_target

        inputs = ids[: self.block_size_target]
        targets = ids[1 : self.block_size_target + 1]

        return self.transform((inputs, targets))
