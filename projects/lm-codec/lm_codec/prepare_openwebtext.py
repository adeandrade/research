import os
from collections.abc import Callable
from typing import Any

import datasets
import defopt
import numpy as np
import tiktoken
import tqdm
from datasets import DatasetDict


def create_tokenizer(encoding_name: str) -> Callable[[str], dict[str, Any]]:
    codec = tiktoken.get_encoding(encoding_name)

    def process(text: str):
        # `encode_ordinary` ignores any special tokens
        ids = codec.encode_ordinary(text)

        # add the end of text token, e.g. 50256 for gpt2 bpe
        ids.append(codec.eot_token)

        return {'ids': ids, 'length': len(ids)}

    return process


def create_dataset_openwebtext(
    num_proc: int,
    name: str = 'openwebtext',
    test_size: float = 0.0005,
    seed: int = 2357,
) -> DatasetDict:
    """
    Takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)

    Results in:
    DatasetDict({
        train: Dataset({
            features: ['text'],
            num_rows: 8009762
        })
        val: Dataset({
            features: ['text'],
            num_rows: 4007
        })
    })
    """
    dataset = datasets.load_dataset(name, num_proc=num_proc)
    assert isinstance(dataset, DatasetDict)

    # openwebtext by default only contains the 'train' split, so create a test split
    split_dataset = dataset['train'].train_test_split(test_size=test_size, seed=seed, shuffle=True)

    # rename the test split to val
    split_dataset['val'] = split_dataset.pop('test')

    return split_dataset


def prepare_dataset(
    path: str,
    *,
    name: str = 'openwebtext',
    encoding_name: str = 'gpt2',
    dtype: type = np.uint16,
    total_batches: int = 1024,
    num_proc: int = 8,
    num_proc_load_dataset: int = 8,
) -> None:
    """
    Splits and tokenizes a dataset and zips it into a single file.

    :param name: dataset name
    :param encoding_name: tokenizer type
    :param dtype: `encoder.max_token_value`. For the GPT-2 codec, it is 50256, which is < 2**16
    :param num_proc: number of workers in `map` call. Good number to use is half number of cpu cores
    :param num_proc_load_dataset: number of workers in `load_dataset`. Best number might be different from `num_proc`. It depends on network speed
    """
    if name == 'openwebtext':
        dataset = create_dataset_openwebtext(num_proc_load_dataset)

    else:
        raise ValueError(f'Dataset {name} not supported')

    tokenize = create_tokenizer(encoding_name)

    # tokenize the dataset
    tokenized = dataset.map(
        lambda example: tokenize(example['text']),
        remove_columns=['text'],
        desc='Tokenizing the splits',
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split_label, split in tokenized.items():
        length = np.sum(split['length'], dtype=np.uint64).item()
        filename = os.path.join(path, f'{split_label}.bin')
        array = np.memmap(filename, dtype=dtype, mode='w+', shape=(length,))

        index = 0

        for batch_index in tqdm.tqdm(range(total_batches), desc=f'Writing {filename}'):
            # batch together samples for faster write
            batch = split.shard(num_shards=total_batches, index=batch_index, contiguous=True).with_format('numpy')

            array_batch = np.concatenate(batch['ids'])

            # write into mmap
            array[index : index + len(array_batch)] = array_batch

            index += len(array_batch)

        array.flush()


def main():
    defopt.run(prepare_dataset, parsers={type: np.dtype})


if __name__ == '__main__':
    main()
