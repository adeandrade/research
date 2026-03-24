import os
import time
import zlib
from multiprocessing.pool import ThreadPool

import defopt
import mlflow
import numpy as np
import sfu_torch_lib.mlflow as mlflow_lib
import torch
from numpy.typing import DTypeLike
from nvidia.nvcomp import Codec  # type: ignore
from pytorch_lightning import LightningModule
from sfu_torch_lib import slack, state
from torch import Generator, Size, Tensor
from torch.utils.data import DataLoader, RandomSampler

from lm_codec import functions
from lm_codec.dataset_openwebtext import OpenWebText
from lm_codec.model_lm_codec import LMCodec


def code_cpu(inputs: Tensor, pool: ThreadPool) -> tuple[Tensor, int]:
    match inputs.dtype:
        case torch.int32:
            dtype = np.int32
        case torch.float16:
            dtype = np.float16
        case _:
            raise ValueError('Unsupported data type.')

    byte_strings = encode_cpu(inputs, pool)
    reconstructions = decode_cpu(byte_strings, inputs.shape, dtype, inputs.device, pool)

    num_bits = sum(8 * len(byte_string) for byte_string in byte_strings)

    return reconstructions, num_bits


def encode_cpu(inputs: Tensor, pool: ThreadPool) -> list[bytes]:
    inputs = torch.flatten(inputs, start_dim=0, end_dim=1)

    return list(
        pool.map(
            lambda sample: zlib.compress(sample.tobytes()),
            functions.to_numpy(inputs),
        )
    )


def decode_cpu(
    byte_strings: list[bytes],
    shape: Size,
    dtype: DTypeLike,
    device: torch.device,
    pool: ThreadPool,
) -> Tensor:
    arrays = list(
        pool.map(
            lambda byte_string: np.frombuffer(
                zlib.decompress(byte_string),
                dtype=dtype,
            ),
            byte_strings,
        )
    )

    tensor = np.stack(arrays)
    tensor = torch.tensor(tensor, device=device)
    return torch.reshape(tensor, shape)


def code_gpu(tensor: Tensor, codec: Codec) -> tuple[Tensor, int]:
    shape = tensor.shape
    dtype = tensor.dtype

    tensors = torch.flatten(tensor, start_dim=0, end_dim=1)
    tensors = torch.unbind(tensors, dim=0)
    tensors = codec.encode(tensors)

    num_bits = sum(8 * tensor.buffer_size for tensor in tensors)

    tensors = codec.decode(tensors)
    tensor = torch.stack([torch.as_tensor(tensor, dtype=dtype) for tensor in tensors], dim=0)
    tensor = torch.reshape(tensor, shape)

    return tensor, num_bits


@torch.no_grad
def calculate_time_complexity_and_rate(
    model: LightningModule,
    dataloader: DataLoader,
    gpu: bool,
) -> tuple[float, float]:
    quantize = isinstance(model, LMCodec)
    split_index = model.split_index if isinstance(model, LMCodec) else 3

    time_complexity = 0.0
    rate = 0.0

    with ThreadPool(os.process_cpu_count()) as pool:
        if gpu:
            codec = Codec(
                algorithm='GDeflate',
                data_type='<i4' if quantize else '<f2',
                cuda_stream=torch.cuda.current_stream().cuda_stream,
            )

            code = lambda x: code_gpu(x, codec)

        else:
            code = lambda x: code_cpu(x, pool)

        for index, inputs in enumerate(iter(dataloader)):
            inputs = inputs.cuda()

            batch_size, block_size, *_ = inputs.shape

            *_, (representations, *_) = model.forward(inputs, return_blocks={split_index}, quantize=quantize)

            representations = representations.to(torch.int32 if quantize else torch.float16)

            torch.cuda.synchronize()
            start_time = time.monotonic_ns()

            reconstructions, rate_batch = code(representations)

            torch.cuda.synchronize()
            end_time = time.monotonic_ns()

            torch.testing.assert_close(reconstructions, representations)

            if index == 0:
                continue

            time_complexity_batch = end_time - start_time
            time_complexity_batch /= batch_size * block_size
            time_complexity += (time_complexity_batch - time_complexity) / index

            rate_batch /= batch_size * block_size
            rate += (rate_batch - rate) / index

    return time_complexity, rate


def batch_transform(batch: tuple[Tensor, Tensor]) -> Tensor:
    inputs, *_ = batch

    return inputs


@slack.notify
@mlflow_lib.install
def analyze(
    *,
    run_id_pretrained: str,
    run_id: str | None = None,
    gpu: bool = False,
    dataset_type: str = 'openwebtext',
    num_documents: int = 1000,
    block_size: int = 1024,
    batch_size: int = 10,
    seed: int = 110069,
) -> None:
    """
    Measure the time complexity and rate of encoding and decoding the representations of a pretrained model using Deflate.

    :param run_id_pretrained: run ID of the training run
    :param run_id: run ID of the current run
    :param gpu: whether to use GPU for encoding and decoding
    :param dataset_type: dataset name
    :param num_documents: number of samples
    :param block_size: context size
    :param batch_size: batch size for dataset generation
    :param seed: random seed
    """
    model = state.load_model(run_id_pretrained, cache=True, overwrite=False)
    model = model.eval()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if dataset_type == 'openwebtext':
        dataset = OpenWebText('validation', block_size, transform=batch_transform)

    else:
        raise ValueError(f'Validation dataset {dataset_type} not supported.')

    generator = Generator().manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size,
        sampler=RandomSampler(dataset, False, num_documents, generator),
        pin_memory=True,
    )

    time_complexity, rate = calculate_time_complexity_and_rate(model, dataloader, gpu)

    mlflow.log_metrics({
        'Time Complexity': time_complexity,
        'Rate': rate,
    })


def main():
    defopt.run(analyze)


if __name__ == '__main__':
    main()
