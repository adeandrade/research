import time

import defopt
import mlflow
import sfu_torch_lib.mlflow as mlflow_lib
import torch
from sfu_torch_lib import slack, state
from torch import Generator, Size, Tensor
from torch.utils.data import DataLoader, RandomSampler

from lm_codec.dataset_openwebtext import OpenWebText
from lm_codec.model_lm_codec import LMCodec


def code_cpu(model: LMCodec, inputs: Tensor) -> tuple[Tensor, int]:
    batch_size, block_size, *_ = inputs.shape

    byte_strings_representation, byte_strings_hyper_prior = encode_cpu(model, inputs)

    reconstruction, _ = decode_cpu(
        model,
        byte_strings_representation,
        byte_strings_hyper_prior,
        batch_size,
        block_size,
    )

    bpt_representation = sum(8 * len(byte_string) for byte_string in byte_strings_representation)
    bpt_prior = sum(8 * len(byte_string) for byte_string in byte_strings_hyper_prior)
    bpt = bpt_representation + bpt_prior

    return reconstruction, bpt


def code_cpu_profiled(model: LMCodec, inputs: Tensor) -> tuple[Tensor, int, int]:
    batch_size, block_size, *_ = inputs.shape

    byte_strings_representation, byte_strings_hyper_prior, elapsed_encode = encode_cpu_profiled(model, inputs)

    reconstruction, _, elapsed_decode = decode_cpu_profiled(
        model,
        byte_strings_representation,
        byte_strings_hyper_prior,
        batch_size,
        block_size,
    )

    bpt_representation = sum(8 * len(byte_string) for byte_string in byte_strings_representation)
    bpt_prior = sum(8 * len(byte_string) for byte_string in byte_strings_hyper_prior)
    bpt = bpt_representation + bpt_prior

    elapsed = elapsed_encode + elapsed_decode

    return reconstruction, bpt, elapsed


def encode_cpu(model: LMCodec, inputs: Tensor) -> tuple[list[bytes], list[bytes]]:
    hyper_prior = model.analysis_prior.forward(inputs)
    hyper_prior = torch.round(hyper_prior)

    byte_strings_hyper_prior = model.model_prior.encode(hyper_prior.cpu())

    parameters_representation = model.synthesis_prior.forward(hyper_prior)

    byte_strings_representation = model.model_representation.encode(inputs.cpu(), parameters_representation.cpu())

    return byte_strings_representation, byte_strings_hyper_prior


def encode_cpu_profiled(model: LMCodec, inputs: Tensor) -> tuple[list[bytes], list[bytes], int]:
    torch.cuda.synchronize()
    tic = time.monotonic_ns()

    hyper_prior = model.analysis_prior.forward(inputs)
    hyper_prior = torch.round(hyper_prior)

    torch.cuda.synchronize()
    toc = time.monotonic_ns()
    elapsed = toc - tic

    byte_strings_hyper_prior = model.model_prior.encode(hyper_prior.cpu())

    torch.cuda.synchronize()
    tic = time.monotonic_ns()

    parameters_representation = model.synthesis_prior(hyper_prior)

    torch.cuda.synchronize()
    toc = time.monotonic_ns()
    elapsed = elapsed + toc - tic

    byte_strings_representation = model.model_representation.encode(
        inputs.cpu(),
        parameters_representation.cpu(),
    )

    return byte_strings_representation, byte_strings_hyper_prior, elapsed


def decode_cpu(
    model: LMCodec,
    byte_strings_representation: list[bytes],
    byte_strings_hyper_prior: list[bytes],
    batch_size: int,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    shape_hyper_prior = Size((batch_size, block_size, model.n_embd_out))
    shape_representation = Size((batch_size, block_size, model.lm.n_embd))

    hyper_prior = model.model_prior.decode(
        byte_strings_hyper_prior,
        shape_hyper_prior,
        model.parameters_prior.dtype,
    )

    parameters = model.synthesis_prior(hyper_prior.cuda())

    inputs, *_ = model.model_representation.decode(
        byte_strings_representation,
        None,
        parameters.cpu(),
        shape_representation,
    )

    return inputs, hyper_prior


def decode_cpu_profiled(
    model: LMCodec,
    byte_strings_representation: list[bytes],
    byte_strings_hyper_prior: list[bytes],
    batch_size: int,
    block_size: int,
) -> tuple[Tensor, Tensor, int]:
    dtype = model.parameters_prior.dtype

    shape_representation = Size((batch_size, block_size, model.lm.n_embd))
    shape_hyper_prior = Size((batch_size, block_size, model.n_embd_out))

    hyper_prior = model.model_prior.decode(byte_strings_hyper_prior, shape_hyper_prior, dtype)

    torch.cuda.synchronize()
    tic = time.monotonic_ns()

    parameters = model.synthesis_prior(hyper_prior.cuda())

    torch.cuda.synchronize()
    toc = time.monotonic_ns()
    elapsed = toc - tic

    inputs, *_ = model.model_representation.decode(
        byte_strings_representation,
        None,
        parameters.cpu(),
        shape_representation,
    )

    return inputs, hyper_prior, elapsed


def code_gpu(model: LMCodec, inputs: Tensor) -> tuple[Tensor, int]:
    batch_size, block_size, *_ = inputs.shape

    byte_strings_representation, byte_strings_hyper_prior = encode_gpu(model, inputs)

    reconstruction, _ = decode_gpu(
        model,
        byte_strings_representation,
        byte_strings_hyper_prior,
        batch_size,
        block_size,
    )

    bpt_representation = sum(8 * len(byte_string) for byte_string in byte_strings_representation)
    bpt_prior = sum(8 * len(byte_string) for byte_string in byte_strings_hyper_prior)
    bpt = bpt_representation + bpt_prior

    return reconstruction, bpt


def code_gpu_profiled(model: LMCodec, inputs: Tensor) -> tuple[Tensor, int, int]:
    batch_size, block_size, *_ = inputs.shape

    byte_strings_representation, byte_strings_hyper_prior, elapsed_encode = encode_gpu_profiled(model, inputs)

    reconstruction, _, elapsed_decode = decode_gpu_profiled(
        model,
        byte_strings_representation,
        byte_strings_hyper_prior,
        batch_size,
        block_size,
    )

    bpt_representation = sum(8 * len(byte_string) for byte_string in byte_strings_representation)
    bpt_prior = sum(8 * len(byte_string) for byte_string in byte_strings_hyper_prior)
    bpt = bpt_representation + bpt_prior

    elapsed = elapsed_encode + elapsed_decode

    return reconstruction, bpt, elapsed


def encode_gpu(model: LMCodec, inputs: Tensor) -> tuple[list[bytes], list[bytes]]:
    hyper_prior = model.analysis_prior(inputs)
    hyper_prior = torch.round(hyper_prior)

    byte_strings_hyper_prior = model.model_prior.encode_gpu(hyper_prior)

    parameters_representation = model.synthesis_prior(hyper_prior)

    byte_strings_representation = model.model_representation.encode_gpu(inputs, parameters_representation)

    return byte_strings_representation, byte_strings_hyper_prior


def encode_gpu_profiled(model: LMCodec, inputs: Tensor) -> tuple[list[bytes], list[bytes], int]:
    torch.cuda.synchronize()
    tic = time.monotonic_ns()

    hyper_prior = model.analysis_prior(inputs)
    hyper_prior = torch.round(hyper_prior)

    torch.cuda.synchronize()
    toc = time.monotonic_ns()
    elapsed = toc - tic

    byte_strings_hyper_prior = model.model_prior.encode_gpu(hyper_prior)

    torch.cuda.synchronize()
    tic = time.monotonic_ns()

    parameters_representation = model.synthesis_prior(hyper_prior)

    torch.cuda.synchronize()
    toc = time.monotonic_ns()
    elapsed = elapsed + toc - tic

    byte_strings_representation = model.model_representation.encode_gpu(inputs, parameters_representation)

    return byte_strings_representation, byte_strings_hyper_prior, elapsed


def decode_gpu(
    model: LMCodec,
    byte_strings_representation: list[bytes],
    byte_strings_hyper_prior: list[bytes],
    batch_size: int,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    dtype = model.parameters_prior.dtype
    device = model.parameters_prior.device

    shape_representation = Size((batch_size, block_size, model.lm.n_embd))
    shape_hyper_prior = Size((batch_size, block_size, model.n_embd_out))

    hyper_prior = model.model_prior.decode_gpu(byte_strings_hyper_prior, shape_hyper_prior, dtype, device)

    parameters_representation = model.synthesis_prior(hyper_prior)

    reconstructions, *_ = model.model_representation.decode_gpu(
        byte_strings_representation,
        None,
        parameters_representation,
        shape_representation,
    )

    return reconstructions, hyper_prior


def decode_gpu_profiled(
    model: LMCodec,
    byte_strings_representation: list[bytes],
    byte_strings_hyper_prior: list[bytes],
    batch_size: int,
    block_size: int,
) -> tuple[Tensor, Tensor, int]:
    dtype = model.parameters_prior.dtype
    device = model.parameters_prior.device

    shape_representation = Size((batch_size, block_size, model.lm.n_embd))
    shape_hyper_prior = Size((batch_size, block_size, model.n_embd_out))

    hyper_prior = model.model_prior.decode_gpu(byte_strings_hyper_prior, shape_hyper_prior, dtype, device)

    torch.cuda.synchronize()
    tic = time.monotonic_ns()

    parameters_representation = model.synthesis_prior(hyper_prior)

    toc = time.monotonic_ns()
    torch.cuda.synchronize()
    elapsed = toc - tic

    reconstructions, *_ = model.model_representation.decode_gpu(
        byte_strings_representation,
        None,
        parameters_representation,
        shape_representation,
    )

    return reconstructions, hyper_prior, elapsed


@torch.no_grad
def calculate_time_complexity(
    model: LMCodec,
    dataloader: DataLoader,
    batch_size: int,
    block_size: int,
    gpu: bool,
    profiled: bool,
) -> tuple[float, float, float]:
    split_index = model.split_index if isinstance(model, LMCodec) else 3

    match gpu, profiled:
        case True, False:
            code = lambda x, y: code_gpu(x, y) + (0,)

        case True, True:
            code = code_gpu_profiled

        case False, False:
            code = lambda x, y: code_cpu(x, y) + (0,)

        case False, True:
            code = code_cpu_profiled

    time_complexity = 0.0
    time_complexity_prior = 0.0
    rate = 0.0

    shape = Size((batch_size, block_size, model.lm.n_embd))

    if gpu:
        model.model_prior.initialize_codec_gpu(model.parameters_prior, shape)
        model.model_representation.initialize_codec_gpu(model.parameters_prior.device)
    else:
        model.model_prior.initialize_codec(model.parameters_prior.cpu(), shape)
        model.model_representation.initialize_codec()

    for index, inputs in enumerate(iter(dataloader)):
        inputs = inputs.cuda()

        batch_size, block_size, *_ = inputs.shape

        *_, (representations, *_) = model.forward(inputs, return_blocks={split_index}, quantize=True)

        torch.cuda.synchronize()
        start_time = time.monotonic_ns()

        reconstructions, rate_batch, time_complexity_prior_batch = code(model, representations)

        torch.cuda.synchronize()
        end_time = time.monotonic_ns()

        assert (reconstructions.cuda() - representations).abs().mean() < 0.01

        if index == 0:
            continue

        time_complexity_batch = end_time - start_time
        time_complexity_batch /= batch_size * block_size
        time_complexity += (time_complexity_batch - time_complexity) / index

        time_complexity_prior_batch /= batch_size * block_size
        time_complexity_prior += (time_complexity_prior_batch - time_complexity_prior) / index

        rate_batch /= batch_size * block_size
        rate += (rate_batch - rate) / index

    return time_complexity, rate, time_complexity_prior


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
    profiled: bool = True,
    dataset_type: str = 'openwebtext',
    num_documents: int = 1000,
    batch_size: int = 10,
    block_size: int = 1024,
    seed: int = 110069,
) -> None:
    """
    Measure the time complexity of encoding and decoding in terms of time per token and bits per token using the proposed codec.

    :param run_id_pretrained: run ID of the training run
    :param run_id: run ID of the current run
    :param dataset_type: dataset name
    :param gpu: whether to use GPU for encoding and decoding
    :param profiled: whether to measure the hyper prior GPU time separately
    :param num_documents: number of samples
    :param block_size: context size
    :param batch_size: batch size for dataset generation
    :param seed: random seed
    """
    model = state.load_model(run_id_pretrained, LMCodec, cache=True, overwrite=False)
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

    time_complexity, rate, time_complexity_prior = calculate_time_complexity(
        model,
        dataloader,
        batch_size,
        block_size,
        gpu,
        profiled,
    )

    mlflow.log_metrics({
        'Time Complexity': time_complexity,
        'Time Complexity Prior': time_complexity_prior,
        'Rate': rate,
    })


def main():
    defopt.run(analyze, no_negated_flags=True)


if __name__ == '__main__':
    main()
