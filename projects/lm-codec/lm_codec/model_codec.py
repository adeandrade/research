import torch
from torch import Size, Tensor
from torch.nn import Module, Parameter, Sequential

from lm_codec import functions
from lm_codec.model_entropy import DistributionFreeModel, GaussianModel
from lm_codec.model_layers import Block, BlockAutoregressive


class PriorAnalysis(Sequential):
    def __init__(
        self,
        n_embd: int,
        n_embd_out: int,
        n_head: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(
            Block(n_head, n_embd, bias, dropout, 384),
            Block(n_head, 384, bias, dropout, 192),
            Block(n_head, 192, bias, dropout, 96),
            Block(n_head, 96, bias, dropout, n_embd_out),
        )

    @torch.compile
    def forward(self, inputs: Tensor) -> Tensor:
        return super().forward(inputs)


class PriorSynthesis(Sequential):
    def __init__(
        self,
        n_embd: int,
        n_head: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(
            Block(n_head, n_embd, bias, dropout),
            Block(n_head, n_embd, bias, dropout),
            Block(n_head, n_embd, bias, dropout),
            Block(n_head, n_embd, bias, dropout),
        )

    @torch.compile
    def forward(self, inputs: Tensor) -> Tensor:
        return super().forward(inputs)


class RepresentationLST(Module):
    def __init__(
        self,
        n_embd: int,
        num_parameters: int,
        num_parameters_side: int,
        n_head: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_embd = n_embd
        self.num_parameters = num_parameters

        self.masker = BlockAutoregressive(n_head, n_embd, num_parameters_side, bias, dropout)

        self.transformer = Sequential(
            Block(n_head, n_embd + num_parameters_side, bias, dropout),
            Block(n_head, n_embd + num_parameters_side, bias, dropout),
            Block(n_head, n_embd + num_parameters_side, bias, dropout, n_embd * num_parameters),
        )

    def forward(self, inputs: Tensor, side: Tensor) -> Tensor:
        batch_size, block_size, _ = inputs.shape

        outputs = self.masker(inputs, side)
        outputs = self.transformer(outputs)
        return torch.reshape(outputs, (batch_size, block_size, self.n_embd, self.num_parameters))


class Codec(Module):
    def __init__(
        self,
        n_embd: int,
        n_embd_prior: int,
    ) -> None:
        super().__init__()

        self.n_embd = n_embd
        self.n_embd_prior = n_embd_prior

        self.model_prior = DistributionFreeModel()
        self.model_representation = GaussianModel()

        num_parameters = self.model_representation.num_parameters
        self.lst_representation = RepresentationLST(n_embd, num_parameters, n_embd_prior)

        self.analysis_prior = PriorAnalysis(n_embd, n_embd_prior)
        self.synthesis_prior = PriorSynthesis(n_embd_prior)

        parameters_prior = self.model_prior.initial_prior(n_embd_prior)
        self.parameters_prior = Parameter(parameters_prior)

    def forward(
        self,
        inputs: Tensor,
        step: int,
        quantized: bool = False,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        _, block_size, *_ = inputs.shape

        inputs = inputs if quantized else functions.quantize(inputs)

        hyper_prior = self.analysis_prior(inputs)
        hyper_prior = functions.quantize(hyper_prior)

        likelihoods = self.model_prior.nll_discrete(hyper_prior, self.parameters_prior[None, None])
        bpt_prior = functions.calculate_bpe(likelihoods, block_size)

        parameters_representation = self.synthesis_prior(hyper_prior)
        parameters_representation = self.lst_representation(inputs, parameters_representation)

        likelihoods = self.model_representation.nll_discrete(inputs, parameters_representation)
        bpt_representation = functions.calculate_bpe(likelihoods, block_size)

        return (inputs, hyper_prior), (bpt_representation, bpt_prior)

    def forward_coded(
        self,
        inputs: Tensor,
        quantized: bool = False,
    ) -> tuple[tuple[Tensor, Tensor], tuple[float, float]]:
        batch_size, block_size, *_ = inputs.shape

        byte_strings_representation, byte_strings_hyper_prior = self.encode(inputs, quantized)

        inputs, hyper_prior = self.decode(byte_strings_representation, byte_strings_hyper_prior, block_size)

        bpt_representation = sum(8 * len(byte_string) for byte_string in byte_strings_representation)
        bpt_representation /= batch_size * block_size

        bpt_prior = sum(8 * len(byte_string) for byte_string in byte_strings_hyper_prior)
        bpt_prior /= batch_size * block_size

        return (inputs, hyper_prior), (bpt_representation, bpt_prior)

    @torch.no_grad
    def encode(self, inputs: Tensor, quantized: bool = False) -> tuple[list[bytes], list[bytes]]:
        inputs = inputs if quantized else functions.quantize(inputs)

        hyper_prior = self.analysis_prior(inputs)
        hyper_prior = functions.quantize(hyper_prior)

        byte_strings_hyper_prior = self.model_prior.encode_gpu(hyper_prior)

        parameters_representation = self.synthesis_prior(hyper_prior)
        parameters_representation = self.lst_representation(inputs, parameters_representation)

        byte_strings_representation = self.model_representation.encode_gpu(inputs, parameters_representation)

        return byte_strings_representation, byte_strings_hyper_prior

    @torch.no_grad
    def decode(
        self,
        byte_strings_representation: list[bytes],
        byte_strings_hyper_prior: list[bytes],
        block_size: int,
    ) -> tuple[Tensor, Tensor]:
        batch_size = len(byte_strings_representation)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        shape_representation = (batch_size, self.n_embd)
        shape_hyper_prior = (batch_size, block_size, self.n_embd_prior)
        shape_representation, shape_hyper_prior = Size(shape_representation), Size(shape_hyper_prior)

        hyper_prior = self.model_prior.decode_gpu(byte_strings_hyper_prior, shape_hyper_prior, dtype, device)

        parameters = self.synthesis_prior(hyper_prior)

        inputs = torch.zeros((batch_size, block_size, self.n_embd), dtype=dtype, device=device)
        stream = None

        for step in range(block_size):
            parameters_step = self.lst_representation(inputs[:, : step + 1], parameters[:, : step + 1])[:, step]

            inputs[:, step], stream = self.model_representation.decode_gpu(
                byte_strings_representation,
                stream,
                parameters_step,
                shape_representation,
            )

        return inputs, hyper_prior
