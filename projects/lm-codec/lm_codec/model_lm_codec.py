import itertools

import torch
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Size, Tensor
from torch.nn import Parameter, Sequential
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR

from lm_codec import functions
from lm_codec.model_codec import Codec, PriorAnalysis
from lm_codec.model_entropy import DistributionFreeModel, FourierModel, GaussianModel
from lm_codec.model_layers import Block
from lm_codec.model_lm import GPT, LinearCosineCoefficient


class PriorSynthesis(Sequential):
    def __init__(
        self,
        n_embd: int,
        n_embd_out: int,
        num_parameters: int,
        n_head: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        self.n_embd_out = n_embd_out
        self.num_parameters = num_parameters

        super().__init__(
            Block(n_head, n_embd, bias, dropout, 96),
            Block(n_head, 96, bias, dropout, 192),
            Block(n_head, 192, bias, dropout, 384),
            Block(n_head, 384, bias, dropout, n_embd_out * num_parameters),
        )

    @torch.compile
    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, block_size, _ = inputs.shape

        outputs = super().forward(inputs)
        return torch.reshape(outputs, (batch_size, block_size, self.n_embd_out, self.num_parameters))


class LMCodec(LightningModule):
    def __init__(
        self,
        split_index: int,
        alpha: float,
        max_steps: int,
        model_type_lm: str | None = None,
        n_embd_out: int = 24,
        **_,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.alpha = alpha
        self.max_steps = max_steps
        self.model_type_lm = model_type_lm
        self.split_index = split_index
        self.n_embd_out = n_embd_out

        self.lm = GPT.from_pretrained(model_type_lm) if model_type_lm else GPT()

        self.model_prior = DistributionFreeModel()
        self.model_representation = GaussianModel()

        self.num_parameters = self.model_representation.num_parameters

        self.analysis_prior = PriorAnalysis(self.lm.n_embd, n_embd_out)
        self.synthesis_prior = PriorSynthesis(n_embd_out, self.lm.n_embd, self.num_parameters)

        parameters_prior = self.model_prior.initial_prior(n_embd_out)
        self.parameters_prior = Parameter(parameters_prior)

    @torch.compile
    def forward(self, *args, quantize: bool = True, **kwargs):
        return self.lm(*args, **kwargs, quantize=quantize)

    @torch.compile
    def calculate_metrics(self, batch: tuple[Tensor, Tensor]) -> AttributeDict[Tensor]:
        inputs, _ = batch
        _, block_size = inputs.shape

        _, distortion, (representation, *_) = self.lm(*batch, {self.split_index}, quantize=True)

        hyper_prior = self.analysis_prior(representation)
        hyper_prior = functions.quantize(hyper_prior)

        likelihoods = self.model_prior.nll_discrete(hyper_prior, self.parameters_prior[None, None])
        bpt_prior = functions.calculate_bpe(likelihoods, block_size)

        parameters_representation = self.synthesis_prior(hyper_prior)

        likelihoods = self.model_representation.nll_discrete(representation, parameters_representation)
        bpt_representation = functions.calculate_bpe(likelihoods, block_size)

        bpt = bpt_prior + bpt_representation

        loss = distortion + self.alpha * bpt

        return AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'BPT': bpt,
            'BPT Hyper Prior': bpt_prior,
            'BPT Representation': bpt_representation,
        })

    def configure_optimizers(self, betas: tuple[float, float] = (0.9, 0.95)):
        groups = self.lm.configure_parameter_groups()

        groups.append({
            'params': itertools.chain(
                self.analysis_prior.parameters(),
                self.synthesis_prior.parameters(),
                self.parameters(recurse=False),
            ),
            'weight_decay': 0.0,
        })

        coefficient = LinearCosineCoefficient(self.max_steps, learning_rate_max=0.00006, warmup_steps=0)

        optimizer = AdamW(groups, coefficient.learning_rate_max, betas)

        scheduler = {
            'scheduler': LambdaLR(optimizer, coefficient.get_coefficient),
            'interval': 'step',
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()})

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Validation {name}': value for name, value in metrics.items()})

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Testing {name}': value for name, value in metrics.items()})


class LMCodecFourier(LightningModule):
    def __init__(
        self,
        split_index: int,
        alpha: float,
        max_steps: int,
        model_type_lm: str | None = None,
        n_embd_out: int = 24,
        gamma: float = 1e-6,
        **_,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.alpha = alpha
        self.max_steps = max_steps
        self.model_type_lm = model_type_lm
        self.split_index = split_index
        self.n_embd_out = n_embd_out
        self.gamma = gamma

        self.lm = GPT.from_pretrained(model_type_lm) if model_type_lm else GPT()

        self.model_prior = FourierModel()
        self.model_representation = GaussianModel()

        self.num_parameters = self.model_representation.num_parameters

        self.analysis_prior = PriorAnalysis(self.lm.n_embd, n_embd_out)
        self.synthesis_prior = PriorSynthesis(n_embd_out, self.lm.n_embd, self.num_parameters)

        parameters_prior = self.model_prior.initial_prior(n_embd_out)
        self.parameters_prior = Parameter(parameters_prior)

    @torch.compile
    def forward(self, *args, **_):
        return self.lm(*args, quantize=True)

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

        byte_strings_hyper_prior = self.model_prior.encode(hyper_prior, self.parameters_prior)

        parameters_representation = self.synthesis_prior(hyper_prior)

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

        shape_representation = (batch_size, block_size, self.lm.n_embd)
        shape_hyper_prior = (batch_size, block_size, self.n_embd_out)
        shape_representation, shape_hyper_prior = Size(shape_representation), Size(shape_hyper_prior)

        hyper_prior, *_ = self.model_prior.decode(byte_strings_hyper_prior, self.parameters_prior, shape_hyper_prior)

        parameters = self.synthesis_prior(hyper_prior)

        inputs, *_ = self.model_representation.decode_gpu(
            byte_strings_representation,
            None,
            parameters,
            shape_representation,
        )

        return inputs, hyper_prior

    @torch.compile
    def calculate_metrics(self, batch: tuple[Tensor, Tensor]) -> AttributeDict[Tensor]:
        inputs, _ = batch
        _, block_size = inputs.shape

        _, distortion, (representation, *_) = self.forward(*batch, {self.split_index})

        hyper_prior = self.analysis_prior(representation)
        hyper_prior = functions.quantize(hyper_prior)

        likelihoods = self.model_prior.nll_discrete(hyper_prior, self.parameters_prior[None, None])
        bpt_prior = functions.calculate_bpe(likelihoods, block_size)

        parameters_representation = self.synthesis_prior(hyper_prior)

        likelihoods = self.model_representation.nll_discrete(representation, parameters_representation)
        bpt_representation = functions.calculate_bpe(likelihoods, block_size)

        bpt = bpt_prior + bpt_representation

        regularization, *_ = self.model_prior.get_parameters(self.parameters_prior)
        regularization = functions.unnormalized_density_variation(regularization)

        loss = distortion + self.alpha * bpt + self.gamma * regularization

        return AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'BPT': bpt,
            'BPT Hyper Prior': bpt_prior,
            'BPT Representation': bpt_representation,
        })

    def configure_optimizers(self, betas: tuple[float, float] = (0.9, 0.95)):
        parameters = itertools.chain(
            self.analysis_prior.parameters(),
            self.synthesis_prior.parameters(),
            self.parameters(recurse=False),
        )

        groups = self.lm.configure_parameter_groups()
        groups.append({'params': parameters, 'weight_decay': 0.0})

        coefficient = LinearCosineCoefficient(self.max_steps)

        optimizer = AdamW(groups, coefficient.learning_rate_max, betas)

        scheduler = {
            'scheduler': LambdaLR(optimizer, coefficient.get_coefficient),
            'interval': 'step',
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()})

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Validation {name}': value for name, value in metrics.items()})

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Testing {name}': value for name, value in metrics.items()})


class LMCodecHyperPrior(LightningModule):
    def __init__(
        self,
        split_index: int,
        alpha: float,
        max_steps: int,
        model_type_lm: str | None = None,
        n_embd_out: int = 24,
        **_,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.alpha = alpha
        self.max_steps = max_steps
        self.model_type_lm = model_type_lm
        self.split_index = split_index
        self.n_embd_out = n_embd_out

        self.lm = GPT.from_pretrained(model_type_lm) if model_type_lm else GPT()
        self.codec = Codec(self.lm.n_embd, n_embd_out)

    @torch.compile
    def forward(self, *args, **_):
        return self.lm(*args, quantize=True)

    @torch.compile
    def calculate_metrics(self, batch: tuple[Tensor, Tensor]) -> AttributeDict[Tensor]:
        _, distortion, (representation, *_) = self.forward(*batch, {self.split_index})

        _, (bpt_representation, bpt_prior) = self.codec(representation, self.global_step)

        bpt = bpt_prior + bpt_representation

        loss = distortion + self.alpha * bpt

        return AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'BPT': bpt,
            'BPT Hyper Prior': bpt_prior,
            'BPT Representation': bpt_representation,
        })

    def configure_optimizers(self, betas: tuple[float, float] = (0.9, 0.95)):
        groups = self.lm.configure_parameter_groups()
        groups.append({'params': self.codec.parameters(), 'weight_decay': 0.0})

        coefficient = LinearCosineCoefficient(self.max_steps, learning_rate_max=1e-4)

        optimizer = AdamW(groups, coefficient.learning_rate_max, betas)

        scheduler = {
            'scheduler': LambdaLR(optimizer, coefficient.get_coefficient),
            'interval': 'step',
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()})

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Validation {name}': value for name, value in metrics.items()})

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Testing {name}': value for name, value in metrics.items()})


class LMQuantized(LightningModule):
    def __init__(
        self,
        max_steps: int,
        split_indices: set[int] | None = None,
        model_type_lm: str | None = None,
        quantize: bool = False,
        **_,
    ):
        if split_indices is None:
            split_indices = set()
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.max_steps = max_steps
        self.split_indices = split_indices
        self.model_type_lm = model_type_lm
        self.quantize = quantize

        self.lm = GPT.from_pretrained(model_type_lm) if model_type_lm else GPT()

    @torch.compile
    def forward(self, *args, quantize: bool = True, **kwargs):
        return self.lm(*args, **kwargs, quantize=self.quantize)

    @torch.compile
    def calculate_metrics(self, batch: tuple[Tensor, Tensor]) -> AttributeDict[Tensor]:
        _, distortion, _ = self.lm(*batch, self.split_indices, quantize=self.quantize)

        return AttributeDict({'Loss': distortion, 'Distortion': distortion})

    def configure_optimizers(self, betas: tuple[float, float] = (0.9, 0.95)):
        groups = self.lm.configure_parameter_groups()

        coefficient = LinearCosineCoefficient(self.max_steps)

        optimizer = AdamW(groups, coefficient.learning_rate_max, betas)

        scheduler = {
            'scheduler': LambdaLR(optimizer, coefficient.get_coefficient),
            'interval': 'step',
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()})

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Validation {name}': value for name, value in metrics.items()})

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Testing {name}': value for name, value in metrics.items()})
