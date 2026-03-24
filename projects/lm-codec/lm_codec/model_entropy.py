import functools
import math
import random
from collections.abc import Sequence

import numpy as np
import torch
from scipy import special
from torch import Size, Tensor, func
from torch.nn import Module, functional, init

import torch_ans
from coding import conversion, distribution, normal, range_fixed
from lm_codec import functions

ParameterIndices = list[tuple[tuple[int, int], tuple[int, int], tuple[int, int] | None]]


class EntropyModelTrait(Module):
    num_parameters: int
    likelihood_lower_bound: Tensor

    def initial_prior(self, num_features: int, support_length: float) -> Tensor: ...
    def support_range(self, parameters: Tensor) -> Tensor: ...
    def cdf(self, inputs: Tensor, parameters: Tensor) -> Tensor: ...
    def pdf(self, inputs: Tensor, parameters: Tensor) -> Tensor: ...
    def log_pdf(self, inputs: Tensor, parameters: Tensor, lower_bound: bool = False) -> Tensor: ...
    def nll(self, inputs: Tensor, parameters: Tensor) -> Tensor: ...
    def nll_discrete(self, inputs: Tensor, parameters: Tensor, scales: None | Tensor = None) -> Tensor: ...
    def encode(self, inputs: Tensor, parameters: Tensor) -> list[bytes]: ...
    def decode(
        self,
        byte_strings: list[bytes],
        parameters: Tensor,
        shape: Size,
        states: list[int] | None = None,
        positions: list[int] | None = None,
    ) -> tuple[Tensor, Sequence[int], Sequence[int]]: ...


class FourierModel(EntropyModelTrait):
    def __init__(
        self,
        num_coefficients: int = 60,
        likelihood_lower_bound: float = 1e-9,
    ) -> None:
        super().__init__()

        self.num_coefficients = num_coefficients

        self.num_parameters = 2 * num_coefficients + 2
        self.likelihood_lower_bound = torch.tensor(likelihood_lower_bound)

    def initial_prior(
        self,
        num_features: int,
        support_length: float = 10.0,
        likelihood_lower_bound: float = 1e-3,
    ) -> Tensor:
        std = support_length / (math.sqrt(2) * special.erfinv(1 - likelihood_lower_bound))

        coefficients = torch.randn((num_features, 2 * self.num_coefficients), dtype=torch.float32)
        coefficients = std * coefficients

        scales = torch.ones((num_features, 1), dtype=torch.float32)
        offsets = torch.zeros((num_features, 1), dtype=torch.float32)

        return torch.concatenate((scales, offsets, coefficients), dim=-1)

    def get_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        coefficients, scales, offsets = parameters[..., 2:], parameters[..., 0], parameters[..., 1]

        shape = list(coefficients.shape[:-1]) + [self.num_coefficients, 2]

        coefficients = torch.reshape(coefficients, shape)

        scales = functional.softplus(scales)

        return coefficients, scales, offsets

    def pdf(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        coefficients, scales, offsets = self.get_parameters(parameters)

        # change of variables using scaled and shifted hyperbolic tangent.
        g_inv = torch.tanh((inputs - offsets) / scales)
        dg_inv_dx = (1 - torch.square(inputs)) / scales

        pdfs = functions.periodic_probability(coefficients, g_inv)
        return pdfs * dg_inv_dx

    def cdf(self, inputs: Tensor, parameters: Tensor, temperature: Tensor | float = 0) -> Tensor:
        coefficients, scales, offsets = self.get_parameters(parameters)

        # inputs = functions.soft_round_inverse(inputs, temperature)
        inputs = torch.tanh((inputs - offsets) / scales)

        return functions.periodic_cummulative_probability(coefficients, inputs)

    def cdf_symbol(
        self,
        inputs: Tensor,
        parameters: Tensor,
        scales: None | Tensor = None,
    ) -> Tensor:
        _, num_features, *_ = inputs.shape

        if scales is None:
            scales = torch.ones(num_features, dtype=inputs.dtype, device=inputs.device)

        inputs = functions.scale(inputs, scales)

        coefficients, scales, offsets = self.get_parameters(parameters)

        # transformation for soft rounding
        # upper = functions.soft_round_inverse(inputs + 0.5, temperature)

        # soft_round is periodic with period 1, so we don't need to call it again
        # lower = upper - 1

        lower = torch.tanh((inputs - 0.5 - offsets) / scales)
        upper = torch.tanh((inputs + 0.5 - offsets) / scales)

        return functions.periodic_probability_discrete(coefficients, lower, upper)

    def log_pdf(self, inputs: Tensor, parameters: Tensor, lower_bound: bool = False) -> Tensor:
        log_pdf = self.pdf(inputs, parameters)

        if lower_bound:
            log_pdf = functions.lower_bound(log_pdf, self.likelihood_lower_bound)

        return torch.log(log_pdf)

    def nll(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        return -self.log_pdf(inputs, parameters, lower_bound=True)

    def nll_discrete(
        self,
        inputs: Tensor,
        parameters: Tensor,
        scales: None | Tensor = None,
    ) -> Tensor:
        nll = self.cdf_symbol(inputs, parameters, scales)
        nll = functions.lower_bound(nll, self.likelihood_lower_bound)
        return -1 * torch.log2(nll)

    def support_range(self, parameters: Tensor) -> Tensor:
        targets = (self.likelihood_lower_bound / 2, 1 - self.likelihood_lower_bound / 2)
        targets = torch.tensor(targets, dtype=parameters.dtype, device=parameters.device)
        targets = torch.tile(targets[:, *[None] * (parameters.ndim - 1)], [1] + list(parameters.shape[:-1]))

        return self.inverse_sampling(parameters, targets=targets)

    def sample_uniform(self, parameters: Tensor) -> Tensor:
        samples = torch.rand(size=parameters.shape[:-1], dtype=parameters.dtype, device=parameters.device)
        return self.likelihood_lower_bound / 2 + samples * (1 - self.likelihood_lower_bound)

    def bisect_method(
        self,
        parameters: Tensor,
        lower_bounds: Tensor,
        upper_bounds: Tensor,
        targets: Tensor,
        num_iterations: int = 10,
    ) -> tuple[Tensor, Tensor, Tensor]:
        candidates = (upper_bounds + lower_bounds) / 2

        for _ in range(num_iterations):
            lengths = upper_bounds - lower_bounds
            extended_lower_bounds = lower_bounds - lengths
            extended_upper_bounds = upper_bounds + lengths

            sign_candidates = torch.sign(self.cdf(candidates, parameters, 0) - targets)
            sign_lower_bounds = torch.sign(self.cdf(lower_bounds, parameters, 0) - targets)
            sign_upper_bounds = torch.sign(self.cdf(upper_bounds, parameters, 0) - targets)

            positive_lower_bounds = sign_lower_bounds == 1
            negative_upper_bounds = sign_upper_bounds == -1
            different_sign_bounds = torch.logical_not(torch.logical_or(positive_lower_bounds, negative_upper_bounds))
            shrink_lower_bounds = torch.logical_and(different_sign_bounds, sign_candidates == sign_lower_bounds)
            shrink_upper_bounds = torch.logical_and(different_sign_bounds, sign_candidates == sign_upper_bounds)

            lower_bounds = torch.where(positive_lower_bounds, extended_lower_bounds, lower_bounds)
            lower_bounds = torch.where(shrink_lower_bounds, candidates, lower_bounds)
            upper_bounds = torch.where(negative_upper_bounds, extended_upper_bounds, upper_bounds)
            upper_bounds = torch.where(shrink_upper_bounds, candidates, upper_bounds)

            candidates = (upper_bounds + lower_bounds) / 2

        return candidates, lower_bounds, upper_bounds

    def inverse_sampling(
        self,
        parameters: Tensor,
        support_range: Tensor | None = None,
        targets: Tensor | None = None,
        max_num_iterations_bisect: int = 15,
        max_num_iterations_newton: int = 10,
        threshold: float = 1e-9,
    ) -> Tensor:
        if support_range is None:
            *dimensions, _ = parameters.shape

            support_range = torch.stack([
                torch.full(dimensions, fill_value=-1, dtype=parameters.dtype, device=parameters.device),
                torch.full(dimensions, fill_value=1, dtype=parameters.dtype, device=parameters.device),
            ])

        targets = targets if targets is not None else self.sample_uniform(parameters)

        lower_bounds, upper_bounds = support_range

        def calculate_pdfs(inputs: Tensor) -> tuple[Tensor, Tensor]:
            cdfs = self.cdf(inputs, parameters, 0)
            loss = torch.sum(cdfs)

            return loss, cdfs

        samples, lower_bounds, upper_bounds = self.bisect_method(parameters, lower_bounds, upper_bounds, targets)

        for _ in range(max_num_iterations_bisect):
            for _ in range(max_num_iterations_newton):
                samples = torch.detach(samples)

                gradients, cdfs = func.grad(calculate_pdfs, has_aux=True)(samples)
                errors = cdfs - targets

                samples = samples - errors / torch.where(gradients != 0, gradients, 1)

                convergence = torch.abs(errors)
                convergence = convergence <= threshold
                convergence = torch.all(convergence)

                if convergence:
                    return samples

            samples, lower_bounds, upper_bounds = self.bisect_method(parameters, lower_bounds, upper_bounds, targets)

        return samples

    def calculate_cdfs(self, parameters: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parameters = torch.reshape(parameters, shape=(-1, self.num_parameters))

        support_range = self.support_range(parameters)

        minima, maxima = support_range[0], support_range[1]
        minima, maxima = torch.floor(minima), torch.ceil(maxima)
        minima, maxima = minima.int(), maxima.int()

        lengths = maxima - minima + 1
        max_length = int(torch.amax(lengths).item())

        samples = torch.arange(max_length, dtype=parameters.dtype, device=parameters.device)
        samples = samples[:, None] + minima[None]

        pmfs = self.cdf_symbol(samples, parameters)
        pmfs = torch.transpose(pmfs, 1, 0)

        tail_masses = 1 - torch.sum(pmfs, dim=-1)

        cdfs = distribution.pmfs_to_quantized_cdfs(
            functions.to_numpy(pmfs),
            functions.to_numpy(lengths),
            functions.to_numpy(tail_masses),
            max_length,
        )

        cdf_sizes = lengths + 2
        cdf_sizes = functions.to_numpy(cdf_sizes)

        offsets = minima
        offsets = functions.to_numpy(offsets)

        return cdfs, cdf_sizes, offsets

    def calculate_indices(self, inputs_shape: Size, parameters_shape: Size, device: torch.device) -> Tensor:
        parameters_shape = parameters_shape[:-1]

        num_parameters = functools.reduce(lambda x, y: x * y, parameters_shape)
        num_missing_dimensions = len(inputs_shape) - len(parameters_shape)
        missing_shape = list(inputs_shape[:num_missing_dimensions])

        indices = torch.arange(num_parameters, dtype=torch.int, device=device)
        indices = torch.reshape(indices, parameters_shape)
        indices = torch.tile(indices[*[None] * num_missing_dimensions], missing_shape + [1] * len(parameters_shape))
        return torch.flatten(indices, start_dim=1)

    def encode(self, inputs: Tensor, parameters: Tensor) -> list[bytes]:
        cdfs, cdf_sizes, offsets = self.calculate_cdfs(parameters)

        symbols = inputs.int()
        symbols = torch.flatten(symbols, start_dim=1)

        indices = self.calculate_indices(inputs.shape, parameters.shape, parameters.device)
        indices = torch.flatten(indices, start_dim=1)

        return [
            conversion.int_array_to_bytes(
                range_fixed.encode_symbols(
                    symbols,
                    indices,
                    cdfs,
                    cdf_sizes,
                    offsets,
                )
            )
            for symbols, indices in zip(*functions.to_numpy(symbols, indices))
        ]

    def decode(
        self,
        byte_strings: list[bytes],
        parameters: Tensor,
        shape: Size,
        states: list[int] | None = None,
        positions: list[int] | None = None,
    ) -> tuple[Tensor, Sequence[int], Sequence[int]]:
        states = states or [-1 for _ in byte_strings]
        positions = positions or [-1 for _ in byte_strings]

        cdfs, cdf_sizes, offsets = self.calculate_cdfs(parameters)

        indices = self.calculate_indices(shape, parameters.shape, parameters.device)

        outputs = [
            range_fixed.decode_symbols(
                state=state,
                position=position,
                cdf_indices=indices,
                codes=conversion.bytes_to_int_array(byte_string),
                cdfs=cdfs,
                cdf_sizes=cdf_sizes,
                offsets=offsets,
            )
            for state, position, indices, byte_string in zip(
                states,
                positions,
                functions.to_numpy(indices),
                byte_strings,
            )
        ]

        tensors, states_new, positions_new = zip(*outputs)

        tensors = torch.stack([
            torch.tensor(
                tensor,
                dtype=parameters.dtype,
                device=parameters.device,
            )
            for tensor in tensors
        ])
        tensors = torch.reshape(tensors, shape)

        return tensors, states_new, positions_new


class GaussianModel(EntropyModelTrait):
    scale_table: Tensor | None = None
    offsets: Tensor | None = None

    def __init__(
        self,
        scale_lower_bound: float = 0.11,
        likelihood_lower_bound: float = 1e-9,
    ) -> None:
        super().__init__()

        self.scale_lower_bound = torch.tensor(scale_lower_bound)
        self.likelihood_lower_bound = torch.tensor(likelihood_lower_bound)

        self.num_parameters = 2

    def initial_prior(
        self,
        num_features: int,
        support_length: float = 10.0,
        likelihood_lower_bound: float = 1e-3,
    ) -> Tensor:
        std = support_length / (math.sqrt(2) * special.erfinv(1 - likelihood_lower_bound))

        parameters = torch.empty([num_features, self.num_parameters], dtype=torch.float32)
        parameters[:, ::2] = 0
        parameters[:, 1::2] = std

        return parameters

    @torch.compile
    def get_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        means, stds = parameters[..., 0], parameters[..., 1]
        stds = functions.lower_bound(stds, self.scale_lower_bound)

        return means, stds

    def support_range(self, parameters: Tensor) -> Tensor:
        means, stds = self.get_parameters(parameters)

        support_range = 1 - self.likelihood_lower_bound
        support_range = stds * math.sqrt(2) * special.erfinv(support_range)
        return torch.stack((means - support_range, means + support_range))

    def cdf(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        means, stds = self.get_parameters(parameters)

        z_scores = (inputs - means) / stds

        return functions.calculate_standardized_cumulative(z_scores)

    def log_pdf(self, inputs: Tensor, parameters: Tensor, lower_bound: bool = False) -> Tensor:
        means, stds = self.get_parameters(parameters)

        log_probabilities = functions.calculate_normal_log_probability(inputs, means, stds)

        if lower_bound:
            log_probabilities = functions.lower_bound(log_probabilities, torch.log(self.likelihood_lower_bound))

        return log_probabilities

    def pdf(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        return torch.exp(self.log_pdf(inputs, parameters, lower_bound=False))

    def nll(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        return -self.log_pdf(inputs, parameters, lower_bound=True)

    def nll_discrete(self, inputs: Tensor, parameters: Tensor, scales: None | Tensor = None) -> Tensor:
        _, num_features, *_ = inputs.shape

        if scales is None:
            scales = torch.ones(num_features, dtype=inputs.dtype, device=inputs.device)

        means, stds = self.get_parameters(parameters)

        z_scores = inputs - means
        z_scores = functions.scale_quantize(z_scores, scales)
        z_scores_higher = functions.descale(z_scores + 0.5, scales) / stds
        z_scores_lower = functions.descale(z_scores - 0.5, scales) / stds

        nll = functions.calculate_standardized_cumulative(z_scores_higher)
        nll = nll - functions.calculate_standardized_cumulative(z_scores_lower)
        nll = functions.lower_bound(nll, self.likelihood_lower_bound)
        return -1 * torch.log2(nll)

    @torch.compile
    def calculate_cdf_indices(self, scales: Tensor) -> Tensor:
        assert self.scale_table is not None

        indices = scales[:, :, None] <= self.scale_table[:-1][None, None, :]
        indices = torch.sum(indices, dim=-1, dtype=torch.int32)
        return len(self.scale_table) - indices - 1

    @torch.no_grad
    def initialize_codec(self) -> None:
        scale_table = normal.get_scale_table(self.scale_lower_bound.item())
        cdfs, cdf_sizes, offsets = normal.calculate_cdfs(scale_table, 15, self.likelihood_lower_bound.item())

        self.scale_table = torch.tensor(scale_table)
        self.cdfs = torch.tensor(cdfs)
        self.cdf_sizes = torch.tensor(cdf_sizes)
        self.offsets = torch.tensor(offsets)

    @torch.no_grad
    def encode(self, inputs: Tensor, parameters: Tensor) -> list[bytes]:
        assert self.cdfs is not None
        assert self.cdf_sizes is not None
        assert self.offsets is not None

        batch_size, length, *_ = inputs.shape
        num_elements = batch_size * length

        means, stds = self.get_parameters(parameters)

        indices = torch.flatten(stds, start_dim=0, end_dim=1)
        indices = self.calculate_cdf_indices(indices)

        symbols = functions.round_half_down(inputs - means)
        symbols = torch.flatten(symbols, start_dim=0, end_dim=1)
        symbols = symbols.to(torch.int32)

        stream = torch_ans.rans32_16_init_stream(num_elements)

        torch_ans.rans32_16_push(
            stream,
            symbols,
            indices,
            self.cdfs,
            self.cdf_sizes,
            self.offsets,
            freq_precision=15,
            bypass_coding=True,
        )

        return torch_ans.rans_stream_to_byte_strings(stream)

    @torch.no_grad
    def decode(
        self,
        byte_strings: list[bytes],
        stream: Tensor | None,
        parameters: Tensor,
        shape: Size,
    ) -> tuple[Tensor, Tensor]:
        assert self.cdfs is not None
        assert self.cdf_sizes is not None
        assert self.offsets is not None

        if stream is None:
            stream = torch_ans.rans_byte_strings_to_stream(byte_strings)

        assert stream is not None

        means, stds = self.get_parameters(parameters)

        indices = torch.flatten(stds, start_dim=0, end_dim=1)
        indices = self.calculate_cdf_indices(indices)

        outputs = torch_ans.rans32_16_pop(
            stream,
            indices,
            self.cdfs,
            self.cdf_sizes,
            self.offsets,
            freq_precision=15,
            bypass_coding=True,
        )

        outputs = outputs.to(parameters.dtype)
        outputs = torch.reshape(outputs, shape)
        outputs = functions.round_half_up(outputs + means)

        return outputs, stream

    @torch.no_grad
    def initialize_codec_gpu(self, device: torch.device) -> None:
        scale_table = normal.get_scale_table(self.scale_lower_bound.item())
        cdfs, cdf_sizes, offsets = normal.calculate_cdfs(scale_table, 15, self.likelihood_lower_bound.item())

        self.scale_table = torch.tensor(scale_table, device=device)
        self.cdfs = torch.tensor(cdfs, device=device)
        self.cdf_sizes = torch.tensor(cdf_sizes, device=device)
        self.offsets = torch.tensor(offsets, device=device)

    @torch.no_grad
    def encode_gpu(self, inputs: Tensor, parameters: Tensor) -> list[bytes]:
        assert self.cdfs is not None
        assert self.cdf_sizes is not None
        assert self.offsets is not None

        batch_size, length, *_ = inputs.shape
        num_elements = batch_size * length

        means, stds = self.get_parameters(parameters)

        indices = torch.flatten(stds, start_dim=0, end_dim=1)
        indices = self.calculate_cdf_indices(indices)

        symbols = functions.round_half_down(inputs - means)
        symbols = torch.flatten(symbols, start_dim=0, end_dim=1)
        symbols = symbols.to(torch.int32)

        stream = torch_ans.rans32_16_init_stream(num_elements).to(parameters.device)

        torch_ans.rans32_16_push(
            stream,
            symbols,
            indices,
            self.cdfs,
            self.cdf_sizes,
            self.offsets,
            freq_precision=15,
            bypass_coding=True,
        )

        return torch_ans.rans_stream_to_byte_strings(stream)

    @torch.no_grad
    def decode_gpu(
        self,
        byte_strings: list[bytes],
        stream: Tensor | None,
        parameters: Tensor,
        shape: Size,
    ) -> tuple[Tensor, Tensor]:
        assert self.cdfs is not None
        assert self.cdf_sizes is not None
        assert self.offsets is not None

        if stream is None:
            stream = torch_ans.rans_byte_strings_to_stream(byte_strings).to(parameters.device)

        assert stream is not None

        means, stds = self.get_parameters(parameters)

        indices = torch.flatten(stds, start_dim=0, end_dim=1)
        indices = self.calculate_cdf_indices(indices)

        outputs = torch_ans.rans32_16_pop(
            stream,
            indices,
            self.cdfs,
            self.cdf_sizes,
            self.offsets,
            freq_precision=15,
            bypass_coding=True,
        )

        outputs = outputs.to(parameters.dtype)
        outputs = torch.reshape(outputs, shape)
        outputs = functions.round_half_up(outputs + means)

        return outputs, stream


class GaussianMixtureModel(EntropyModelTrait):
    def __init__(
        self,
        num_mixtures: int,
        scale_lower_bound: float = 0.11,
        likelihood_lower_bound: float = 1e-9,
    ) -> None:
        super().__init__()

        self.num_mixtures = num_mixtures
        self.scale_lower_bound = torch.tensor(scale_lower_bound)
        self.likelihood_lower_bound = torch.tensor(likelihood_lower_bound)

        self.num_parameters = 3 * num_mixtures

    def initial_prior(
        self,
        num_features: int,
        support_length: float = 10.0,
        likelihood_lower_bound: float = 1e-3,
    ) -> Tensor:
        std = support_length / (math.sqrt(2) * special.erfinv(1 - likelihood_lower_bound))

        parameters = torch.empty([num_features, self.num_parameters], dtype=torch.float32)
        parameters[:, 0::3] = torch.softmax(parameters[:, 2::3], dim=-1)
        parameters[:, 1::3] = support_length * torch.randn_like(parameters[:, 1::3])
        parameters[:, 2::3] = std

        return parameters

    def get_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        proportions, means, stds = parameters[..., 0::3], parameters[..., 1::3], parameters[..., 2::3]
        proportions = torch.softmax(proportions, dim=-1)
        stds = functions.lower_bound(stds, self.scale_lower_bound)

        return proportions, means, stds

    def support_range(self, parameters: Tensor) -> Tensor:
        proportions, means, stds = self.get_parameters(parameters)

        support_range = 1 - self.likelihood_lower_bound
        support_range = stds * math.sqrt(2) * special.erfinv(support_range)
        support_range = torch.stack((means - support_range, means + support_range))
        return torch.einsum('...x,...x->...', proportions[None], support_range)

    def cdf(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        proportions, means, stds = self.get_parameters(parameters)

        z_scores = (inputs[..., None] - means) / stds

        cdfs = functions.calculate_standardized_cumulative(z_scores)
        return torch.einsum('...x,...x->...', proportions, cdfs)

    def log_pdf(self, inputs: Tensor, parameters: Tensor, lower_bound: bool = False) -> Tensor:
        proportions, means, stds = self.get_parameters(parameters)

        log_probabilities = functions.calculate_normal_log_probability(inputs, means, stds)
        log_probabilities = torch.einsum('...x,...x->...', proportions, log_probabilities)

        if lower_bound:
            log_probabilities = functions.lower_bound(log_probabilities, torch.log(self.likelihood_lower_bound))

        return log_probabilities

    def pdf(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        return torch.exp(self.log_pdf(inputs, parameters, lower_bound=False))

    def nll(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        return -self.log_pdf(inputs, parameters, lower_bound=True)

    def nll_discrete(self, inputs: Tensor, parameters: Tensor, scales: None | Tensor = None) -> Tensor:
        _, num_features, *_ = inputs.shape

        if scales is None:
            scales = torch.ones(num_features, dtype=inputs.dtype, device=inputs.device)

        proportions, means, stds = self.get_parameters(parameters)

        z_scores = inputs[..., None] - means
        z_scores = functions.scale_quantize(z_scores, scales)
        z_scores_higher = functions.descale(z_scores + 0.5, scales) / stds
        z_scores_lower = functions.descale(z_scores - 0.5, scales) / stds

        nll = functions.calculate_standardized_cumulative(z_scores_higher)
        nll = nll - functions.calculate_standardized_cumulative(z_scores_lower)
        nll = torch.einsum('...x,...x->...', proportions, nll)
        nll = functions.lower_bound(nll, self.likelihood_lower_bound)
        return -1 * torch.log2(nll)


class DistributionFreeModel(EntropyModelTrait):
    indices: Tensor | None = None
    offsets: Tensor | None = None
    cdfs: Tensor | None = None
    cdf_sizes: Tensor | None = None

    def __init__(
        self,
        num_layers: int = 9,
        hidden_size: int = 3,
        likelihood_lower_bound: float = 1e-9,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.likelihood_lower_bound = torch.tensor(likelihood_lower_bound)

        self.num_parameters = (num_layers - 2) * hidden_size**2 + 2 * num_layers * hidden_size + 1
        self.parameter_indices = self.calculate_parameter_indices(hidden_size, num_layers)

    def initial_prior(self, num_features: int, support_length: float = 10.0) -> Tensor:
        parameters = torch.empty([num_features, self.num_parameters], dtype=torch.float32)

        for (weight_start, weight_end), (bias_start, bias_end), scale_indices in self.parameter_indices:
            num_outputs = bias_end - bias_start

            weight = support_length ** (1 / (self.num_layers - 1))
            weight = math.expm1(1 / weight / num_outputs)
            weight = math.log(weight)

            parameters[:, weight_start:weight_end].fill_(weight)

            init.uniform_(parameters[:, bias_start:bias_end], -0.5, 0.5)

            if scale_indices is not None:
                scale_start, scale_end = scale_indices
                init.zeros_(parameters[:, scale_start:scale_end])

        return parameters

    @staticmethod
    def calculate_parameter_indices(hidden_size: int, num_layers: int) -> ParameterIndices:
        offset = 3 + (num_layers - 2) * (hidden_size + 2)

        return [
            (
                (0 * hidden_size, 1 * hidden_size),
                (1 * hidden_size, 2 * hidden_size),
                (2 * hidden_size, 3 * hidden_size),
            ),
            *[
                (
                    ((offset + 0) * hidden_size, (offset + hidden_size) * hidden_size),
                    ((offset + hidden_size) * hidden_size, (offset + hidden_size + 1) * hidden_size),
                    ((offset + hidden_size + 1) * hidden_size, (offset + hidden_size + 2) * hidden_size),
                )
                for offset in range(3, offset, hidden_size + 2)
            ],
            (
                ((offset + 0) * hidden_size, (offset + 1) * hidden_size),
                ((offset + 1) * hidden_size, (offset + 1) * hidden_size + 1),
                None,
            ),
        ]

    @staticmethod
    def get_parameters_subset(parameters: Tensor, index_range: tuple[int, int]) -> Tensor:
        start_index, end_index = index_range
        indices = torch.arange(start_index, end_index, dtype=torch.int32, device=parameters.device)
        return torch.index_select(parameters, dim=-1, index=indices)

    def reshape_weights(self, weights: Tensor, layer_index: int) -> Tensor:
        tensor_shape = weights.shape[:-1]

        if layer_index == 0:
            shape = tensor_shape + (self.hidden_size, 1)

        elif layer_index == self.num_layers - 1:
            shape = tensor_shape + (1, self.hidden_size)

        else:
            shape = tensor_shape + (self.hidden_size, self.hidden_size)

        return torch.reshape(weights, shape)

    def extract_parameters(self, parameters: Tensor) -> list[tuple[Tensor, Tensor, Tensor | None]]:
        return [
            (
                self.reshape_weights(self.get_parameters_subset(parameters, weights_indices), layer_index),
                self.get_parameters_subset(parameters, biases_indices),
                self.get_parameters_subset(parameters, scales_indices) if scales_indices is not None else None,
            )
            for layer_index, (weights_indices, biases_indices, scales_indices) in enumerate(self.parameter_indices)
        ]

    def cdf(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        outputs = torch.unsqueeze(inputs, dim=-1)

        for weights, biases, scales in self.extract_parameters(parameters):
            weights = functional.softplus(weights)
            outputs = torch.einsum('...f,...df->...d', outputs, weights) + biases

            if scales is not None:
                scales = torch.tanh(scales)
                outputs = outputs + scales * torch.tanh(outputs)

        outputs = torch.sigmoid(outputs)
        return torch.squeeze(outputs, dim=-1)

    def pdf(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        return func.grad(lambda inputs: torch.sum(self.cdf(inputs, parameters)))(inputs)

    def log_pdf(self, inputs: Tensor, parameters: Tensor, lower_bound: bool = False) -> Tensor:
        log_probabilities = self.pdf(inputs, parameters)

        if lower_bound:
            log_probabilities = functions.lower_bound(log_probabilities, self.likelihood_lower_bound)

        return torch.log(log_probabilities)

    def nll(self, inputs: Tensor, parameters: Tensor) -> Tensor:
        return -self.log_pdf(inputs, parameters, lower_bound=True)

    def nll_discrete(self, inputs: Tensor, parameters: Tensor, scales: None | Tensor = None) -> Tensor:
        _, num_features, *_ = inputs.shape

        if scales is None:
            scales = torch.ones(num_features, dtype=inputs.dtype, device=inputs.device)

        inputs = functions.scale_quantize(inputs, scales)

        inputs_higher = functions.descale(inputs + 0.5, scales)
        inputs_lower = functions.descale(inputs - 0.5, scales)

        nll = self.cdf(inputs_higher, parameters)
        nll = nll - self.cdf(inputs_lower, parameters)
        nll = functions.lower_bound(nll, self.likelihood_lower_bound)
        return -1 * torch.log2(nll)

    def support_range(self, parameters: Tensor) -> Tensor:
        targets = (self.likelihood_lower_bound / 2, 1 - self.likelihood_lower_bound / 2)
        targets = torch.tensor(targets, dtype=parameters.dtype, device=parameters.device)
        targets = torch.tile(targets[:, *[None] * (parameters.ndim - 1)], [1] + list(parameters.shape[:-1]))

        return self.inverse_sampling(parameters, targets=targets)

    def sample_uniform(self, parameters: Tensor) -> Tensor:
        samples = torch.rand(size=parameters.shape[:-1], dtype=parameters.dtype, device=parameters.device)
        return self.likelihood_lower_bound / 2 + samples * (1 - self.likelihood_lower_bound)

    def bisect_method(
        self,
        parameters: Tensor,
        lower_bounds: Tensor,
        upper_bounds: Tensor,
        targets: Tensor,
        num_iterations: int = 10,
    ) -> tuple[Tensor, Tensor, Tensor]:
        candidates = (upper_bounds + lower_bounds) / 2

        for _ in range(num_iterations):
            lengths = upper_bounds - lower_bounds
            extended_lower_bounds = lower_bounds - lengths
            extended_upper_bounds = upper_bounds + lengths

            sign_candidates = torch.sign(self.cdf(candidates, parameters) - targets)
            sign_lower_bounds = torch.sign(self.cdf(lower_bounds, parameters) - targets)
            sign_upper_bounds = torch.sign(self.cdf(upper_bounds, parameters) - targets)

            positive_lower_bounds = sign_lower_bounds == 1
            negative_upper_bounds = sign_upper_bounds == -1
            different_sign_bounds = torch.logical_not(torch.logical_or(positive_lower_bounds, negative_upper_bounds))
            shrink_lower_bounds = torch.logical_and(different_sign_bounds, sign_candidates == sign_lower_bounds)
            shrink_upper_bounds = torch.logical_and(different_sign_bounds, sign_candidates == sign_upper_bounds)

            lower_bounds = torch.where(positive_lower_bounds, extended_lower_bounds, lower_bounds)
            lower_bounds = torch.where(shrink_lower_bounds, candidates, lower_bounds)
            upper_bounds = torch.where(negative_upper_bounds, extended_upper_bounds, upper_bounds)
            upper_bounds = torch.where(shrink_upper_bounds, candidates, upper_bounds)

            candidates = (upper_bounds + lower_bounds) / 2

        return candidates, lower_bounds, upper_bounds

    def inverse_sampling(
        self,
        parameters: Tensor,
        support_range: Tensor | None = None,
        targets: Tensor | None = None,
        max_num_iterations_bisect: int = 15,
        max_num_iterations_newton: int = 10,
        threshold: float = 1e-9,
    ) -> Tensor:
        if support_range is None:
            *dimensions, _ = parameters.shape

            support_range = torch.stack([
                torch.full(dimensions, fill_value=-1, dtype=parameters.dtype, device=parameters.device),
                torch.full(dimensions, fill_value=1, dtype=parameters.dtype, device=parameters.device),
            ])

        targets = targets if targets is not None else self.sample_uniform(parameters)

        lower_bounds, upper_bounds = support_range

        def calculate_pdfs(inputs: Tensor) -> tuple[Tensor, Tensor]:
            cdfs = self.cdf(inputs, parameters)
            loss = torch.sum(cdfs)

            return loss, cdfs

        samples, lower_bounds, upper_bounds = self.bisect_method(parameters, lower_bounds, upper_bounds, targets)

        for _ in range(max_num_iterations_bisect):
            for _ in range(max_num_iterations_newton):
                samples = torch.detach(samples)

                gradients, cdfs = func.grad(calculate_pdfs, has_aux=True)(samples)
                errors = cdfs - targets

                samples = samples - errors / torch.where(gradients != 0, gradients, 1)

                convergence = torch.abs(errors)
                convergence = convergence <= threshold
                convergence = torch.all(convergence)

                if convergence:
                    return samples

            samples, lower_bounds, upper_bounds = self.bisect_method(parameters, lower_bounds, upper_bounds, targets)

        return samples

    def plot_distribution(
        self,
        parameters: Tensor,
        support_range: Tensor,
        path: str,
        num_samples: int = 10000,
        num_points: int = 100,
        num_columns: int = 4,
    ) -> None:
        num_features, _ = parameters.shape
        num_plots = num_columns**2

        data = {}

        samples = [self.inverse_sampling(parameters, support_range) for _ in range(num_samples)]
        samples = torch.stack(samples)

        indices = list(range(num_features))
        random.shuffle(indices)
        indices = indices[:num_plots]
        indices = sorted(indices)

        for index in indices:
            start = torch.min(samples[:, index])
            end = torch.max(samples[:, index])

            points = torch.linspace(start, end, num_points, dtype=parameters.dtype, device=parameters.device)

            pdfs = self.pdf(points, parameters[index])

            label = f'Feature {index + 1}'

            data[label] = (points, pdfs, samples[:, index])

        functions.plot_distribution(data, path, num_columns)

    def calculate_qmfs(self, parameters: Tensor) -> tuple[list[np.ndarray], Tensor]:
        parameters = torch.reshape(parameters, shape=(-1, self.num_parameters))

        support_range = self.support_range(parameters)

        minima, maxima = support_range[0], support_range[1]
        minima, maxima = torch.floor(minima), torch.ceil(maxima)
        minima, maxima = minima.int(), maxima.int()

        lengths = maxima - minima + 1

        qmfs = []

        for length, minimum, parameter in zip(lengths, minima, parameters):
            length = int(length.item())

            samples = torch.arange(length, dtype=parameter.dtype, device=parameter.device)
            samples = samples + minimum

            upper = self.cdf(samples + 0.5, parameter)
            lower = self.cdf(samples - 0.5, parameter)

            pmf = upper - lower
            pmf = pmf.numpy(force=True)

            cdf = distribution.pmf_to_quantized_cdf(pmf, 15)

            qmf = cdf[1:] - cdf[:-1]

            qmfs.append(qmf)

        return qmfs, minima

    def calculate_indices(self, inputs_shape: Size, parameters_shape: Size, device: torch.device) -> Tensor:
        parameters_shape = parameters_shape[:-1]

        num_parameters = functools.reduce(lambda x, y: x * y, parameters_shape)
        num_missing_dimensions = len(inputs_shape) - len(parameters_shape)
        missing_shape = list(inputs_shape[:num_missing_dimensions])

        indices = torch.arange(num_parameters, dtype=torch.int, device=device)
        indices = torch.reshape(indices, parameters_shape)

        return torch.tile(indices[*[None] * num_missing_dimensions], missing_shape + [1] * len(parameters_shape))

    def calculate_cdfs(self, parameters: Tensor, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        parameters = torch.reshape(parameters, shape=(-1, self.num_parameters))

        support_range = self.support_range(parameters)

        minima, maxima = support_range[0], support_range[1]
        minima, maxima = torch.floor(minima), torch.ceil(maxima)
        minima, maxima = minima.int(), maxima.int()

        lengths = maxima - minima + 1
        max_length = int(torch.amax(lengths).item())

        samples = torch.arange(max_length, dtype=parameters.dtype, device=parameters.device)
        samples = samples[:, None] + minima[None]

        upper = self.cdf(samples + 0.5, parameters)
        lower = self.cdf(samples - 0.5, parameters)

        pmfs = torch.transpose(upper - lower, 1, 0)
        tail_mass = 1 + lower[0] - upper[-1]

        cdfs = distribution.pmfs_to_quantized_cdfs(
            functions.to_numpy(pmfs),
            functions.to_numpy(lengths),
            functions.to_numpy(tail_mass),
            max_length,
            15,
        )

        cdfs = torch.tensor(cdfs, device=device)
        cdf_sizes = (lengths + 2).to(device=device)
        offsets = minima.to(device=device)

        return cdfs, cdf_sizes, offsets

    def initialize_codec(self, parameters: Tensor, shape: Size) -> None:
        self.cdfs, self.cdf_sizes, self.offsets = self.calculate_cdfs(parameters, torch.device('cpu'))

        indices = self.calculate_indices(shape, parameters.shape, parameters.device)
        self.indices = torch.flatten(indices, start_dim=0, end_dim=1)

    def encode(self, inputs: Tensor) -> list[bytes]:
        assert self.indices is not None
        assert self.cdfs is not None
        assert self.cdf_sizes is not None
        assert self.offsets is not None

        batch_size, length, *_ = inputs.shape

        num_elements = batch_size * length

        symbols = torch.flatten(inputs, start_dim=0, end_dim=1)
        symbols = symbols.to(torch.int32)

        stream = torch_ans.rans32_16_init_stream(num_elements)

        torch_ans.rans32_16_push(
            stream,
            symbols,
            self.indices,
            self.cdfs,
            self.cdf_sizes,
            self.offsets,
            freq_precision=15,
            bypass_coding=True,
        )

        return torch_ans.rans_stream_to_byte_strings(stream)

    def decode(self, byte_strings: list[bytes], shape: Size, dtype: torch.dtype) -> Tensor:
        assert self.indices is not None
        assert self.cdfs is not None
        assert self.cdf_sizes is not None
        assert self.offsets is not None

        stream = torch_ans.rans_byte_strings_to_stream(byte_strings)

        outputs = torch_ans.rans32_16_pop(
            stream,
            self.indices,
            self.cdfs,
            self.cdf_sizes,
            self.offsets,
            freq_precision=15,
            bypass_coding=True,
        )
        outputs = outputs.to(dtype)

        return torch.reshape(outputs, shape)

    def initialize_codec_gpu(self, parameters: Tensor, shape: Size) -> None:
        self.cdfs, self.cdf_sizes, self.offsets = self.calculate_cdfs(parameters, parameters.device)

        indices = self.calculate_indices(shape, parameters.shape, parameters.device)
        indices = torch.flatten(indices, start_dim=0, end_dim=1)

        self.indices = indices

    def encode_gpu(self, inputs: Tensor) -> list[bytes]:
        assert self.indices is not None
        assert self.cdfs is not None
        assert self.cdf_sizes is not None
        assert self.offsets is not None

        device = inputs.device
        batch_size, length, *_ = inputs.shape

        num_elements = batch_size * length

        symbols = torch.flatten(inputs, start_dim=0, end_dim=1)
        symbols = symbols.int()

        stream = torch_ans.rans32_16_init_stream(num_elements).to(device)

        torch_ans.rans32_16_push(
            stream,
            symbols,
            self.indices,
            self.cdfs,
            self.cdf_sizes,
            self.offsets,
            freq_precision=15,
            bypass_coding=True,
        )

        return torch_ans.rans_stream_to_byte_strings(stream)

    def decode_gpu(
        self,
        byte_strings: list[bytes],
        shape: Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        assert self.indices is not None
        assert self.cdfs is not None
        assert self.cdf_sizes is not None
        assert self.offsets is not None

        stream = torch_ans.rans_byte_strings_to_stream(byte_strings).to(device)

        outputs = torch_ans.rans32_16_pop(
            stream,
            self.indices,
            self.cdfs,
            self.cdf_sizes,
            self.offsets,
            freq_precision=15,
            bypass_coding=True,
        )
        outputs = outputs.to(dtype)

        return torch.reshape(outputs, shape)
