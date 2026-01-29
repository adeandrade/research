from enum import Enum

import torch
import torch.nn.functional as functional
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d

import common_information.functions as functions
import common_information.model_lst as model_lst
from common_information.model_entropy_balle import GaussianConditionalEntropyModel, GaussianEntropyModel


class CodecType(Enum):
    SHARED = 1
    INDEPENDENT = 2
    JOINT = 3
    SEPARATED = 4
    COMBINED = 5


class SharedCodec(Module):
    def __init__(
        self,
        input_num_channels: int,
        latent_num_channels: int,
        alpha: float,
        beta: float,
        scale: float,
    ) -> None:
        super().__init__()

        self.input_num_channels = input_num_channels
        self.latent_num_channels = latent_num_channels
        self.alpha = alpha
        self.beta = beta
        self.scale = scale

        self.num_channels_block = latent_num_channels // 2

        self.encoder_a = model_lst.shuffle_lst_downsample(input_num_channels, num_blocks=1)
        self.encoder_b = model_lst.shuffle_lst_downsample(input_num_channels, num_blocks=1)

        self.decoder_a = model_lst.shuffle_lst_upsample(latent_num_channels, num_blocks=1)
        self.decoder_b = model_lst.shuffle_lst_upsample(latent_num_channels, num_blocks=1)

        self.entropy_model_a = GaussianConditionalEntropyModel(self.num_channels_block, self.num_channels_block)
        self.entropy_model_b = GaussianConditionalEntropyModel(self.num_channels_block, self.num_channels_block)
        self.entropy_model_common = GaussianEntropyModel(self.num_channels_block)

    def forward(self, inputs: Tensor, masks: Tensor) -> tuple[tuple[Tensor, Tensor], AttributeDict[Tensor]]:
        representations_a = self.encoder_a(inputs)
        representations_a = functions.quantize(representations_a)
        common_a = representations_a[:, : self.num_channels_block]
        representations_a = representations_a[:, self.num_channels_block :]

        representations_b = self.encoder_b(inputs)
        representations_b = functions.quantize(representations_b)
        common_b = representations_b[:, : self.num_channels_block]
        representations_b = representations_b[:, self.num_channels_block :]

        representations_common = functions.combine_and_mask(common_a, common_b)

        _, likelihoods = self.entropy_model_common(representations_common)
        bpp_common = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)

        _, likelihoods = self.entropy_model_a(representations_a, representations_common)
        bpp_a = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)
        representations_a = torch.concat((representations_common, representations_a), dim=1)
        reconstructions_a = self.decoder_a(representations_a)

        _, likelihoods = self.entropy_model_b(representations_b, representations_common)
        bpp_b = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)
        representations_b = torch.concat((representations_common, representations_b), dim=1)
        reconstructions_b = self.decoder_b(representations_b)

        bpp = bpp_common + bpp_a + bpp_b
        bpp_scaled = self.alpha * (self.beta * bpp_common + bpp_a + bpp_b)

        distortion_common = functional.mse_loss(common_a, common_b)

        rmse_a = functions.calculate_reconstruction_loss(reconstructions_a, inputs, masks, self.scale)
        rmse_b = functions.calculate_reconstruction_loss(reconstructions_b, inputs, masks, self.scale)

        auxiliary_loss = distortion_common

        metrics = AttributeDict({
            'BPP Common': bpp_common,
            'BPP A': bpp_a,
            'BPP B': bpp_b,
            'BPP': bpp,
            'BPP Scaled': bpp_scaled,
            'Distortion Common': distortion_common,
            'RMSE A': rmse_a,
            'RMSE B': rmse_b,
            'Auxiliary Loss': auxiliary_loss,
        })

        return (reconstructions_a, reconstructions_b), metrics


class IndependentCodec(Module):
    def __init__(
        self,
        input_num_channels: int,
        latent_num_channels: int,
        alpha: float,
        beta: float,
        scale: float,
    ) -> None:
        super().__init__()

        self.input_num_channels = input_num_channels
        self.latent_num_channels = latent_num_channels
        self.alpha = alpha
        self.beta = beta
        self.scale = scale

        self.encoder_a = model_lst.shuffle_lst_downsample(input_num_channels, num_blocks=1)
        self.encoder_b = model_lst.shuffle_lst_downsample(input_num_channels, num_blocks=1)

        self.decoder_a = model_lst.shuffle_lst_upsample(latent_num_channels, num_blocks=1)
        self.decoder_b = model_lst.shuffle_lst_upsample(latent_num_channels, num_blocks=1)

        self.entropy_model_a = GaussianEntropyModel(latent_num_channels)
        self.entropy_model_b = GaussianEntropyModel(latent_num_channels)

    def forward(self, inputs: Tensor, masks: Tensor) -> tuple[tuple[Tensor, Tensor], AttributeDict[Tensor]]:
        representations_a = self.encoder_a(inputs)
        representations_a = functions.quantize(representations_a)

        representations_b = self.encoder_b(inputs)
        representations_b = functions.quantize(representations_b)

        _, likelihoods = self.entropy_model_a(representations_a)
        bpp_a = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)
        reconstructions_a = self.decoder_a(representations_a)

        _, likelihoods = self.entropy_model_b(representations_b)
        bpp_b = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)
        reconstructions_b = self.decoder_b(representations_b)

        bpp = bpp_a + bpp_b
        bpp_scaled = self.alpha * bpp

        rmse_a = functions.calculate_reconstruction_loss(reconstructions_a, inputs, masks, self.scale)
        rmse_b = functions.calculate_reconstruction_loss(reconstructions_b, inputs, masks, self.scale)

        metrics = AttributeDict({
            'BPP Common': torch.tensor(0.0),
            'BPP A': bpp_a,
            'BPP B': bpp_b,
            'BPP': bpp,
            'BPP Scaled': bpp_scaled,
            'Distortion Common': torch.tensor(0.0),
            'RMSE A': rmse_a,
            'RMSE B': rmse_b,
        })

        return (reconstructions_a, reconstructions_b), metrics


class JointCodec(Module):
    def __init__(
        self,
        input_num_channels: int,
        latent_num_channels: int,
        alpha: float,
        beta: float,
        scale: float,
    ) -> None:
        super().__init__()

        self.input_num_channels = input_num_channels
        self.latent_num_channels = latent_num_channels
        self.alpha = alpha
        self.beta = beta
        self.scale = scale

        self.num_channels_block = latent_num_channels // 2

        self.encoder = model_lst.shuffle_lst_downsample(input_num_channels, num_blocks=1)

        self.decoder_a = model_lst.shuffle_lst_upsample(latent_num_channels, num_blocks=1)
        self.decoder_b = model_lst.shuffle_lst_upsample(latent_num_channels, num_blocks=1)

        self.entropy_model = GaussianEntropyModel(latent_num_channels)

    def forward(self, inputs: Tensor, masks: Tensor) -> tuple[tuple[Tensor, Tensor], AttributeDict[Tensor]]:
        representations = self.encoder(inputs)
        representations = functions.quantize(representations)

        _, likelihoods = self.entropy_model(representations)
        bpp_representation = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)

        reconstructions_a = self.decoder_a(representations)
        reconstructions_b = self.decoder_b(representations)

        bpp_scaled = self.alpha * bpp_representation

        rmse_a = functions.calculate_reconstruction_loss(reconstructions_a, inputs, masks, self.scale)
        rmse_b = functions.calculate_reconstruction_loss(reconstructions_b, inputs, masks, self.scale)

        metrics = AttributeDict({
            'BPP Common': torch.tensor(0.0),
            'BPP A': bpp_representation,
            'BPP B': bpp_representation,
            'BPP': bpp_representation,
            'BPP Scaled': bpp_scaled,
            'Distortion Common': torch.tensor(0.0),
            'RMSE A': rmse_a,
            'RMSE B': rmse_b,
        })

        return (reconstructions_a, reconstructions_b), metrics


class SeparatedCodec(Module):
    def __init__(
        self,
        input_num_channels: int,
        latent_num_channels: int,
        alpha: float,
        beta: float,
        scale: float,
    ) -> None:
        super().__init__()

        self.input_num_channels = input_num_channels
        self.latent_num_channels = latent_num_channels
        self.alpha = alpha
        self.beta = beta
        self.scale = scale

        self.num_channels_block = latent_num_channels // 2

        self.encoder_a = Sequential(
            model_lst.shuffle_lst_downsample(input_num_channels, num_blocks=1),
            Conv2d(latent_num_channels, self.num_channels_block, 5, 1, 'same'),
        )
        self.encoder_b = Sequential(
            model_lst.shuffle_lst_downsample(input_num_channels, num_blocks=1),
            Conv2d(latent_num_channels, self.num_channels_block, 5, 1, 'same'),
        )
        self.encoder_common = Sequential(
            model_lst.shuffle_lst_downsample(input_num_channels, num_blocks=1),
            Conv2d(latent_num_channels, self.num_channels_block, 5, 1, 'same'),
        )

        self.decoder_a = model_lst.shuffle_lst_upsample(latent_num_channels, num_blocks=1)
        self.decoder_b = model_lst.shuffle_lst_upsample(latent_num_channels, num_blocks=1)

        self.entropy_model_a = GaussianConditionalEntropyModel(self.num_channels_block, self.num_channels_block)
        self.entropy_model_b = GaussianConditionalEntropyModel(self.num_channels_block, self.num_channels_block)
        self.entropy_model_common = GaussianEntropyModel(self.num_channels_block)

    def forward(self, inputs: Tensor, masks: Tensor) -> tuple[tuple[Tensor, Tensor], AttributeDict[Tensor]]:
        representations_a = self.encoder_a(inputs)
        representations_a = functions.quantize(representations_a)

        representations_b = self.encoder_b(inputs)
        representations_b = functions.quantize(representations_b)

        representations_common = self.encoder_common(inputs)
        representations_common = functions.quantize(representations_common)

        _, likelihoods = self.entropy_model_common(representations_common)
        bpp_common = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)

        _, likelihoods = self.entropy_model_a(representations_a, representations_common)
        bpp_a = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)
        representations_a = torch.concat((representations_common, representations_a), dim=1)
        reconstructions_a = self.decoder_a(representations_a)

        _, likelihoods = self.entropy_model_b(representations_b, representations_common)
        bpp_b = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)
        representations_b = torch.concat((representations_common, representations_b), dim=1)
        reconstructions_b = self.decoder_b(representations_b)

        bpp = bpp_common + bpp_a + bpp_b
        bpp_scaled = self.alpha * (self.beta * bpp_common + bpp_a + bpp_b)

        distortion_common = functional.mse_loss(representations_common, representations_common)

        rmse_a = functions.calculate_reconstruction_loss(reconstructions_a, inputs, masks, self.scale)
        rmse_b = functions.calculate_reconstruction_loss(reconstructions_b, inputs, masks, self.scale)

        metrics = AttributeDict({
            'BPP Common': bpp_common,
            'BPP A': bpp_a,
            'BPP B': bpp_b,
            'BPP': bpp,
            'BPP Scaled': bpp_scaled,
            'Distortion Common': distortion_common,
            'RMSE A': rmse_a,
            'RMSE B': rmse_b,
        })

        return (reconstructions_a, reconstructions_b), metrics


class CombinedCodec(Module):
    def __init__(
        self,
        input_num_channels: int,
        latent_num_channels: int,
        alpha: float,
        beta: float,
        scale: float,
    ) -> None:
        super().__init__()

        self.input_num_channels = input_num_channels
        self.latent_num_channels = latent_num_channels
        self.alpha = alpha
        self.beta = beta
        self.scale = scale

        self.num_channels_block = latent_num_channels // 2

        self.encoder = Sequential(
            model_lst.shuffle_lst_downsample(input_num_channels, num_blocks=1),
            Conv2d(latent_num_channels, 3 * self.num_channels_block, 5, 1, 'same'),
        )

        self.decoder_a = model_lst.shuffle_lst_upsample(latent_num_channels, num_blocks=1)
        self.decoder_b = model_lst.shuffle_lst_upsample(latent_num_channels, num_blocks=1)

        self.entropy_model_a = GaussianConditionalEntropyModel(self.num_channels_block, self.num_channels_block)
        self.entropy_model_b = GaussianConditionalEntropyModel(self.num_channels_block, self.num_channels_block)
        self.entropy_model_common = GaussianEntropyModel(self.num_channels_block)

    def forward(self, inputs: Tensor, masks: Tensor) -> tuple[tuple[Tensor, Tensor], AttributeDict[Tensor]]:
        representations = self.encoder(inputs)
        representations = functions.quantize(representations)

        representations_a = representations[:, : self.num_channels_block]
        representations_b = representations[:, self.num_channels_block : 2 * self.num_channels_block]
        representations_common = representations[:, 2 * self.num_channels_block :]

        _, likelihoods = self.entropy_model_common(representations_common)
        bpp_common = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)

        _, likelihoods = self.entropy_model_a(representations_a, representations_common)
        bpp_a = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)
        representations_a = torch.concat((representations_common, representations_a), dim=1)
        reconstructions_a = self.decoder_a(representations_a)

        _, likelihoods = self.entropy_model_b(representations_b, representations_common)
        bpp_b = functions.calculate_likelihood_bpp_masked(likelihoods, masks, downsample_factor=16)
        representations_b = torch.concat((representations_common, representations_b), dim=1)
        reconstructions_b = self.decoder_b(representations_b)

        bpp = bpp_common + bpp_a + bpp_b
        bpp_scaled = self.alpha * (self.beta * bpp_common + bpp_a + bpp_b)

        distortion_common = functional.mse_loss(representations_common, representations_common)

        rmse_a = functions.calculate_reconstruction_loss(reconstructions_a, inputs, masks, self.scale)
        rmse_b = functions.calculate_reconstruction_loss(reconstructions_b, inputs, masks, self.scale)

        metrics = AttributeDict({
            'BPP Common': bpp_common,
            'BPP A': bpp_a,
            'BPP B': bpp_b,
            'BPP': bpp,
            'BPP Scaled': bpp_scaled,
            'Distortion Common': distortion_common,
            'RMSE A': rmse_a,
            'RMSE B': rmse_b,
        })

        return (reconstructions_a, reconstructions_b), metrics


def get_codec(codec_type: CodecType, *args, **kwargs) -> Module:
    match codec_type:
        case CodecType.SHARED:
            return SharedCodec(*args, **kwargs)

        case CodecType.INDEPENDENT:
            return IndependentCodec(*args, **kwargs)

        case CodecType.JOINT:
            return JointCodec(*args, **kwargs)

        case CodecType.SEPARATED:
            return SeparatedCodec(*args, **kwargs)

        case CodecType.COMBINED:
            return CombinedCodec(*args, **kwargs)
