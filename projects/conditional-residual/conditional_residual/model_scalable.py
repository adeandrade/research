import itertools
from typing import Optional, Type

import sfu_torch_lib.state as state
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import coding.models.functions as coding
import conditional_residual.functions as functions
import conditional_residual.modules as model_lst
from coding.models.gaussian import GaussianConditionalEntropyModel, GaussianEntropyModel
from conditional_residual.model_baselines import SegmentationCompressedReconstruction, DetectionCompressedReconstruction


class Residual(LightningModule):
    def __init__(
            self,
            image_channels: int,
            alpha: float,
            learning_rate: float,
            run_id_base: str,
            base_class: Type[LightningModule] = DetectionCompressedReconstruction,
            num_channels: Optional[int] = None,
            run_id_reconstruction: Optional[str] = None,
            group_size: int = 16,
    ) -> None:

        super().__init__()

        assert run_id_reconstruction or (num_channels and alpha is not None)

        self.save_hyperparameters(logger=False)

        if run_id_reconstruction:
            model = state.load_model(run_id_reconstruction, Baseline)
            num_channels = model.entropy_model.num_channels
            alpha = model.alpha if alpha is None else alpha

            self.encoder_reconstruction = model.encoder_reconstruction
            self.decoder = model.decoder

        else:
            self.encoder_reconstruction = model_lst.elic_lst_downscale(image_channels, num_channels)
            self.decoder = model_lst.elic_lst_upscale(num_channels, image_channels)

        model = state.load_model(run_id_base, base_class)

        self.encoder_segmentation = model.encoder
        self.transformer = model_lst.elic_lst_upscale(model.num_channels, image_channels)
        self.entropy_model = GaussianEntropyModel(num_channels, pre_group_size=group_size)

        self.num_channels = num_channels
        self.alpha = alpha
        self.learning_rate = learning_rate

    def forward(self, inputs: Tensor) -> Tensor:
        return self.encoder_reconstruction(inputs)

    def calculate_train_metrics(self, inputs: Tensor) -> AttributeDict[Tensor]:
        base = self.encoder_segmentation(inputs)
        base = coding.perturb_or_quantize(base, self.training)
        base = self.transformer(base)

        residual = inputs - base

        enhancement = self.encoder_reconstruction(residual)
        enhancement, likelihoods = self.entropy_model(enhancement)

        reconstruction = self.decoder(enhancement) + base

        rmse, psnr = functions.calculate_reconstruction_loss(reconstruction, inputs)
        rmse_base, psrn_base = functions.calculate_reconstruction_loss(base, inputs)
        entropy = functions.calculate_likelihood_entropy(likelihoods)
        bits = functions.calculate_likelihood_entropy(likelihoods, normalize=False)

        loss = rmse + self.alpha * entropy

        metrics = AttributeDict({
            'Loss': loss,
            'RMSE': rmse,
            'PSNR': psnr,
            'RMSE Base': rmse_base,
            'PSNR Base': psrn_base,
            'Entropy': entropy,
            'Bits': bits,
        })

        return metrics

    def configure_optimizers(self):
        parameters = itertools.chain(
            self.transformer.parameters(),
            self.entropy_model.parameters(),
            self.encoder_reconstruction.parameters(),
            self.decoder.parameters(),
        )

        optimizer = Adam(parameters, self.learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, factor=.75)

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'Validation Loss',
            },
        }

        return configuration

    def training_step(self, inputs: Tensor, *_) -> None:
        metrics = self.calculate_train_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, inputs: Tensor, *_) -> None:
        metrics = self.calculate_train_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, inputs: Tensor, *_) -> None:
        metrics = self.calculate_train_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)


class Conditional(LightningModule):
    def __init__(
            self,
            image_channels: int,
            alpha: float,
            learning_rate: float,
            run_id_base: str,
            base_class: Type[LightningModule] = DetectionCompressedReconstruction,
            num_channels: Optional[int] = None,
            run_id_reconstruction: Optional[str] = None,
            group_size: int = 16,
    ) -> None:

        super().__init__()

        assert run_id_reconstruction or (num_channels and alpha is not None)

        self.save_hyperparameters(logger=False)

        if run_id_reconstruction:
            model = state.load_model(run_id_reconstruction, Baseline)
            num_channels = model.entropy_model.num_channels
            alpha = model.alpha if alpha is None else alpha

            self.encoder_reconstruction = model.encoder_reconstruction
            self.decoder = model.decoder

        else:
            self.encoder_reconstruction = model_lst.elic_lst_downscale(image_channels, num_channels)
            self.decoder = model_lst.elic_lst_upscale(num_channels, image_channels)

        model = state.load_model(run_id_base, base_class)

        self.encoder_segmentation = model.encoder
        self.entropy_model = GaussianConditionalEntropyModel(
            num_channels=num_channels,
            num_channels_conditional=model.num_channels,
            pre_group_size=group_size,
        )

        self.num_channels = num_channels
        self.alpha = alpha
        self.learning_rate = learning_rate

    def forward(self, inputs: Tensor) -> Tensor:
        return self.encoder_reconstruction(inputs)

    def calculate_training_metrics(self, inputs: Tensor) -> AttributeDict[Tensor]:
        base = self.encoder_segmentation(inputs)
        base = coding.perturb_or_quantize(base, self.training)

        enhancement = self.encoder_reconstruction(inputs)
        enhancement, likelihoods = self.entropy_model(enhancement, base)

        reconstruction = self.decoder(enhancement)

        rmse, psnr = functions.calculate_reconstruction_loss(reconstruction, inputs)
        entropy = functions.calculate_likelihood_entropy(likelihoods)
        bits = functions.calculate_likelihood_entropy(likelihoods, normalize=False)

        loss = rmse + self.alpha * entropy

        metrics = AttributeDict({
            'Loss': loss,
            'RMSE': rmse,
            'PSNR': psnr,
            'Entropy': entropy,
            'Bits': bits,
        })

        return metrics

    def configure_optimizers(self):
        parameters = itertools.chain(
            self.entropy_model.parameters(),
            self.encoder_reconstruction.parameters(),
            self.decoder.parameters(),
        )

        optimizer = Adam(parameters, self.learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, factor=.75)

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'Validation Loss',
            },
        }

        return configuration

    def training_step(self, inputs: Tensor, *args) -> None:
        metrics = self.calculate_training_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, inputs: Tensor, *args) -> None:
        metrics = self.calculate_training_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, inputs: Tensor, *args) -> None:
        metrics = self.calculate_training_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)


class Baseline(LightningModule):
    def __init__(
            self,
            image_channels: int,
            alpha: float,
            learning_rate: float,
            num_channels: Optional[int] = None,
            run_id_reconstruction: Optional[str] = None,
            group_size: int = 16,
            **_,
    ) -> None:

        super().__init__()

        assert run_id_reconstruction or (num_channels and alpha is not None)

        self.save_hyperparameters(logger=False)

        if run_id_reconstruction:
            model = state.load_model(run_id_reconstruction, Baseline)
            num_channels = model.entropy_model.num_channels
            alpha = model.alpha if alpha is None else alpha

            self.encoder_reconstruction = model.encoder_reconstruction
            self.decoder = model.decoder

        else:
            self.encoder_reconstruction = model_lst.elic_lst_downscale(image_channels, num_channels)
            self.decoder = model_lst.elic_lst_upscale(num_channels, image_channels)

        self.entropy_model = GaussianEntropyModel(num_channels, pre_group_size=group_size)

        self.num_channels = num_channels
        self.alpha = alpha
        self.learning_rate = learning_rate

    def forward(self, inputs: Tensor) -> Tensor:
        return self.encoder_reconstruction(inputs)

    def calculate_train_metrics(self, inputs: Tensor) -> AttributeDict[Tensor]:
        enhancement = self.encoder_reconstruction(inputs)
        enhancement, likelihoods = self.entropy_model(enhancement)

        reconstruction = self.decoder(enhancement)

        rmse, psnr = functions.calculate_reconstruction_loss(reconstruction, inputs)
        entropy = functions.calculate_likelihood_entropy(likelihoods)
        bits = functions.calculate_likelihood_entropy(likelihoods, normalize=False)
        pixels = float(inputs.shape[2] * inputs.shape[3])

        loss = rmse + self.alpha * entropy

        metrics = AttributeDict({
            'Loss': loss,
            'RMSE': rmse,
            'PSNR': psnr,
            'Entropy': entropy,
            'Bits': bits,
            'Pixels': pixels,
        })

        return metrics

    def configure_optimizers(self):
        parameters = itertools.chain(
            self.entropy_model.parameters(),
            self.encoder_reconstruction.parameters(),
            self.decoder.parameters(),
        )

        optimizer = Adam(parameters, self.learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, factor=.75)

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'Validation Loss',
            },
        }

        return configuration

    def training_step(self, inputs: Tensor, *_) -> None:
        metrics = self.calculate_train_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, inputs: Tensor, *_) -> None:
        metrics = self.calculate_train_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, inputs: Tensor, *_) -> None:
        metrics = self.calculate_train_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)


class Preview(LightningModule):
    def __init__(
            self,
            image_channels: int,
            learning_rate: float,
            run_id_segmentation: str,
            **_,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        model = state.load_model(run_id_segmentation, SegmentationCompressedReconstruction)

        self.encoder_segmentation = model.encoder
        self.transformer = model_lst.elic_lst_upscale(model.num_channels, image_channels)

        self.learning_rate = learning_rate

    def forward(self, inputs: Tensor) -> Tensor:
        base = self.encoder_segmentation(inputs)
        base = coding.perturb_or_quantize(base, self.training)
        base = self.transformer(base)

        return base

    def calculate_train_metrics(self, inputs: Tensor) -> AttributeDict[Tensor]:
        base = self(inputs)

        rmse, psnr = functions.calculate_reconstruction_loss(base, inputs)

        metrics = AttributeDict({'Loss': rmse, 'RMSE': rmse, 'PSNR': psnr})

        return metrics

    def configure_optimizers(self):
        optimizer = Adam(self.transformer.parameters(), self.learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, factor=.75)

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'Validation Loss',
            },
        }

        return configuration

    def training_step(self, inputs: Tensor, *_) -> None:
        metrics = self.calculate_train_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, inputs: Tensor, *_) -> None:
        metrics = self.calculate_train_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, inputs: Tensor, *_) -> None:
        metrics = self.calculate_train_metrics(inputs)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)
