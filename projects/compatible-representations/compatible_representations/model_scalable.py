from typing import Tuple

import sfu_torch_lib.state as state
import sfu_torch_lib.utils as utils
import torch
import torch.nn.functional as functional
import torchvision.models.segmentation as segmentation
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.optim import Adam, Optimizer
from torchmetrics import Accuracy, JaccardIndex, MeanMetric

import compatible_representations.functions as functions
import compatible_representations.model_lst as model_lst
from compatible_representations.model_entropy_balle import GaussianConditionalEntropyModel
from compatible_representations.model_lst import DownSample, UpSample
from compatible_representations.processing import BatchTypeTensors, BatchTypeSegmentation


class ConditionalReconstruction(LightningModule):
    def __init__(
        self,
        run_id_encoder: str,
        encoder_type: str,
        num_channels_input: int,
        alpha: float,
        learning_rate: float,
        num_channels: int = 192,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_channels_input = num_channels_input
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.encoder_type = encoder_type
        self.num_channels = num_channels

        base_class = utils.get_type_or_run_class(encoder_type, run_id_encoder)
        self.model_base = state.load_model(run_id_encoder, base_class)

        self.encoder_enhancement = model_lst.elic_lst_downsample(num_channels_input, num_channels)
        self.decoder = model_lst.elic_lst_upsample(num_channels, num_channels_input)
        self.entropy_model = GaussianConditionalEntropyModel(num_channels, num_channels)

    def forward(self, *args, **_) -> Tensor:
        return self.encoder_enhancement(*args)

    def calculate_training_metrics(self, batch: Tensor) -> AttributeDict[Tensor]:
        self.model_base.eval()

        with torch.no_grad():
            inputs, base, likelihoods_base = self.model_base(batch)

        enhancement = self.encoder_enhancement(inputs)
        enhancement, likelihoods_enhancement = self.entropy_model(enhancement, base)

        reconstruction = self.decoder(enhancement)

        rmse, psnr = functions.calculate_reconstruction_loss(reconstruction, inputs)

        bpp_enhancement = functions.calculate_likelihood_bpp_fixed(likelihoods_enhancement, inputs.shape)
        bpp_base = functions.calculate_likelihood_bpp_fixed(likelihoods_base, inputs.shape)
        bpp = bpp_enhancement + bpp_base

        loss = self.alpha * bpp + rmse

        metrics = AttributeDict({
            'Loss': loss,
            'RMSE': rmse,
            'PSNR': psnr,
            'BPP': bpp,
            'BPP Enhancement': bpp_enhancement,
            'BPP Base': bpp_base,
        })

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), self.learning_rate)

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Test {name}', value)


class StandaloneReconstruction(ConditionalReconstruction):
    def calculate_training_metrics(self, batch: Tensor) -> AttributeDict[Tensor]:
        self.model_base.eval()

        with torch.no_grad():
            inputs, base, likelihoods_base = self.model_base(batch)
            base = torch.zeros_like(base)

        enhancement = self.encoder_enhancement(inputs)
        enhancement = torch.detach(enhancement)
        enhancement, likelihoods_enhancement = self.entropy_model(enhancement, base)

        with torch.no_grad():
            reconstruction = self.decoder(enhancement)

        rmse, psnr = functions.calculate_reconstruction_loss(reconstruction, inputs)

        bpp_enhancement = functions.calculate_likelihood_bpp_fixed(likelihoods_enhancement, inputs.shape)
        bpp_base = functions.calculate_likelihood_bpp_fixed(likelihoods_base, inputs.shape)
        bpp = bpp_enhancement + bpp_base

        loss = self.alpha * bpp + rmse

        metrics = AttributeDict({
            'Loss': loss,
            'RMSE': rmse,
            'PSNR': psnr,
            'BPP': bpp,
            'BPP Enhancement': bpp_enhancement,
            'BPP Base': bpp_base,
        })

        return metrics


class ConditionalReconstructionDynamic(LightningModule):
    def __init__(
        self,
        run_id_encoder: str,
        encoder_type: str,
        num_channels_input: int,
        alpha: float,
        learning_rate: float,
        num_channels: int = 192,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_channels_input = num_channels_input
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.encoder_type = encoder_type
        self.num_channels = num_channels

        base_class = utils.get_type_or_run_class(encoder_type, run_id_encoder)
        self.model_base = state.load_model(run_id_encoder, base_class)

        self.encoder_enhancement = model_lst.elic_lst_downsample(num_channels_input, num_channels)
        self.decoder = model_lst.elic_lst_upsample(num_channels, num_channels_input)
        self.entropy_model = GaussianConditionalEntropyModel(num_channels, num_channels)

    def forward(self, *args, **_) -> Tensor:
        return self.encoder_enhancement(*args)

    def calculate_training_metrics(self, batch: BatchTypeTensors) -> AttributeDict[Tensor]:
        self.model_base.eval()

        with torch.no_grad():
            inputs, base, likelihoods_base = self.model_base(batch)

        enhancement = self.encoder_enhancement(inputs.tensors)
        enhancement, likelihoods_enhancement = self.entropy_model(enhancement, base)

        reconstruction = self.decoder(enhancement)

        rmse, psnr = functions.calculate_reconstruction_loss(reconstruction, inputs.tensors)

        bpp_enhancement = functions.calculate_likelihood_bpp_dynamic(likelihoods_enhancement, inputs.image_sizes)
        bpp_base = functions.calculate_likelihood_bpp_dynamic(likelihoods_base, inputs.image_sizes)
        bpp = bpp_enhancement + bpp_base

        loss = self.alpha * bpp + rmse

        metrics = AttributeDict({
            'Loss': loss,
            'RMSE': rmse,
            'PSNR': psnr,
            'BPP': bpp,
            'BPP Enhancement': bpp_enhancement,
            'BPP Base': bpp_base,
        })

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), self.learning_rate)

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Test {name}', value)


class StandaloneReconstructionDynamic(ConditionalReconstructionDynamic):
    def calculate_training_metrics(self, batch: BatchTypeTensors) -> AttributeDict[Tensor]:
        self.model_base.eval()

        with torch.no_grad():
            inputs, base, likelihoods_base = self.model_base(batch)
            base = torch.zeros_like(base)

        enhancement = self.encoder_enhancement(inputs.tensors)
        enhancement = torch.detach(enhancement)
        enhancement, likelihoods_enhancement = self.entropy_model(enhancement, base)

        with torch.no_grad():
            reconstruction = self.decoder(enhancement)

        rmse, psnr = functions.calculate_reconstruction_loss(reconstruction, inputs.tensors)

        bpp_enhancement = functions.calculate_likelihood_bpp_dynamic(likelihoods_enhancement, inputs.image_sizes)
        bpp_base = functions.calculate_likelihood_bpp_dynamic(likelihoods_base, inputs.image_sizes)
        bpp = bpp_enhancement + bpp_base

        loss = self.alpha * bpp + rmse

        metrics = AttributeDict({
            'Loss': loss,
            'RMSE': rmse,
            'PSNR': psnr,
            'BPP': bpp,
            'BPP Enhancement': bpp_enhancement,
            'BPP Base': bpp_base,
        })

        return metrics


class ConditionalSegmentation(LightningModule):
    def __init__(
        self,
        run_id_encoder: str,
        encoder_type: str,
        num_channels_input: int,
        num_classes: int,
        ignore_index: int,
        alpha: float,
        learning_rate: float,
        num_channels: int = 192,
        downsample_factor: int = 2,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_channels_input = num_channels_input
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.encoder_type = encoder_type
        self.num_channels = num_channels
        self.downsample_factor = downsample_factor

        base_class = utils.get_type_or_run_class(encoder_type, run_id_encoder)
        self.model_base = state.load_model(run_id_encoder, base_class)

        self.encoder_enhancement = model_lst.elic_lst_downsample(num_channels_input, num_channels)
        self.decoder = model_lst.elic_lst_upsample(num_channels, num_channels_input, num_blocks=1)
        self.classifier = segmentation.deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
        self.entropy_model = GaussianConditionalEntropyModel(num_channels, num_channels)
        self.downsampler = DownSample(num_channels_input, downsample_factor)
        self.upsampler = UpSample(num_channels=num_classes, factor=downsample_factor)

        self.accuracy_segmentation = Accuracy(task='multiclass', num_classes=num_classes, ignore_index=ignore_index)
        self.iou_segmentation = JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=ignore_index)
        self.bpp = MeanMetric()

    def forward(self, *args, **_) -> Tensor:
        return self.encoder_enhancement(self.downsampler(*args))

    def predict_segmentation(self, representations: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.classifier(self.decoder(representations))
        predictions = self.upsampler(outputs['out'])
        predictions_auxiliary = self.upsampler(outputs['aux'])

        return predictions, predictions_auxiliary

    def calculate_segmentation_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = functional.cross_entropy(predictions, targets, ignore_index=self.ignore_index)
        loss *= self.num_classes

        return loss

    def calculate_training_metrics(self, batch: BatchTypeSegmentation) -> AttributeDict[Tensor]:
        inputs, targets = batch

        self.model_base.eval()

        with torch.no_grad():
            _, base, likelihoods_base = self.model_base(inputs)

        enhancement = self.downsampler(inputs)
        enhancement = self.encoder_enhancement(enhancement)
        enhancement, likelihoods_enhancement = self.entropy_model(enhancement, base)

        predictions, predictions_auxiliary = self.predict_segmentation(enhancement)
        distortion = self.calculate_segmentation_loss(predictions, targets)
        distortion_auxiliary = self.calculate_segmentation_loss(predictions_auxiliary, targets)

        bpp_enhancement = functions.calculate_likelihood_bpp_fixed(likelihoods_enhancement, inputs.shape)
        bpp_base = functions.calculate_likelihood_bpp_fixed(likelihoods_base, inputs.shape)
        bpp = bpp_enhancement + bpp_base

        rate_distortion = loss = self.alpha * bpp + distortion + distortion_auxiliary

        self.accuracy_segmentation(predictions, targets)
        self.iou_segmentation(predictions, targets)
        self.bpp(bpp)

        metrics = AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'BPP Enhancement': bpp_enhancement,
            'BPP Base': bpp_base,
            'Rate Distortion': rate_distortion,
        })

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), self.learning_rate)

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Test {name}', value)

    def log_reset_metrics(self, prefix: str) -> None:
        if not self.accuracy_segmentation._update_called:
            return

        self.log(f'{prefix} Segmentation Accuracy', self.accuracy_segmentation.compute())
        self.accuracy_segmentation.reset()

        self.log(f'{prefix} Segmentation IoU', self.iou_segmentation.compute())
        self.iou_segmentation.reset()

        self.log(f'{prefix} BPP', self.bpp.compute())
        self.bpp.reset()

    def on_validation_epoch_start(self) -> None:
        self.log_reset_metrics('Train')

    def on_validation_epoch_end(self) -> None:
        self.log_reset_metrics('Validation')

    def on_test_epoch_end(self) -> None:
        self.log_reset_metrics('Test')


class StandaloneSegmentation(ConditionalSegmentation):
    def calculate_training_metrics(self, batch: BatchTypeSegmentation) -> AttributeDict[Tensor]:
        inputs, targets = batch

        self.model_base.eval()

        with torch.no_grad():
            _, base, likelihoods_base = self.model_base(inputs)
            base = torch.zeros_like(base)

        enhancement = self.downsampler(inputs)
        enhancement = self.encoder_enhancement(enhancement)
        enhancement = torch.detach(enhancement)
        enhancement, likelihoods_enhancement = self.entropy_model(enhancement, base)

        with torch.no_grad():
            predictions, predictions_auxiliary = self.predict_segmentation(enhancement)

        distortion = self.calculate_segmentation_loss(predictions, targets)
        distortion_auxiliary = self.calculate_segmentation_loss(predictions_auxiliary, targets)

        bpp_enhancement = functions.calculate_likelihood_bpp_fixed(likelihoods_enhancement, inputs.shape)
        bpp_base = functions.calculate_likelihood_bpp_fixed(likelihoods_base, inputs.shape)
        bpp = bpp_enhancement + bpp_base

        rate_distortion = loss = self.alpha * bpp + distortion + distortion_auxiliary

        self.accuracy_segmentation(predictions, targets)
        self.iou_segmentation(predictions, targets)
        self.bpp(bpp)

        metrics = AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'BPP Enhancement': bpp_enhancement,
            'BPP Base': bpp_base,
            'Rate Distortion': rate_distortion,
        })

        return metrics
