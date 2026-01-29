import itertools

import torch
import torch.nn.functional as functional
import torchvision.models.segmentation as segmentation
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.optim.adamw import AdamW
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

import common_information.functions as functions
import deeplabv3plus.modeling as deeplabv3plus
from common_information.model_lst import DownSample, UpSample


class Depth(LightningModule):
    def __init__(
        self,
        learning_rate: float,
        num_channels_input: int = 3,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.learning_rate = learning_rate
        self.num_channels_input = num_channels_input

        self.classifier_depth = segmentation.lraspp_mobilenet_v3_large(num_classes=1)

    def predict(self, reconstructions: Tensor) -> Tensor:
        return self.classifier_depth(reconstructions)['out'][:, 0]

    def calculate_loss(self, predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        loss = functions.mse_loss(predictions, targets, mask)
        loss = torch.sqrt(loss) * 128

        return loss

    def calculate_metrics(self, batch: tuple[Tensor, tuple[Tensor, Tensor]]) -> AttributeDict[Tensor]:
        images, (targets_depth, mask_depth) = batch

        predictions_depth = self.predict(images)

        distortion_depth = self.calculate_loss(predictions_depth, targets_depth, mask_depth)

        metrics = AttributeDict({
            'Loss': distortion_depth,
            'Distortion Depth': distortion_depth,
        })

        return metrics

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), self.learning_rate)

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


class DepthScaled(Depth):
    def __init__(
        self,
        learning_rate: float,
        num_channels_input: int = 3,
        downsample_factor: int = 2,
        **_,
    ) -> None:
        super().__init__(learning_rate, num_channels_input)

        self.downsample_factor = downsample_factor

        self.downsampler = DownSample(num_channels_input, downsample_factor)
        self.upsampler = UpSample(num_channels=1, factor=downsample_factor)

    def predict(self, reconstructions: Tensor) -> Tensor:
        predictions = self.downsampler(reconstructions)
        predictions = super().predict(predictions)
        predictions = self.upsampler(predictions[:, None])[:, 0]

        return predictions


class SegmentationScaled(LightningModule):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        learning_rate: float,
        num_channels_input: int = 3,
        downsample_factor: int = 2,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.learning_rate = learning_rate
        self.num_channels_input = num_channels_input

        self.downsample_factor = downsample_factor

        self.classifier = deeplabv3plus.deeplabv3plus_mobilenet(num_classes)

        self.downsampler = DownSample(num_channels_input, downsample_factor)
        self.upsampler = UpSample(num_channels=self.num_classes, factor=downsample_factor)

        self.accuracy = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index)
        self.iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index)

    def predict(self, reconstructions: Tensor) -> Tensor:
        predictions = self.downsampler(reconstructions)
        predictions = self.classifier(predictions)
        predictions = self.upsampler(predictions)

        return predictions

    def calculate_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = functional.cross_entropy(predictions, targets, ignore_index=self.ignore_index)
        loss = loss * self.num_classes

        return loss

    def calculate_metrics(self, batch: tuple[Tensor, Tensor]) -> tuple[AttributeDict[Tensor], Tensor]:
        images, targets = batch

        self.classifier.eval()

        predictions_segmentation = self.predict(images)

        distortion_segmentation = self.calculate_loss(predictions_segmentation, targets)

        metrics = AttributeDict({
            'Loss': distortion_segmentation,
            'Distortion Segmentation': distortion_segmentation,
        })

        return metrics, predictions_segmentation

    def configure_optimizers(self) -> AdamW:
        parameters = itertools.chain(
            self.upsampler.parameters(),
            self.downsampler.parameters(),
        )

        optimizer = AdamW(parameters, self.learning_rate)

        return optimizer

    def training_step(self, *args, **_) -> Tensor:
        metrics, *_ = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()})

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        (_, targets), *_ = args

        metrics, predictions = self.calculate_metrics(args[0])
        self.log_dict({f'Validation {name}': value for name, value in metrics.items()})

        self.accuracy(predictions, targets)
        self.log('Validation Accuracy', self.accuracy, on_epoch=True)

        self.iou(predictions, targets)
        self.log('Validation mIoU', self.iou, on_epoch=True)

    def test_step(self, *args, **_) -> None:
        metrics, *_ = self.calculate_metrics(args[0])
        self.log_dict({f'Testing {name}': value for name, value in metrics.items()})
