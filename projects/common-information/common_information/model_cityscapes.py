import sfu_torch_lib.state as state
import torch
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.optim.adamw import AdamW
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

import common_information.model_ci as model_ci
from common_information.model_base import DepthScaled, SegmentationScaled
from common_information.model_ci import CodecType


class Cityscapes(LightningModule):
    def __init__(
        self,
        input_num_channels: int,
        num_classes: int,
        ignore_index: int,
        latent_num_channels: int,
        alpha: float,
        learning_rate: float,
        beta: float = 1.0,
        gamma: float = 1.0,
        scale: float = 255.0,
        codec_type: CodecType = CodecType.JOINT,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.scale = scale

        self.codec = model_ci.get_codec(codec_type, input_num_channels, latent_num_channels, alpha, beta, scale)

        self.classifier_segmentation = state.load_model('8cccd1e01a0e43689e5fc6e771622acd', SegmentationScaled)
        self.classifier_depth = state.load_model('9c9025f4759f4c2489a136d0a7277438', DepthScaled)

        self.accuracy = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index)
        self.iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index)

    def calculate_metrics(
        self,
        batch: tuple[Tensor, Tensor, tuple[Tensor, Tensor]],
    ) -> tuple[AttributeDict[Tensor], Tensor]:
        images, targets_segmentation, targets_depth = batch

        masks = torch.ones_like(images[:, 0])

        self.classifier_segmentation.eval()
        self.classifier_depth.eval()

        (reconstructions_segmentation, reconstructions_depth), metrics = self.codec(images, masks)

        predictions_segmentation = self.classifier_segmentation.predict(reconstructions_segmentation)
        distortion_segmentation = self.classifier_segmentation.calculate_loss(
            predictions_segmentation,
            targets_segmentation,
        )

        predictions_depth = self.classifier_depth.predict(reconstructions_depth)
        distortion_depth = self.classifier_depth.calculate_loss(predictions_depth, *targets_depth)

        loss = (
            metrics.bpp_scaled
            + distortion_segmentation
            + distortion_depth
            + self.gamma * (metrics.rmse_a + metrics.rmse_b)
        )

        if 'Auxiliary Loss' in metrics:
            loss = loss + metrics.auxiliary_loss

        metrics |= AttributeDict({
            'Loss': loss,
            'Distortion Segmentation': distortion_segmentation,
            'Distortion Depth': distortion_depth,
        })

        return metrics, predictions_segmentation

    def configure_optimizers(self, weight_decay: float = 0.0) -> AdamW:
        parameters = [
            {
                'params': self.codec.parameters(),
                'weight_decay': 0.0,
            },
        ]

        optimizer = AdamW(parameters, self.learning_rate, weight_decay=weight_decay)

        return optimizer

    def training_step(self, *args, **_) -> Tensor:
        metrics, *_ = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()})

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        (_, targets, _), *_ = args

        metrics, predictions = self.calculate_metrics(args[0])

        self.accuracy(predictions, targets)
        self.iou(predictions, targets)

        self.log_dict({f'Validation {name}': value for name, value in metrics.items()})

    def on_validation_epoch_end(self) -> None:
        self.log('Validation Accuracy', self.accuracy)
        self.log('Validation mIoU', self.iou)

        iou = self.trainer.callback_metrics['Validation mIoU']
        distortion_depth = self.trainer.callback_metrics['Validation Distortion Depth']
        distortion = 1 / iou + distortion_depth
        self.log('Validation Distortion', distortion)

        bpp_scaled = self.trainer.callback_metrics['Validation BPP Scaled']
        rate_distortion = bpp_scaled + distortion
        self.log('Validation Rate Distortion', rate_distortion)
