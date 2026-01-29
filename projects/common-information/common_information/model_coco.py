import torchvision.models.detection as detection
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.nn import BatchNorm1d, BatchNorm2d, InstanceNorm1d, InstanceNorm2d, LayerNorm
from torch.optim.adamw import AdamW
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.image_list import ImageList

import common_information.functions as functions
import common_information.model_ci as model_ci
from common_information.metrics import KeypointMeanAveragePrecision, MeanAveragePrecision
from common_information.model_ci import CodecType


class COCO(LightningModule):
    def __init__(
        self,
        input_num_channels: int,
        latent_num_channels: int,
        num_classes_detection: int,
        num_classes_keypointing: int,
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

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.scale = scale

        self.codec = model_ci.get_codec(codec_type, input_num_channels, latent_num_channels, alpha, beta, scale)

        self.classifier_detection = detection.fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            num_classes=num_classes_detection,
        )

        self.classifier_keypointing = detection.keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1,
            num_classes=num_classes_keypointing,
        )

        self.mean_average_precision_detection = MeanAveragePrecision()
        self.mean_average_precision_keypointing = KeypointMeanAveragePrecision()

    def transform_detection(
        self,
        images: Tensor,
        masks: Tensor,
        targets: list[dict[str, Tensor]] | None,
    ) -> tuple[ImageList, list[dict[str, Tensor]] | None, list[tuple[int, int]]]:
        image_list = functions.tensor_to_images(images, masks)

        original_image_sizes = [(image.shape[-2], image.shape[-1]) for image in image_list]

        image_list, targets = self.classifier_detection.transform(image_list, targets)

        return image_list, targets, original_image_sizes

    def classify_detection(
        self,
        images: Tensor,
        masks: Tensor,
        targets: list[dict[str, Tensor]],
        training: bool,
    ) -> tuple[Tensor | None, list[dict[str, Tensor]] | None]:
        targets_or_none = targets if self.training else None

        image_list, targets_or_none, original_image_sizes = self.transform_detection(images, masks, targets_or_none)

        features = self.classifier_detection.backbone(image_list.tensors)

        proposals, proposal_losses = self.classifier_detection.rpn(image_list, features, targets_or_none)
        detections, detector_losses = self.classifier_detection.roi_heads(
            features,
            proposals,
            image_list.image_sizes,
            targets_or_none,
        )

        losses = detector_losses | proposal_losses
        losses: Tensor = sum(losses.values())  # type: ignore

        if training:
            return losses, None
        else:
            detections = self.classifier_detection.transform.postprocess(  # type: ignore
                detections,
                image_list.image_sizes,
                original_image_sizes,
            )

            return None, detections

    def transform_keypointing(
        self,
        images: Tensor,
        masks: Tensor,
        targets: list[dict[str, Tensor]] | None,
    ) -> tuple[ImageList, list[dict[str, Tensor]] | None, list[tuple[int, int]]]:
        image_list = functions.tensor_to_images(images, masks)

        original_image_sizes = [(image.shape[-2], image.shape[-1]) for image in image_list]

        image_list, targets = self.classifier_keypointing.transform(image_list, targets)

        return image_list, targets, original_image_sizes

    def classify_keypointing(
        self,
        images: Tensor,
        masks: Tensor,
        targets: list[dict[str, Tensor]],
        training: bool,
    ) -> tuple[Tensor | None, list[dict[str, Tensor]] | None]:
        targets_or_none = targets if self.training else None

        image_list, targets_or_none, original_image_sizes = self.transform_keypointing(images, masks, targets_or_none)

        features = self.classifier_keypointing.backbone(image_list.tensors)

        proposals, proposal_losses = self.classifier_keypointing.rpn(image_list, features, targets_or_none)
        detections, detector_losses = self.classifier_keypointing.roi_heads(
            features,
            proposals,
            image_list.image_sizes,
            targets_or_none,
        )

        losses = detector_losses | proposal_losses
        losses: Tensor = sum(losses.values())  # type: ignore

        if training:
            return losses, None
        else:
            detections = self.classifier_keypointing.transform.postprocess(  # type: ignore
                detections,
                image_list.image_sizes,
                original_image_sizes,
            )

            return None, detections

    def eval_task_models(self):
        for module in self.classifier_detection.modules():
            if isinstance(module, (LayerNorm, InstanceNorm1d, InstanceNorm2d, BatchNorm2d, BatchNorm1d)):
                module.eval()

        for module in self.classifier_keypointing.modules():
            if isinstance(module, (LayerNorm, InstanceNorm1d, InstanceNorm2d, BatchNorm2d, BatchNorm1d)):
                module.eval()

    def calculate_metrics(
        self,
        batch: tuple[tuple[Tensor, Tensor], list[dict[str, Tensor]], Tensor, list[dict[str, Tensor]]],
    ) -> AttributeDict[Tensor]:
        (images, masks), targets_detection, _, targets_keypointing = batch

        self.eval_task_models()

        (reconstructions_detection, reconstructions_keypointing), metrics = self.codec(images, masks)

        distortion_detection, predictions_detection = self.classify_detection(
            reconstructions_detection,
            masks,
            targets_detection,
            self.training,
        )

        distortion_keypointing, predictions_keypointing = self.classify_keypointing(
            reconstructions_keypointing,
            masks,
            targets_keypointing,
            self.training,
        )

        if self.training:
            assert distortion_detection is not None and distortion_keypointing is not None

            loss = (
                metrics.bpp_scaled
                + distortion_detection
                + distortion_keypointing
                + self.gamma * (metrics.rmse_a + metrics.rmse_b)
            )

            if 'Auxiliary Loss' in metrics:
                loss = loss + metrics.auxiliary_loss

            metrics |= AttributeDict({
                'Loss': loss,
                'Distortion Detection': distortion_detection,
                'Distortion Keypointing': distortion_keypointing,
            })

        else:
            self.mean_average_precision_detection(predictions_detection, targets_detection)
            self.mean_average_precision_keypointing(predictions_keypointing, targets_keypointing)

        return metrics

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
        ((inputs, *_), *_), *_ = args
        batch_size, *_ = inputs.shape

        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()}, batch_size=batch_size)

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        ((inputs, *_), *_), *_ = args
        batch_size, *_ = inputs.shape

        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Validation {name}': value for name, value in metrics.items()}, batch_size=batch_size)

    def test_step(self, *args, **_) -> None:
        ((inputs, *_), *_), *_ = args
        batch_size, *_ = inputs.shape

        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Testing {name}': value for name, value in metrics.items()}, batch_size=batch_size)

    def on_validation_epoch_end(self) -> None:
        self.log('Validation Mean Average Precision Detection', self.mean_average_precision_detection)
        map_detection = self.trainer.callback_metrics['Validation Mean Average Precision Detection']
        distortion_detection = 1 / map_detection
        self.log('Validation Distortion Detection', distortion_detection)

        self.log('Validation Mean Average Precision Keypointing', self.mean_average_precision_keypointing)
        map_keypointing = self.trainer.callback_metrics['Validation Mean Average Precision Keypointing']
        distortion_keypointing = 1 / map_keypointing
        self.log('Validation Distortion Keypointing', distortion_keypointing)

        distortion = distortion_detection + distortion_keypointing
        self.log('Validation Distortion', distortion)

        bpp_scaled = self.trainer.callback_metrics['Validation BPP Scaled']
        rate_distortion = bpp_scaled + distortion
        self.log('Validation Rate Distortion', rate_distortion)
