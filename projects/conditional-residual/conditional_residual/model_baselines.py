import itertools
from collections import OrderedDict
from typing import Tuple, Sequence, List, Dict, Optional

import torch
import torch.nn.functional as functional
import torchvision.models.detection as detection
import torchvision.models.segmentation as segmentation
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.optim import Adam, SGD
from torchmetrics import Accuracy, JaccardIndex
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.image_list import ImageList

import conditional_residual.functions as functions
import conditional_residual.modules as model_lst
from coding.models.gaussian import GaussianEntropyModel
from conditional_residual.processing_cityscapes import BatchType
from conditional_residual.processing_coco import BatchType as BatchTypeCoco, TargetType as TargetTypeCoco


class Segmentation(LightningModule):
    def __init__(
            self,
            num_classes: int,
            ignore_index: int,
            learning_rate: float,
            momentum: float,
            weight_decay: float,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.model = segmentation.deeplabv3_mobilenet_v3_large(weights_backbone=None, num_classes=num_classes)

        self.accuracy_segmentation = Accuracy(task='multiclass', num_classes=num_classes, ignore_index=ignore_index)
        self.iou_segmentation = JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.model(inputs)['out']

    def calculate_segmentation_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = functional.cross_entropy(predictions, targets, ignore_index=self.ignore_index)
        loss *= self.num_classes

        self.accuracy_segmentation(predictions, targets)
        self.iou_segmentation(predictions, targets)

        return loss

    def calculate_metrics(self, batch: BatchType, *_) -> AttributeDict[Tensor]:
        inputs, targets_segmentation, _ = batch

        loss = self.calculate_segmentation_loss(self(inputs), targets_segmentation)

        metrics = AttributeDict({'Loss': loss})

        return metrics

    def configure_optimizers(self):
        return SGD(self.parameters(), self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)

    def training_step(self, batch: BatchType, *_) -> Tensor:
        metrics = self.calculate_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, batch: BatchType, *_) -> None:
        metrics = self.calculate_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, batch: BatchType, *_) -> None:
        metrics = self.calculate_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)

    def log_reset_metrics(self, prefix: str) -> None:
        if self.accuracy_segmentation._update_called:
            self.log(f'{prefix} Segmentation Accuracy', self.accuracy_segmentation.compute())
            self.log(f'{prefix} Segmentation IoU', self.iou_segmentation.compute())

        self.accuracy_segmentation.reset()
        self.iou_segmentation.reset()

    def on_validation_epoch_start(self) -> None:
        self.log_reset_metrics('Train')

    def on_validation_epoch_end(self) -> None:
        self.log_reset_metrics('Validation')

    def on_test_epoch_end(self) -> None:
        self.log_reset_metrics('Test')


class Detection(LightningModule):
    def __init__(
            self,
            num_classes: int,
            learning_rate: float,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.classifier = detection.fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            num_classes=num_classes,
        )

        self.mean_average_precision = MeanAveragePrecision()

    @staticmethod
    def targets_to_map(targets: Sequence[TargetTypeCoco]) -> List[Dict[str, Tensor]]:
        return [{'boxes': boxes, 'labels': labels, 'masks': masks} for (boxes, labels, masks), *_ in targets]

    def forward(self, images: Sequence[Tensor]) -> List[Dict[str, Tensor]]:
        return self.classifier(images)

    def transform(
            self,
            images: Sequence[Tensor],
            targets: Sequence[Dict[str, Tensor]],
    ) -> Tuple[Tensor, List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]]]:

        original_image_sizes = [tuple(image.shape[-2:]) for image in images]

        images, targets = self.classifier.transform(images, targets)
        image_sizes = images.image_sizes
        images = images.tensors

        return images, targets, original_image_sizes, image_sizes

    def classify(
            self,
            images: Tensor,
            targets: Sequence[Dict[str, Tensor]],
            original_image_sizes: Sequence[Tuple[int, int]],
            image_sizes: List[Tuple[int, int]],
    ) -> Tuple[Optional[Tensor], Optional[List[Dict[str, Tensor]]]]:

        features = self.classifier.backbone(images)
        if isinstance(features, Tensor):
            features = OrderedDict([('0', features)])

        images = ImageList(images, image_sizes)

        proposals, proposal_losses = self.classifier.rpn(images, features, targets)
        detections, detector_losses = self.classifier.roi_heads(features, proposals, image_sizes, targets)
        detections = self.classifier.transform.postprocess(detections, image_sizes, original_image_sizes)
        losses = detector_losses | proposal_losses

        if not torch.jit.is_scripting():
            if self.training:
                losses = self.classifier.eager_outputs(losses, detections)
                detections = None

            else:
                detections = self.classifier.eager_outputs(losses, detections)
                losses = None

        losses = sum(losses.values()) if losses else losses

        return losses, detections

    def calculate_training_metrics(self, batch: BatchTypeCoco) -> AttributeDict[Tensor]:
        images, targets = batch

        targets = self.targets_to_map(targets)

        loss, _ = self.classify(*self.transform(images, targets))

        metrics = AttributeDict({'Loss': loss})

        return metrics

    def calculate_test_metrics(self, batch: BatchTypeCoco) -> AttributeDict[Tensor]:
        images, targets = batch

        targets = self.targets_to_map(targets)

        _, predictions = self.classify(*self.transform(images, targets))

        self.mean_average_precision(predictions, targets)

        metrics = AttributeDict({})

        return metrics

    def configure_optimizers(self):
        return Adam(self.parameters(), self.learning_rate)

    def training_step(self, batch: BatchTypeCoco, *_) -> Tensor:
        metrics = self.calculate_training_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, batch: BatchTypeCoco, *_) -> None:
        metrics = self.calculate_test_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, batch: BatchTypeCoco, *_) -> None:
        metrics = self.calculate_test_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)

    def log_reset_metrics(self, prefix: str) -> None:
        if not self.mean_average_precision._update_called:
            return

        self.log(f'{prefix} Mean Average Precision', self.mean_average_precision.compute()['map'])
        self.mean_average_precision.reset()

    def on_validation_epoch_start(self) -> None:
        self.log_reset_metrics('Train')

    def on_validation_epoch_end(self) -> None:
        self.log_reset_metrics('Validation')

    def on_test_epoch_end(self) -> None:
        self.log_reset_metrics('Test')


class SegmentationCompressedReconstruction(LightningModule):
    def __init__(
            self,
            num_channels: int,
            image_channels: int,
            num_classes: int,
            ignore_index: int,
            alpha: float,
            coder_learning_rate: float,
            classifier_learning_rate: float,
            classifier_momentum: float,
            classifier_weight_decay: float,
            group_size: int = 16,
            **_,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_channels = num_channels
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.coder_learning_rate = coder_learning_rate
        self.classifier_learning_rate = classifier_learning_rate
        self.classifier_momentum = classifier_momentum
        self.classifier_weight_decay = classifier_weight_decay

        self.encoder = model_lst.elic_lst_downscale(image_channels, num_channels)
        self.decoder = model_lst.elic_lst_upscale(num_channels, image_channels)
        self.reconstructor = model_lst.elic_lst_upscale(num_channels, image_channels)
        self.classifier = segmentation.deeplabv3_mobilenet_v3_large(weights_backbone=None, num_classes=num_classes)
        self.entropy_model = GaussianEntropyModel(num_channels, pre_group_size=group_size)

        self.accuracy_segmentation = Accuracy(task='multiclass', num_classes=num_classes, ignore_index=ignore_index)
        self.iou_segmentation = JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def classify(self, representations: Tensor) -> Tensor:
        return self.classifier(self.decoder(representations))['out']

    def calculate_segmentation_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = functional.cross_entropy(predictions, targets, ignore_index=self.ignore_index)
        loss *= self.num_classes

        return loss

    def calculate_metrics(self, batch: BatchType, *_) -> AttributeDict[Tensor]:
        inputs, targets_segmentation, _ = batch

        representation = self(inputs)
        representations_quantized, likelihoods = self.entropy_model(representation)
        predictions = self.classify(representations_quantized)

        distortion = self.calculate_segmentation_loss(predictions, targets_segmentation)
        entropy = functions.calculate_likelihood_entropy(likelihoods)
        bits = functions.calculate_likelihood_entropy(likelihoods, normalize=False)

        reconstruction = self.reconstructor(representations_quantized)
        rmse, psnr = functions.calculate_reconstruction_loss(reconstruction, inputs)

        loss = distortion + self.alpha * entropy + .1 * rmse

        self.accuracy_segmentation(predictions, targets_segmentation)
        self.iou_segmentation(predictions, targets_segmentation)

        metrics = AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'Entropy': entropy,
            'Bits': bits,
            'PSNR': psnr,
            'Correlation': self.entropy_model.correlation,
        })

        return metrics

    def configure_optimizers(self):
        coder_optimizer = Adam(
            params=itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
                self.reconstructor.parameters(),
                self.entropy_model.parameters(),
            ),
            lr=self.coder_learning_rate,
        )

        classifier_optimizer = SGD(
            params=self.classifier.parameters(),
            lr=self.classifier_learning_rate,
            momentum=self.classifier_momentum,
            weight_decay=self.classifier_weight_decay,
        )

        return coder_optimizer, classifier_optimizer

    def training_step(self, batch: BatchType, *_) -> Tensor:
        metrics = self.calculate_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, batch: BatchType, *_) -> None:
        metrics = self.calculate_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, batch: BatchType, *_) -> None:
        metrics = self.calculate_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)

    def log_reset_metrics(self, prefix: str) -> None:
        if not self.accuracy_segmentation._update_called:
            return

        self.log(f'{prefix} Segmentation Accuracy', self.accuracy_segmentation.compute())
        self.accuracy_segmentation.reset()

        self.log(f'{prefix} Segmentation IoU', self.iou_segmentation.compute())
        self.iou_segmentation.reset()

    def on_validation_epoch_start(self) -> None:
        self.log_reset_metrics('Train')

    def on_validation_epoch_end(self) -> None:
        self.log_reset_metrics('Validation')

    def on_test_epoch_end(self) -> None:
        self.log_reset_metrics('Test')


class DetectionCompressedReconstruction(LightningModule):
    def __init__(
            self,
            num_channels: int,
            image_channels: int,
            num_classes: int,
            alpha: float,
            learning_rate: float,
            group_size: int = 16,
            beta: float = .05,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_channels = num_channels
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.group_size = group_size
        self.beta = beta

        self.encoder = model_lst.elic_lst_downscale(image_channels, num_channels)
        self.decoder = model_lst.elic_lst_upscale(num_channels, image_channels)
        self.entropy_model = GaussianEntropyModel(num_channels, pre_group_size=group_size)
        self.classifier = detection.fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            num_classes=num_classes,
        )
        self.reconstructor = model_lst.elic_lst_upscale(num_channels, image_channels)

        self.mean_average_precision = MeanAveragePrecision()

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        representations = self.encoder(inputs)
        representations, likelihoods = self.entropy_model(representations)
        reconstructions = self.decoder(representations)

        return representations, likelihoods, reconstructions

    @staticmethod
    def targets_to_map(targets: Sequence[TargetTypeCoco]) -> List[Dict[str, Tensor]]:
        return [{'boxes': boxes, 'labels': labels, 'masks': masks} for (boxes, labels, masks), *_ in targets]

    def transform(
            self,
            images: Sequence[Tensor],
            targets: Sequence[Dict[str, Tensor]],
    ) -> Tuple[Tensor, List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]]]:

        original_image_sizes = [tuple(image.shape[-2:]) for image in images]

        images, targets = self.classifier.transform(images, targets)
        image_sizes = images.image_sizes
        images = images.tensors

        return images, targets, original_image_sizes, image_sizes

    def classify(
            self,
            images: Tensor,
            targets: Sequence[Dict[str, Tensor]],
            original_image_sizes: Sequence[Tuple[int, int]],
            image_sizes: List[Tuple[int, int]],
    ) -> Tuple[Optional[Tensor], Optional[List[Dict[str, Tensor]]]]:

        features = self.classifier.backbone(images)
        if isinstance(features, Tensor):
            features = OrderedDict([('0', features)])

        images = ImageList(images, image_sizes)

        proposals, proposal_losses = self.classifier.rpn(images, features, targets)
        detections, detector_losses = self.classifier.roi_heads(features, proposals, image_sizes, targets)
        detections = self.classifier.transform.postprocess(detections, image_sizes, original_image_sizes)
        losses = detector_losses | proposal_losses

        if not torch.jit.is_scripting():
            if self.training:
                losses = self.classifier.eager_outputs(losses, detections)
                detections = None

            else:
                detections = self.classifier.eager_outputs(losses, detections)
                losses = None

        losses = sum(losses.values()) if losses else losses

        return losses, detections

    def calculate_training_metrics(self, batch: BatchTypeCoco) -> AttributeDict[Tensor]:
        images, targets = batch
        targets = self.targets_to_map(targets)

        images, targets_transformed, original_image_sizes, image_sizes = self.transform(images, targets)

        representations, likelihoods, reconstructions = self(images)
        distortion, _ = self.classify(reconstructions, targets_transformed, original_image_sizes, image_sizes)

        entropy = functions.calculate_likelihood_entropy(likelihoods)
        bits = functions.calculate_likelihood_entropy(likelihoods, normalize=False)
        pixels = sum(height * width for height, width in image_sizes) / len(image_sizes)

        reconstruction = self.reconstructor(representations)
        rmse, psnr = functions.calculate_reconstruction_loss(reconstruction, images)

        loss = distortion + self.alpha * entropy + self.beta * rmse

        metrics = AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'Entropy': entropy,
            'Bits': bits,
            'RMSE': rmse,
            'PSNR': psnr,
            'Pixels': pixels,
        })

        return metrics

    def calculate_test_metrics(self, batch: BatchTypeCoco) -> AttributeDict[Tensor]:
        images, targets = batch
        targets = self.targets_to_map(targets)

        images, targets_transformed, original_image_sizes, image_sizes = self.transform(images, targets)

        representations, likelihoods, reconstructions = self(images)
        _, predictions = self.classify(reconstructions, targets_transformed, original_image_sizes, image_sizes)

        self.mean_average_precision(predictions, targets)

        entropy = functions.calculate_likelihood_entropy(likelihoods)
        bits = functions.calculate_likelihood_entropy(likelihoods, normalize=False)
        pixels = sum(height * width for height, width in image_sizes) / len(image_sizes)

        reconstruction = self.reconstructor(representations)
        rmse, psnr = functions.calculate_reconstruction_loss(reconstruction, images)

        metrics = AttributeDict({'Entropy': entropy, 'Bits': bits, 'RMSE': rmse, 'PSNR': psnr, 'Pixels': pixels})

        return metrics

    def configure_optimizers(self):
        return Adam(self.parameters(), self.learning_rate)

    def training_step(self, batch: BatchTypeCoco, *_) -> Tensor:
        metrics = self.calculate_training_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, batch: BatchTypeCoco, *_) -> None:
        metrics = self.calculate_test_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, batch: BatchTypeCoco, *_) -> None:
        metrics = self.calculate_test_metrics(batch)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)

    def log_reset_metrics(self, prefix: str) -> None:
        if not self.mean_average_precision._update_called:
            return

        self.log(f'{prefix} Mean Average Precision', self.mean_average_precision.compute()['map'])
        self.mean_average_precision.reset()

    def on_validation_epoch_start(self) -> None:
        self.log_reset_metrics('Train')

    def on_validation_epoch_end(self) -> None:
        self.log_reset_metrics('Validation')

    def on_test_epoch_end(self) -> None:
        self.log_reset_metrics('Test')
