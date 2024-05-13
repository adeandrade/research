from typing import Dict, List, OrderedDict, Sequence, Tuple

import torch
import torch.nn.functional as functional
import torchvision.models.detection as detection
import torchvision.models.segmentation as segmentation
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.optim import Adam, Optimizer
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.image_list import ImageList

import compatible_representations.functions as functions
import compatible_representations.model_lst as model_lst
from compatible_representations.model_entropy_balle import GaussianEntropyModel
from compatible_representations.model_lst import DownSample, UpSample
from compatible_representations.processing import BatchTypeDepth, BatchTypeDetection, TargetTypeDetection


class DepthForReconstructionBaseline(LightningModule):
    def __init__(
        self,
        input_num_channels: int,
        alpha: float,
        learning_rate: float,
        latent_num_channels: int = 192,
        downsample_factor: int = 2,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.input_num_channels = input_num_channels
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.downsample_factor = downsample_factor

        self.downsampler = DownSample(input_num_channels, downsample_factor)
        self.upsampler_depth = UpSample(num_channels=1, factor=downsample_factor)
        self.upsampler_reconstruction = UpSample(input_num_channels, downsample_factor)

        self.encoder = model_lst.elic_lst_downsample(input_num_channels, latent_num_channels)
        self.entropy_model = GaussianEntropyModel(latent_num_channels)
        self.decoder_depth = model_lst.elic_lst_upsample(
            latent_num_channels, input_num_channels, num_blocks=1, activation='elu'
        )
        self.classifier_depth = segmentation.lraspp_mobilenet_v3_large(num_classes=1)
        self.decoder_reconstruction = model_lst.elic_lst_upsample(
            latent_num_channels, input_num_channels, activation='elu'
        )

    def forward(self, *args, **_) -> Tuple[Tensor, Tensor, Tensor]:
        inputs = functional.interpolate(*args, scale_factor=1 / self.downsampler.factor, mode=self.downsampler.mode)
        representations, likelihoods = self.entropy_model(self.encoder(self.downsampler(*args)))

        return inputs, representations, likelihoods

    def predict_depth(self, representations: Tensor) -> Tensor:
        return self.classifier_depth(self.decoder_depth(representations))['out']

    def predict_reconstruction(self, representations: Tensor) -> Tensor:
        return self.decoder_reconstruction(representations)

    def calculate_depth_loss(self, predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        loss = self.upsampler_depth(predictions)
        loss = functions.mse_loss(loss[:, 0], targets, mask)
        loss = torch.sqrt(loss) * 128

        return loss

    def calculate_reconstruction_loss(self, predictions: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        loss = self.upsampler_reconstruction(predictions)
        loss = functions.calculate_reconstruction_loss(loss, targets)

        return loss

    def calculate_metrics(self, batch: BatchTypeDepth) -> AttributeDict[Tensor]:
        inputs, (targets_depth, mask_depth) = batch

        representations = self.downsampler(inputs)
        representations = self.encoder(representations)
        representations, likelihoods = self.entropy_model(representations)

        predictions_depth = self.predict_depth(representations)
        predictions_reconstruction = self.predict_reconstruction(torch.detach(representations))

        bpp = functions.calculate_likelihood_bpp_fixed(likelihoods, inputs.shape)
        distortion_depth = self.calculate_depth_loss(predictions_depth, targets_depth, mask_depth)
        distortion_reconstruction, psnr = self.calculate_reconstruction_loss(predictions_reconstruction, inputs)

        loss = self.alpha * bpp + distortion_depth + distortion_reconstruction

        rate_distortion = self.alpha * bpp + distortion_depth

        metrics = AttributeDict({
            'Loss': loss,
            'BPP': bpp,
            'Distortion Depth': distortion_depth,
            'Distortion Reconstruction': distortion_reconstruction,
            'PSNR': psnr,
            'Rate Distortion': rate_distortion,
        })

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), self.learning_rate)

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Test {name}', value)


class DepthForReconstructionAndrade(LightningModule):
    def __init__(
        self,
        input_num_channels: int,
        alpha: float,
        learning_rate: float,
        latent_num_channels: int = 192,
        downsample_factor: int = 2,
        beta: float = 0.1,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.input_num_channels = input_num_channels
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.downsample_factor = downsample_factor
        self.beta = beta

        self.downsampler = DownSample(input_num_channels, downsample_factor)
        self.upsampler_depth = UpSample(num_channels=1, factor=downsample_factor)
        self.upsampler_reconstruction = UpSample(input_num_channels, downsample_factor)

        self.encoder = model_lst.elic_lst_downsample(input_num_channels, latent_num_channels)
        self.entropy_model = GaussianEntropyModel(latent_num_channels)
        self.decoder_depth = model_lst.elic_lst_upsample(
            latent_num_channels, input_num_channels, num_blocks=1, activation='elu'
        )
        self.classifier_depth = segmentation.lraspp_mobilenet_v3_large(num_classes=1)
        self.decoder_reconstruction = model_lst.elic_lst_upsample(
            latent_num_channels, input_num_channels, activation='elu'
        )

    def forward(self, *args, **_) -> Tuple[Tensor, Tensor, Tensor]:
        inputs = functional.interpolate(*args, scale_factor=1 / self.downsampler.factor, mode=self.downsampler.mode)
        representations, likelihoods = self.entropy_model(self.encoder(self.downsampler(*args)))

        return inputs, representations, likelihoods

    def predict_depth(self, representations: Tensor) -> Tensor:
        return self.classifier_depth(self.decoder_depth(representations))['out']

    def predict_reconstruction(self, representations: Tensor) -> Tensor:
        return self.decoder_reconstruction(representations)

    def calculate_depth_loss(self, predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        loss = self.upsampler_depth(predictions)
        loss = functions.mse_loss(loss[:, 0], targets, mask)
        loss = torch.sqrt(loss) * 128

        return loss

    def calculate_reconstruction_loss(self, predictions: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        loss = self.upsampler_reconstruction(predictions)
        loss = functions.calculate_reconstruction_loss(loss, targets)

        return loss

    def calculate_metrics(self, batch: BatchTypeDepth, *_) -> AttributeDict[Tensor]:
        inputs, (targets_depth, mask_depth) = batch

        representations = self.downsampler(inputs)
        representations = self.encoder(representations)
        representations, likelihoods = self.entropy_model(representations)

        predictions_depth = self.predict_depth(representations)
        predictions_reconstruction = self.predict_reconstruction(representations)

        bpp = functions.calculate_likelihood_bpp_fixed(likelihoods, inputs.shape)
        distortion_depth = self.calculate_depth_loss(predictions_depth, targets_depth, mask_depth)
        distortion_reconstruction, psnr = self.calculate_reconstruction_loss(predictions_reconstruction, inputs)

        loss = self.alpha * bpp + distortion_depth + self.beta * distortion_reconstruction

        rate_distortion = self.alpha * bpp + distortion_depth

        metrics = AttributeDict({
            'Loss': loss,
            'BPP': bpp,
            'Distortion Depth': distortion_depth,
            'Distortion Reconstruction': distortion_reconstruction,
            'PSNR': psnr,
            'Rate Distortion': rate_distortion,
        })

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), self.learning_rate)

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Test {name}', value)


class DetectionForReconstructionBaseline(LightningModule):
    def __init__(
        self,
        input_num_channels: int,
        latent_num_channels: int,
        num_classes: int,
        alpha: float,
        learning_rate: float,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.input_num_channels = input_num_channels
        self.latent_num_channels = latent_num_channels
        self.num_classes = num_classes
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.encoder = model_lst.elic_lst_downsample(input_num_channels, latent_num_channels)
        self.entropy_model = GaussianEntropyModel(latent_num_channels)
        self.decoder_detection = model_lst.elic_lst_upsample(
            latent_num_channels, input_num_channels, num_blocks=1, activation='elu'
        )
        self.classifier_detection = detection.fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, num_classes=num_classes
        )
        self.decoder_reconstruction = model_lst.elic_lst_upsample(
            latent_num_channels, input_num_channels, activation='elu'
        )

        self.mean_average_precision = MeanAveragePrecision()

    def forward(self, *args, **_) -> Tuple[ImageList, Tensor, Tensor]:
        inputs, _ = self.classifier_detection.transform(*args)
        representations = self.encoder(inputs.tensors)
        representations, likelihoods = self.entropy_model(representations)

        return inputs, representations, likelihoods

    @staticmethod
    def targets_to_map(targets: Sequence[TargetTypeDetection]) -> List[Dict[str, Tensor]]:
        return [{'boxes': boxes, 'labels': labels} for boxes, labels, _ in targets]

    def transform(
        self,
        images: Sequence[Tensor],
        targets: Sequence[Dict[str, Tensor]],
    ) -> Tuple[Tensor, List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        original_image_sizes = [tuple(image.shape[-2:]) for image in images]

        images, targets = self.classifier_detection.transform(images, targets)
        image_sizes = images.image_sizes
        images = images.tensors

        return images, targets, original_image_sizes, image_sizes

    def classify(
        self,
        representations: Tensor,
        targets: Sequence[Dict[str, Tensor]],
        original_image_sizes: Sequence[Tuple[int, int]],
        image_sizes: List[Tuple[int, int]],
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        images = self.decoder_detection(representations)

        features = self.classifier_detection.backbone(images)
        if isinstance(features, Tensor):
            features = OrderedDict([('0', features)])

        images = ImageList(images, image_sizes)

        proposals, proposal_losses = self.classifier_detection.rpn(images, features, targets)
        detections, detector_losses = self.classifier_detection.roi_heads(features, proposals, image_sizes, targets)

        losses = detector_losses | proposal_losses
        losses = sum(losses.values())

        detections = self.classifier_detection.transform.postprocess(detections, image_sizes, original_image_sizes)

        return losses, detections

    def calculate_training_metrics(self, batch: BatchTypeDetection) -> AttributeDict[Tensor]:
        images, targets = batch
        targets = self.targets_to_map(targets)

        images, targets_transformed, original_image_sizes, image_sizes = self.transform(images, targets)

        representations = self.encoder(images)
        representations, likelihoods = self.entropy_model(representations)
        distortion, _ = self.classify(representations, targets_transformed, original_image_sizes, image_sizes)

        assert distortion is not None

        bpp = functions.calculate_likelihood_bpp_dynamic(likelihoods, image_sizes)

        reconstructions = self.decoder_reconstruction(torch.detach(representations))
        rmse, psnr = functions.calculate_reconstruction_loss(reconstructions, images)

        loss = distortion + self.alpha * bpp + rmse

        metrics = AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'BPP': bpp,
            'RMSE': rmse,
            'PSNR': psnr,
        })

        return metrics

    def calculate_test_metrics(self, batch: BatchTypeDetection) -> AttributeDict[Tensor]:
        images, targets = batch
        targets = self.targets_to_map(targets)

        images, targets_transformed, original_image_sizes, image_sizes = self.transform(images, targets)

        representations = self.encoder(images)
        representations, likelihoods = self.entropy_model(representations)
        distortion, predictions = self.classify(representations, targets_transformed, original_image_sizes, image_sizes)

        assert predictions is not None

        self.mean_average_precision(predictions, targets)

        bpp = functions.calculate_likelihood_bpp_dynamic(likelihoods, image_sizes)

        reconstructions = self.decoder_reconstruction(representations)
        rmse, psnr = functions.calculate_reconstruction_loss(reconstructions, images)

        rate_distortion = distortion + self.alpha * bpp

        metrics = AttributeDict({'RMSE': rmse, 'PSNR': psnr, 'BPP': bpp, 'Rate Distortion': rate_distortion})

        return metrics

    def configure_optimizers(self):
        return Adam(self.parameters(), self.learning_rate)

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_test_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_test_metrics(args[0])

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


class DetectionForReconstructionAndrade(LightningModule):
    def __init__(
        self,
        input_num_channels: int,
        latent_num_channels: int,
        num_classes: int,
        alpha: float,
        learning_rate: float,
        beta: float = 0.1,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.input_num_channels = input_num_channels
        self.latent_num_channels = latent_num_channels
        self.num_classes = num_classes
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.beta = beta

        self.encoder = model_lst.elic_lst_downsample(input_num_channels, latent_num_channels)
        self.entropy_model = GaussianEntropyModel(latent_num_channels)
        self.decoder_detection = model_lst.elic_lst_upsample(
            latent_num_channels, input_num_channels, num_blocks=1, activation='elu'
        )
        self.classifier_detection = detection.fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, num_classes=num_classes
        )
        self.decoder_reconstruction = model_lst.elic_lst_upsample(
            latent_num_channels, input_num_channels, activation='elu'
        )

        self.mean_average_precision = MeanAveragePrecision()

    def forward(self, *args, **_) -> Tuple[ImageList, Tensor, Tensor]:
        inputs, _ = self.classifier_detection.transform(*args)
        representations = self.encoder(inputs.tensors)
        representations, likelihoods = self.entropy_model(representations)

        return inputs, representations, likelihoods

    @staticmethod
    def targets_to_map(targets: Sequence[TargetTypeDetection]) -> List[Dict[str, Tensor]]:
        return [{'boxes': boxes, 'labels': labels} for boxes, labels, _ in targets]

    def transform(
        self,
        images: Sequence[Tensor],
        targets: Sequence[Dict[str, Tensor]],
    ) -> Tuple[Tensor, List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        original_image_sizes = [tuple(image.shape[-2:]) for image in images]

        images, targets = self.classifier_detection.transform(images, targets)
        image_sizes = images.image_sizes
        images = images.tensors

        return images, targets, original_image_sizes, image_sizes

    def classify(
        self,
        representations: Tensor,
        targets: Sequence[Dict[str, Tensor]],
        original_image_sizes: Sequence[Tuple[int, int]],
        image_sizes: List[Tuple[int, int]],
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        images = self.decoder_detection(representations)

        features = self.classifier_detection.backbone(images)
        if isinstance(features, Tensor):
            features = OrderedDict([('0', features)])

        images = ImageList(images, image_sizes)

        proposals, proposal_losses = self.classifier_detection.rpn(images, features, targets)
        detections, detector_losses = self.classifier_detection.roi_heads(features, proposals, image_sizes, targets)

        losses = detector_losses | proposal_losses
        losses = sum(losses.values())

        detections = self.classifier_detection.transform.postprocess(detections, image_sizes, original_image_sizes)

        return losses, detections

    def calculate_training_metrics(self, batch: BatchTypeDetection) -> AttributeDict[Tensor]:
        images, targets = batch
        targets = self.targets_to_map(targets)

        images, targets_transformed, original_image_sizes, image_sizes = self.transform(images, targets)

        representations = self.encoder(images)
        representations, likelihoods = self.entropy_model(representations)
        distortion, _ = self.classify(representations, targets_transformed, original_image_sizes, image_sizes)

        assert distortion is not None

        bpp = functions.calculate_likelihood_bpp_dynamic(likelihoods, image_sizes)

        reconstructions = self.decoder_reconstruction(representations)
        rmse, psnr = functions.calculate_reconstruction_loss(reconstructions, images)

        loss = distortion + self.alpha * bpp + self.beta * rmse

        metrics = AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'BPP': bpp,
            'RMSE': rmse,
            'PSNR': psnr,
        })

        return metrics

    def calculate_test_metrics(self, batch: BatchTypeDetection) -> AttributeDict[Tensor]:
        images, targets = batch
        targets = self.targets_to_map(targets)

        images, targets_transformed, original_image_sizes, image_sizes = self.transform(images, targets)

        representations = self.encoder(images)
        representations, likelihoods = self.entropy_model(representations)
        distortion, predictions = self.classify(representations, targets_transformed, original_image_sizes, image_sizes)

        assert predictions is not None

        self.mean_average_precision(predictions, targets)

        bpp = functions.calculate_likelihood_bpp_dynamic(likelihoods, image_sizes)

        reconstructions = self.decoder_reconstruction(representations)
        rmse, psnr = functions.calculate_reconstruction_loss(reconstructions, images)

        rate_distortion = distortion + self.alpha * bpp

        metrics = AttributeDict({'RMSE': rmse, 'PSNR': psnr, 'BPP': bpp, 'Rate Distortion': rate_distortion})

        return metrics

    def configure_optimizers(self):
        return Adam(self.parameters(), self.learning_rate)

    def training_step(self, *args, **_) -> Tensor:
        metrics = self.calculate_training_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_test_metrics(args[0])

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_test_metrics(args[0])

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
