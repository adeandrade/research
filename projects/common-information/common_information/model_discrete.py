import torch
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.nn import Conv2d, Sequential
from torch.optim.adamw import AdamW

import common_information.functions as functions
import common_information.model_ci_shuffle as model_ci_shuffle
from common_information.model_ci_shuffle import CodecType
from common_information.model_lst import ResidualBottleneckBlockStack


class Discrete(LightningModule):
    def __init__(
        self,
        input_num_channels: int,
        latent_num_channels: int,
        alpha: float,
        beta: float,
        learning_rate: float,
        gamma: float = 0.0,
        scale: float = 1.0,
        codec_type: CodecType = CodecType.SHARED,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.scale = scale

        self.codec = model_ci_shuffle.get_codec(codec_type, input_num_channels, latent_num_channels, alpha, beta, scale)

        self.classifier_a = Sequential(
            ResidualBottleneckBlockStack(input_num_channels, 1),
            Conv2d(input_num_channels, 1, 5, 1, 'same'),
        )
        self.classifier_b = Sequential(
            ResidualBottleneckBlockStack(input_num_channels, 1),
            Conv2d(input_num_channels, 1, 5, 1, 'same'),
        )

    def classify_a(self, reconstructions: Tensor, masks: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        predictions = self.classifier_a(reconstructions)[:, 0]

        distortion = functions.calculate_reconstruction_loss(predictions, targets, masks, self.scale)

        return distortion, predictions

    def classify_b(self, reconstructions: Tensor, masks: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        predictions = self.classifier_b(reconstructions)[:, 0]

        distortion = functions.calculate_reconstruction_loss(predictions, targets, masks, self.scale)

        return distortion, predictions

    def calculate_metrics(self, batch: tuple[Tensor, Tensor, Tensor]) -> AttributeDict[Tensor]:
        inputs, targets_a, targets_b = batch

        masks = torch.ones_like(inputs[:, 0])

        (reconstructions_a, reconstructions_b), metrics = self.codec(inputs, masks)

        distortion_a, *_ = self.classify_a(reconstructions_a, masks, targets_a)
        distortion_b, *_ = self.classify_b(reconstructions_b, masks, targets_b)

        rate_distortion = metrics.bpp_scaled + distortion_a + distortion_b

        loss = rate_distortion + self.gamma * (metrics.rmse_a + metrics.rmse_b)

        if 'Auxiliary Loss' in metrics:
            loss = loss + metrics.auxiliary_loss

        metrics |= AttributeDict({
            'Loss': loss,
            'Distortion A': distortion_a,
            'Distortion B': distortion_b,
            'Rate Distortion': rate_distortion,
        })

        return metrics

    def configure_optimizers(self, weight_decay: float = 0.0) -> AdamW:
        return AdamW(self.parameters(), self.learning_rate, weight_decay=weight_decay)

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
