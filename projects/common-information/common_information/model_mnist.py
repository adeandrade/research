import sfu_torch_lib.state as state
import torch
import torch.nn.functional as functional
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.nn import Flatten, Linear, Sequential, Sigmoid
from torch.optim.adamw import AdamW
from torchmetrics import Accuracy

import common_information.functions as functions
import common_information.model_ci as model_ci
import common_information.model_lst as model_lst
from common_information.model_ci import CodecType, SharedCodec


class MNISTColored(LightningModule):
    def __init__(
        self,
        input_num_channels: int,
        latent_num_channels: int,
        alpha: float,
        beta: float,
        learning_rate: float,
        gamma: float = 0.0,
        scale: float = 255.0,
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

        self.codec = model_ci.get_codec(codec_type, input_num_channels, latent_num_channels, alpha, beta, scale)

        self.classifier_a = Sequential(
            Flatten(),
            Linear(3072, 3072),
            Sigmoid(),
            Linear(3072, 10),
        )
        self.classifier_b = Sequential(
            Flatten(),
            Linear(3072, 3072),
            Sigmoid(),
            Linear(3072, 10),
        )

        self.accuracy_digit = Accuracy('multiclass', num_classes=10)
        self.accuracy_color = Accuracy('multiclass', num_classes=10)

    def classify_a(self, reconstructions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        predictions = self.classifier_a(reconstructions)

        distortion = functional.cross_entropy(predictions, targets)

        return distortion, predictions

    def classify_b(self, reconstructions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        predictions = self.classifier_b(reconstructions)

        distortion = functional.cross_entropy(predictions, targets)

        return distortion, predictions

    def calculate_metrics(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[tuple[Tensor, Tensor], AttributeDict[Tensor]]:
        inputs, targets_a, targets_b = batch

        masks = torch.ones_like(inputs[:, 0])

        (reconstructions_a, reconstructions_b), metrics = self.codec(inputs, masks)

        distortion_a, predictions_a = self.classify_a(reconstructions_a, targets_a)
        distortion_b, predictions_b = self.classify_b(reconstructions_b, targets_b)

        rate_distortion = metrics.bpp_scaled + distortion_a + distortion_b

        loss = rate_distortion + self.gamma * (metrics.rmse_a + metrics.rmse_b)

        if 'Auxiliary Loss' in metrics:
            loss = loss + self.alpha * metrics.auxiliary_loss

        metrics |= AttributeDict({
            'Loss': loss,
            'Distortion A': distortion_a,
            'Distortion B': distortion_b,
            'Rate Distortion': rate_distortion,
        })

        return (predictions_a, predictions_b), metrics

    def configure_optimizers(self, weight_decay: float = 0.0) -> AdamW:
        return AdamW(self.parameters(), self.learning_rate, weight_decay=weight_decay)

    def training_step(self, *args, **_) -> Tensor:
        *_, metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()})

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        (*_, targets_digit, targets_color), *_ = args

        (predictions_digits, predictions_color), metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Validation {name}': value for name, value in metrics.items()})

        self.accuracy_digit(predictions_digits, targets_digit)
        self.log('Validation Accuracy Digit', self.accuracy_digit, on_step=False, on_epoch=True)

        self.accuracy_color(predictions_color, targets_color)
        self.log('Validation Accuracy Color', self.accuracy_color, on_step=False, on_epoch=True)

    def test_step(self, *args, **_) -> None:
        *_, metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Testing {name}': value for name, value in metrics.items()})


class MNISTColoredReconstruction(LightningModule):
    def __init__(
        self,
        run_id_model: str,
        learning_rate: float,
        scale: float = 255.0,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.run_id_model = run_id_model
        self.learning_rate = learning_rate
        self.scale = scale

        self.codec = state.load_model(run_id_model, MNISTColored).codec

        assert isinstance(self.codec, SharedCodec)

        num_channels_block = self.codec.latent_num_channels // 2
        input_num_channels = self.codec.input_num_channels

        self.classifier_a = model_lst.elic_lst_upsample(num_channels_block, input_num_channels, num_blocks=3)
        self.classifier_b = model_lst.elic_lst_upsample(num_channels_block, input_num_channels, num_blocks=3)
        self.classifier_c = model_lst.elic_lst_upsample(num_channels_block, input_num_channels, num_blocks=3)

    def classify_a(self, reconstructions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        predictions = self.classifier_a(reconstructions)

        distortion = functions.calculate_reconstruction_loss(predictions, targets, scale=self.scale)

        return distortion, predictions

    def classify_b(self, reconstructions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        predictions = self.classifier_b(reconstructions)

        distortion = functions.calculate_reconstruction_loss(predictions, targets, scale=self.scale)

        return distortion, predictions

    def classify_c(self, reconstructions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        predictions = self.classifier_c(reconstructions)

        distortion = functions.calculate_reconstruction_loss(predictions, targets, scale=self.scale)

        return distortion, predictions

    def reconstruct(self, inputs: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        assert isinstance(self.codec, SharedCodec)
        (representations_a, representations_b, representations_c), *_ = self.codec.get_y(inputs)

        representations_a = torch.detach(representations_a)
        representations_b = torch.detach(representations_b)
        representations_c = torch.detach(representations_c)

        distortion_a, reconstructions_a = self.classify_a(representations_a, inputs)
        distortion_b, reconstructions_b = self.classify_b(representations_b, inputs)
        distortion_c, reconstructions_c = self.classify_c(representations_c, inputs)

        return (distortion_a, distortion_b, distortion_c), (reconstructions_a, reconstructions_b, reconstructions_c)

    def calculate_metrics(self, batch: tuple[Tensor, Tensor, Tensor]) -> AttributeDict[Tensor]:
        inputs, *_ = batch

        (distortion_a, distortion_b, distortion_c), *_ = self.reconstruct(inputs)

        loss = distortion_a + distortion_b + distortion_c

        metrics = AttributeDict({
            'Loss': loss,
            'Distortion A': distortion_a,
            'Distortion B': distortion_b,
            'Distortion C': distortion_c,
        })

        return metrics

    def configure_optimizers(self, weight_decay: float = 0.0) -> AdamW:
        return AdamW(self.parameters(), self.learning_rate, weight_decay=weight_decay)

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
