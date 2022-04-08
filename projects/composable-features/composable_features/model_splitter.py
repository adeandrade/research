import functools
from typing import Tuple, Callable, List

import sfu_torch_lib.optimization as optimization
import torch
import torch.nn.functional as functional
from pytorch_lightning import LightningModule
from sfu_torch_lib.optimization import LossArguments
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.distributions import Bernoulli
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torchmetrics import Accuracy, JaccardIndex

import composable_features.functions as functions
from composable_features.model_entropy import AutoregressiveEntropyModel
from composable_features.processing import BatchType


TripleTensor = Tuple[Tensor, Tensor, Tensor]


class InformationSplitter(LightningModule):
    def __init__(
            self,
            transformer: Module,
            entropy_model: Module,
            reconstruction_model: Module,
            segmentation_model: Module,
            depth_model: Module,
            kernel_encoder: Module,
            bits_model: AutoregressiveEntropyModel,
            ignore_index: int,
            num_classes: int,
            alpha: float,
            gamma: float,
            temperature: float,
            threshold: float,
            learning_rate: float,
            momentum: float,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.transformer = transformer
        self.entropy_model = entropy_model
        self.reconstruction_model = reconstruction_model
        self.segmentation_model = segmentation_model
        self.depth_model = depth_model
        self.kernel_encoder = kernel_encoder
        self.bits_model = bits_model

        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.threshold = threshold

        self.learning_rate = learning_rate
        self.momentum = momentum

        self.accuracy_segmentation = Accuracy(num_classes=num_classes + 1, ignore_index=ignore_index)
        self.iou_segmentation = JaccardIndex(num_classes + 1, ignore_index=ignore_index)

    @staticmethod
    def calculate_penalty(coefficients: Tensor) -> Tensor:
        _, num_heads = coefficients.shape

        penalty = torch.einsum('ch,cd->hd', coefficients, coefficients)
        penalty = torch.triu(penalty, diagonal=1)
        penalty = torch.square(penalty)
        penalty = torch.sum(penalty)

        return penalty

    def get_coefficients(self, kernels_source: Tensor, kernels_target: Tensor) -> Tuple[Tensor, Tensor]:
        num_channels_source = kernels_source.shape[0]
        num_channels_target = kernels_target.shape[1]

        kernels_source = torch.reshape(kernels_source, shape=(num_channels_source, -1))
        kernels_source = torch.repeat_interleave(kernels_source, repeats=num_channels_target, dim=0)

        kernels_target = torch.permute(kernels_target, dims=(1, 0, 2, 3))
        kernels_target = torch.reshape(kernels_target, shape=(num_channels_target, -1))
        kernels_target = torch.tile(kernels_target, dims=(num_channels_source, 1))

        coefficients = torch.cat((kernels_source, kernels_target), dim=1)
        coefficients = self.kernel_encoder(coefficients)
        coefficients = torch.reshape(coefficients, shape=(num_channels_source, num_channels_target))
        coefficients = torch.softmax(coefficients / self.temperature, dim=0)

        penalty = self.calculate_penalty(coefficients)

        if not self.training:
            zero = torch.zeros((), dtype=coefficients.dtype, device=coefficients.device)
            coefficients = torch.where(coefficients > self.threshold, coefficients, zero)

        return coefficients, penalty

    @staticmethod
    def combine_activations(activations: Tensor, coefficients: Tensor) -> Tensor:
        return torch.einsum('bihw,io->bohw', activations, coefficients)

    def get_task_coefficients(self, task: Module) -> Tuple[Tensor, Tensor]:
        kernels_source = torch.detach(functions.get_nested_attribute(self.transformer, index=-1, name='weight'))
        kernels_task = torch.detach(functions.get_nested_attribute(task, index=0, name='weight'))

        coefficients, penalty = self.get_coefficients(kernels_source, kernels_task)

        return coefficients, penalty

    def select_kernels(self, activations: Tensor, task: Module) -> Tuple[Tensor, Tensor]:
        coefficients, penalty = self.get_task_coefficients(task)
        activations = self.combine_activations(activations, coefficients)

        return activations, penalty

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        return self.transformer(inputs)

    def calculate_task_loss(
            self,
            representations: Tensor,
            task: Module,
            calculate_loss: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Tensor]:

        activations, penalty = self.select_kernels(representations, task)
        predictions = task(activations)
        loss = calculate_loss(predictions)

        return loss, penalty

    @staticmethod
    def calculate_reconstruction_loss(predictions: Tensor, targets: Tensor) -> Tensor:
        loss = functional.mse_loss(predictions, targets)
        loss = torch.sqrt(loss) * 255

        return loss

    def calculate_segmentation_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = functional.cross_entropy(predictions, targets, ignore_index=self.ignore_index)
        loss *= self.num_classes

        self.accuracy_segmentation(functions.add_ninf_channel(predictions), targets)
        self.iou_segmentation(functions.add_ninf_channel(predictions), targets)

        return loss

    @staticmethod
    def calculate_depth_loss(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        loss = functions.mse_loss(predictions[:, 0], targets, mask)
        loss = torch.sqrt(loss) * 128

        return loss

    def calculate_losses(self, batch: BatchType) -> AttributeDict[Tensor]:
        inputs, targets_segmentation, (targets_depth, mask_depth) = batch

        representations = self(inputs)

        representations_quantized, likelihoods = self.entropy_model(representations)
        entropy = functions.calculate_likelihood_entropy(likelihoods)

        _, likelihoods = self.bits_model(torch.detach(representations))
        bits = functions.calculate_likelihood_entropy(likelihoods, normalize=False)

        loss_reconstruction, penalty_reconstruction = self.calculate_task_loss(
            representations=representations_quantized,
            task=self.reconstruction_model,
            calculate_loss=functools.partial(self.calculate_reconstruction_loss, targets=inputs),
        )
        loss_segmentation, penalty_segmentation = self.calculate_task_loss(
            representations=representations_quantized,
            task=self.segmentation_model,
            calculate_loss=functools.partial(self.calculate_segmentation_loss, targets=targets_segmentation),
        )
        loss_depth, penalty_depth = self.calculate_task_loss(
            representations=representations_quantized,
            task=self.depth_model,
            calculate_loss=functools.partial(self.calculate_depth_loss, targets=targets_depth, mask=mask_depth),
        )

        loss_tasks = loss_reconstruction + loss_segmentation + loss_depth
        penalties = penalty_reconstruction + penalty_segmentation + penalty_depth

        metrics = AttributeDict({
            'Loss Tasks': loss_tasks,
            'Loss Reconstruction': loss_reconstruction,
            'Loss Segmentation': loss_segmentation,
            'Loss Depth': loss_depth,
            'Entropy': entropy,
            'Penalties': penalties,
            'Bits': bits,
            'Segmentation Accuracy': self.accuracy_segmentation,
            'Segmentation IoU': self.iou_segmentation,
        })

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate, betas=(self.momentum, 0.999))

    def training_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        optimization.adaptive_clipping(
            optimizer=self.optimizers(),
            losses=(
                LossArguments(metrics.loss_tasks, metrics.entropy, self.alpha),
                LossArguments(self.gamma * metrics.penalties + metrics.bits),
            ),
        )

    def validation_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def calculate_bit_rate(self, batch: BatchType) -> Tuple[float, List[float]]:
        inputs, _, _ = batch

        representations = self(inputs)

        bit_rate = sum(
            len(string) * 8
            for string
            in self.bits_model.compress(representations)
        ) / inputs.shape[0]

        _, likelihoods = self.bits_model(representations)
        entropies = functions.calculate_likelihood_entropy(likelihoods, reduce=False, normalize=False).tolist()

        return bit_rate, entropies

    def test_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)

        bit_rate, entropies = self.calculate_bit_rate(batch)

        self.log('Test Bits Rate', bit_rate)

        for index, entropy in enumerate(entropies):
            self.log(f'Test Entropy Channel {index}', entropy)

    def on_validation_start(self) -> None:
        self.accuracy_segmentation.reset()
        self.iou_segmentation.reset()

    def get_task_probabilities(self) -> Tuple[Tensor, Tensor, Tensor]:
        coefficients_reconstruction, _ = self.get_task_coefficients(self.reconstruction_model)
        coefficients_segmentation, _ = self.get_task_coefficients(self.segmentation_model)
        coefficients_depth, _ = self.get_task_coefficients(self.depth_model)

        probabilities_reconstruction = torch.amax(coefficients_reconstruction, dim=1)
        probabilities_segmentation = torch.amax(coefficients_segmentation, dim=1)
        probabilities_depth = torch.amax(coefficients_depth, dim=1)

        return probabilities_reconstruction, probabilities_segmentation, probabilities_depth


class InformationSplitterPolicyGradients(LightningModule):
    def __init__(
            self,
            transformer: Module,
            entropy_model: Module,
            reconstruction_model: Module,
            segmentation_model: Module,
            depth_model: Module,
            kernel_encoder: Module,
            bits_model: AutoregressiveEntropyModel,
            ignore_index: int,
            num_classes: int,
            alpha: float,
            gamma: float,
            learning_rate: float,
            momentum: float,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.transformer = transformer
        self.entropy_model = entropy_model
        self.reconstruction_model = reconstruction_model
        self.segmentation_model = segmentation_model
        self.depth_model = depth_model
        self.kernel_encoder = kernel_encoder
        self.bits_model = bits_model

        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

        self.learning_rate = learning_rate
        self.momentum = momentum

        self.accuracy_segmentation = Accuracy(num_classes=num_classes + 1, ignore_index=ignore_index)
        self.iou_segmentation = JaccardIndex(num_classes + 1, ignore_index=ignore_index)

    def get_logits(self, kernels_source: Tensor, kernels_target: Tensor) -> Tensor:
        num_channels_source = kernels_source.shape[0]
        num_channels_target = kernels_target.shape[1]

        assert num_channels_source == num_channels_target

        kernels_source = torch.reshape(kernels_source, shape=(num_channels_source, -1))

        kernels_target = torch.permute(kernels_target, dims=(1, 0, 2, 3))
        kernels_target = torch.reshape(kernels_target, shape=(num_channels_target, -1))

        logits = torch.cat((kernels_source, kernels_target), dim=1)
        logits = self.kernel_encoder(logits)[:, 0]

        return logits

    def get_task_logits(self, task: Module) -> Tensor:
        kernels_source = torch.detach(functions.get_nested_attribute(self.transformer, index=-1, name='weight'))
        kernels_task = torch.detach(functions.get_nested_attribute(task, index=0, name='weight'))

        logits = self.get_logits(kernels_source, kernels_task)

        return logits

    def get_actions(self, logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        actions_task = Bernoulli(logits=logits).sample() if self.training else (torch.sigmoid(logits) > .5).float()

        uniform_noise = torch.rand_like(logits)

        probabilities_first = torch.sigmoid(-logits)
        actions_first = (uniform_noise > probabilities_first).float()

        probabilities_second = 1 - probabilities_first
        actions_second = (uniform_noise < probabilities_second).float()

        return actions_task, actions_first, actions_second, uniform_noise

    @staticmethod
    def combine_activations(activations: Tensor, actions: Tensor) -> Tensor:
        return torch.einsum('bdhw,d->bdhw', activations, actions)

    def select_kernels(self, activations: Tensor, task: Module) -> Tuple[Tensor, Tensor, TripleTensor, TripleTensor]:
        logits = self.get_task_logits(task)

        actions_task, actions_first, actions_second, noise = self.get_actions(logits)

        activations_task = self.combine_activations(activations, actions_task)
        activations_first = self.combine_activations(activations, actions_first)
        activations_second = self.combine_activations(activations, actions_second)
        combined_activations = (activations_task, activations_first, activations_second)

        ratio_channels_task = torch.mean(actions_task)
        ratio_channels_first = torch.mean(actions_first)
        ratio_channels_second = torch.mean(actions_second)
        ratio_channels = (ratio_channels_task, ratio_channels_first, ratio_channels_second)

        return logits, noise, combined_activations, ratio_channels

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        return self.transformer(inputs)

    def calculate_task_loss(
            self,
            representations: Tensor,
            task: Module,
            calculate_loss: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:

        logits, noise, activations, ratio_channels = self.select_kernels(representations, task)
        activations_task, activations_first, activations_second = activations
        ratio_channels_task, ratio_channels_first, ratio_channels_second = ratio_channels

        predictions_task = task(activations_task)
        loss_task = calculate_loss(predictions_task)
        loss_task = torch.mean(loss_task)

        loss_first = calculate_loss(task(activations_first)) + ratio_channels_first * self.gamma
        loss_second = calculate_loss(task(activations_second)) + ratio_channels_second * self.gamma
        loss_policy = loss_first - loss_second
        loss_policy = loss_policy[:, None] * (noise - .5)
        loss_policy = torch.detach(loss_policy) * logits
        loss_policy = torch.mean(loss_policy)

        return loss_task, loss_policy, ratio_channels_task

    @staticmethod
    def calculate_reconstruction_loss(predictions: Tensor, targets: Tensor) -> Tensor:
        loss = functions.batch_mse_loss(predictions, targets)
        loss = torch.sqrt(loss) * 255

        return loss

    def calculate_segmentation_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = functions.batch_cross_entropy(predictions, targets, ignore_index=self.ignore_index)
        loss *= self.num_classes

        self.accuracy_segmentation(functions.add_ninf_channel(predictions), targets)
        self.iou_segmentation(functions.add_ninf_channel(predictions), targets)

        return loss

    @staticmethod
    def calculate_depth_loss(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        loss = functions.batch_mse_loss_masked(predictions[:, 0], targets, mask)
        loss = torch.sqrt(loss) * 128

        return loss

    def calculate_losses(self, batch: BatchType) -> AttributeDict[Tensor]:
        inputs, targets_segmentation, (targets_depth, mask_depth) = batch

        representations = self(inputs)

        representations_quantized, likelihoods = self.entropy_model(representations)
        entropy = functions.calculate_likelihood_entropy(likelihoods)

        _, likelihoods = self.bits_model(torch.detach(representations))
        bits = functions.calculate_likelihood_entropy(likelihoods, normalize=False)

        loss_reconstruction, policy_loss_reconstruction, ratio_channels_reconstruction = self.calculate_task_loss(
            representations=representations_quantized,
            task=self.reconstruction_model,
            calculate_loss=functools.partial(self.calculate_reconstruction_loss, targets=inputs),
        )
        loss_segmentation, policy_loss_segmentation, ratio_channels_segmentation = self.calculate_task_loss(
            representations=representations_quantized,
            task=self.segmentation_model,
            calculate_loss=functools.partial(self.calculate_segmentation_loss, targets=targets_segmentation),
        )
        loss_depth, policy_loss_depth, ratio_channels_depth = self.calculate_task_loss(
            representations=representations_quantized,
            task=self.depth_model,
            calculate_loss=functools.partial(self.calculate_depth_loss, targets=targets_depth, mask=mask_depth),
        )

        loss_tasks = loss_reconstruction + loss_segmentation + loss_depth
        loss_policies = policy_loss_reconstruction + policy_loss_segmentation + policy_loss_depth

        metrics = AttributeDict({
            'Loss Tasks': loss_tasks,
            'Loss Reconstruction': loss_reconstruction,
            'Loss Segmentation': loss_segmentation,
            'Loss Depth': loss_depth,
            'Entropy': entropy,
            'Bits': bits,
            'Loss Policies': loss_policies,
            'Policy Reconstruction': policy_loss_reconstruction,
            'Policy Segmentation': policy_loss_segmentation,
            'Policy Depth': policy_loss_depth,
            'Ratio Channels Reconstruction': ratio_channels_reconstruction,
            'Ratio Channels Segmentation': ratio_channels_segmentation,
            'Ratio Channels Depth': ratio_channels_depth,
            'Segmentation Accuracy': self.accuracy_segmentation,
            'Segmentation IoU': self.iou_segmentation,
        })

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate, betas=(self.momentum, 0.999))

    def training_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        optimization.adaptive_clipping(
            optimizer=self.optimizers(),
            losses=(
                LossArguments(metrics.loss_tasks, metrics.entropy, self.alpha),
                LossArguments(metrics.loss_policies + metrics.bits),
            ),
        )

    def validation_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def calculate_bit_rate(self, batch: BatchType) -> Tuple[float, List[float]]:
        inputs, _, _ = batch

        representations = self(inputs)

        bit_rate = sum(
            len(string) * 8
            for string
            in self.bits_model.compress(representations)
        ) / inputs.shape[0]

        _, likelihoods = self.bits_model(representations)
        entropies = functions.calculate_likelihood_entropy(likelihoods, reduce=False, normalize=False).tolist()

        return bit_rate, entropies

    def test_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)

        bit_rate, entropies = self.calculate_bit_rate(batch)

        self.log('Test Bits Rate', bit_rate)

        for index, entropy in enumerate(entropies):
            self.log(f'Test Entropy Channel {index}', entropy)

    def on_validation_start(self) -> None:
        self.accuracy_segmentation.reset()
        self.iou_segmentation.reset()

    def get_task_probabilities(self) -> Tuple[Tensor, Tensor, Tensor]:
        logits_reconstruction = self.get_task_logits(self.reconstruction_model)
        logits_segmentation = self.get_task_logits(self.segmentation_model)
        logits_depth = self.get_task_logits(self.depth_model)

        probabilities_reconstruction = torch.sigmoid(logits_reconstruction)
        probabilities_segmentation = torch.sigmoid(logits_segmentation)
        probabilities_depth = torch.sigmoid(logits_depth)

        return probabilities_reconstruction, probabilities_segmentation, probabilities_depth


class Baseline(LightningModule):
    def __init__(
            self,
            transformer: Module,
            entropy_model: Module,
            reconstruction_model: Module,
            segmentation_model: Module,
            depth_model: Module,
            bits_model: AutoregressiveEntropyModel,
            ignore_index: int,
            num_classes: int,
            alpha: float,
            learning_rate: float,
            momentum: float,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.transformer = transformer
        self.entropy_model = entropy_model
        self.reconstruction_model = reconstruction_model
        self.segmentation_model = segmentation_model
        self.depth_model = depth_model
        self.bits_model = bits_model

        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.accuracy_segmentation = Accuracy(num_classes=num_classes + 1, ignore_index=ignore_index)
        self.iou_segmentation = JaccardIndex(num_classes + 1, ignore_index=ignore_index)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        return self.transformer(inputs)

    @staticmethod
    def calculate_task_loss(
            representations: Tensor,
            task: Module,
            calculate_loss: Callable[[Tensor], Tensor],
    ) -> Tensor:

        return calculate_loss(task(representations))

    @staticmethod
    def calculate_reconstruction_loss(predictions: Tensor, targets: Tensor) -> Tensor:
        loss = functional.mse_loss(predictions, targets)
        loss = torch.sqrt(loss) * 255

        return loss

    def calculate_segmentation_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = functional.cross_entropy(predictions, targets, ignore_index=self.ignore_index)
        loss *= self.num_classes

        self.accuracy_segmentation(functions.add_ninf_channel(predictions), targets)
        self.iou_segmentation(functions.add_ninf_channel(predictions), targets)

        return loss

    @staticmethod
    def calculate_depth_loss(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        loss = functions.mse_loss(predictions[:, 0], targets, mask)
        loss = torch.sqrt(loss) * 128

        return loss

    def calculate_losses(self, batch: BatchType) -> AttributeDict[Tensor]:
        inputs, targets_segmentation, (targets_depth, mask_depth) = batch

        representations = self(inputs)

        representations_quantized, likelihoods = self.entropy_model(representations)
        entropy = functions.calculate_likelihood_entropy(likelihoods)

        _, likelihoods = self.bits_model(torch.detach(representations))
        bits = functions.calculate_likelihood_entropy(likelihoods, normalize=False)

        loss_reconstruction = self.calculate_task_loss(
            representations=representations_quantized,
            task=self.reconstruction_model,
            calculate_loss=functools.partial(self.calculate_reconstruction_loss, targets=inputs),
        )
        loss_segmentation = self.calculate_task_loss(
            representations=representations_quantized,
            task=self.segmentation_model,
            calculate_loss=functools.partial(self.calculate_segmentation_loss, targets=targets_segmentation),
        )
        loss_depth = self.calculate_task_loss(
            representations=representations_quantized,
            task=self.depth_model,
            calculate_loss=functools.partial(self.calculate_depth_loss, targets=targets_depth, mask=mask_depth),
        )

        loss_tasks = loss_reconstruction + loss_segmentation + loss_depth

        metrics = AttributeDict({
            'Loss Tasks': loss_tasks,
            'Loss Reconstruction': loss_reconstruction,
            'Loss Segmentation': loss_segmentation,
            'Loss Depth': loss_depth,
            'Entropy': entropy,
            'Bits': bits,
            'Segmentation Accuracy': self.accuracy_segmentation,
            'Segmentation IoU': self.iou_segmentation,
        })

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate, betas=(self.momentum, 0.999))

    def training_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        optimization.adaptive_clipping(
            optimizer=self.optimizers(),
            losses=(
                LossArguments(metrics.loss_tasks, metrics.entropy, self.alpha),
                LossArguments(metrics.bits),
            ),
        )

    def validation_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def calculate_bit_rate(self, batch: BatchType) -> float:
        inputs, _, _ = batch

        representations = self(inputs)

        bit_rate = sum(
            len(string) * 8
            for string
            in self.bits_model.compress(representations)
        ) / inputs.shape[0]

        return bit_rate

    def test_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)

        self.log('Test Bit Rate', self.calculate_bit_rate(batch))

    def on_validation_start(self) -> None:
        self.accuracy_segmentation.reset()
        self.iou_segmentation.reset()


class EntropyBottleneckBaseline(LightningModule):
    def __init__(
            self,
            transformer: Module,
            reconstructor: Module,
            segmentation_classifier: Module,
            depth_classifier: Module,
            entropy_model: Module,
            bits_reconstruction: AutoregressiveEntropyModel,
            bits_segmentation: AutoregressiveEntropyModel,
            bits_depth: AutoregressiveEntropyModel,
            split_size: int,
            reconstruction_size: int,
            num_classes: int,
            ignore_index: int,
            alpha: float,
            learning_rate: float,
            momentum: float,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.transformer = transformer
        self.reconstructor = reconstructor
        self.segmentation_classifier = segmentation_classifier
        self.depth_classifier = depth_classifier
        self.entropy_model = entropy_model
        self.bits_reconstruction = bits_reconstruction
        self.bits_segmentation = bits_segmentation
        self.bits_depth = bits_depth

        self.split_size = split_size
        self.reconstruction_size = reconstruction_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.segmentation_accuracy = Accuracy(num_classes=num_classes + 1, ignore_index=ignore_index)
        self.segmentation_iou = JaccardIndex(num_classes + 1, ignore_index=ignore_index)

    def split_channels(self, representations: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        depth_start = self.reconstruction_size + self.split_size

        reconstruction = representations[:, :self.reconstruction_size]
        segmentation = representations[:, self.reconstruction_size:depth_start]
        depth = representations[:, depth_start:]

        return reconstruction, segmentation, depth

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        return self.transformer(inputs)

    def calculate_predictions(self, representations: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        splits = self.split_channels(representations)
        representations_reconstruction, representations_segmentation, representations_depth = splits

        predictions_reconstruction = self.reconstructor(representations_reconstruction)
        predictions_segmentation = self.segmentation_classifier(representations_segmentation)
        predictions_depth = self.depth_classifier(representations_depth)[:, 0]

        return predictions_reconstruction, predictions_segmentation, predictions_depth

    def calculate_entropies(self, likelihoods: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        entropy = functions.calculate_likelihood_entropy(likelihoods, reduce=False)

        reconstruction, segmentation, depth = self.split_channels(entropy[None, :])
        reconstruction = torch.sum(reconstruction)
        segmentation = torch.sum(segmentation)
        depth = torch.sum(depth)

        return reconstruction, segmentation, depth

    def calculate_bits(self, representations: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        reconstruction, segmentation, depth = self.split_channels(representations)

        _, reconstruction = self.bits_reconstruction(reconstruction)
        reconstruction = functions.calculate_likelihood_entropy(reconstruction, normalize=False)

        _, segmentation = self.bits_segmentation(segmentation)
        segmentation = functions.calculate_likelihood_entropy(segmentation, normalize=False)

        _, depth = self.bits_depth(depth)
        depth = functions.calculate_likelihood_entropy(depth, normalize=False)

        return reconstruction, segmentation, depth

    def calculate_losses(self, batch: BatchType) -> AttributeDict[Tensor]:
        inputs, targets_segmentation, (targets_depth, mask_depth) = batch

        representations = self(inputs)
        representations_quantized, likelihoods = self.entropy_model(representations)

        predictions = self.calculate_predictions(representations_quantized)
        predictions_reconstruction, predictions_segmentation, predictions_depth = predictions

        loss_reconstruction = functional.mse_loss(predictions_reconstruction, inputs)
        loss_reconstruction = torch.sqrt(loss_reconstruction) * 255
        loss_segmentation = functional.cross_entropy(
            input=predictions_segmentation,
            target=targets_segmentation,
            ignore_index=self.ignore_index,
        )
        loss_segmentation *= self.num_classes
        loss_depth = functions.mse_loss(predictions_depth, targets_depth, mask_depth)
        loss_depth = torch.sqrt(loss_depth) * 128

        loss_tasks = loss_reconstruction + loss_segmentation + loss_depth

        entropy_reconstruction, entropy_segmentation, entropy_depth = self.calculate_entropies(likelihoods)
        bits_reconstruction, bits_segmentation, bits_depth = self.calculate_bits(torch.detach(representations))

        self.segmentation_accuracy(functions.add_ninf_channel(predictions_segmentation), targets_segmentation)
        self.segmentation_iou(functions.add_ninf_channel(predictions_segmentation), targets_segmentation)

        metrics = AttributeDict({
            'Loss Tasks': loss_tasks,
            'Loss Reconstruction': loss_reconstruction,
            'Loss Segmentation': loss_segmentation,
            'Loss Depth': loss_depth,
            'Entropy Reconstruction': entropy_reconstruction,
            'Entropy Segmentation': entropy_segmentation,
            'Entropy Depth': entropy_depth,
            'Bits Reconstruction': bits_reconstruction,
            'Bits Segmentation': bits_segmentation,
            'Bits Depth': bits_depth,
            'Segmentation Accuracy': self.segmentation_accuracy,
            'Segmentation IoU': self.segmentation_iou,
        })

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate, betas=(self.momentum, 0.999))

    def training_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Train {name}', value)

        losses = (
            LossArguments(metrics.loss_reconstruction, metrics.entropy_reconstruction, self.alpha),
            LossArguments(metrics.loss_segmentation, metrics.entropy_segmentation, self.alpha),
            LossArguments(metrics.loss_depth, metrics.entropy_depth, self.alpha),
            LossArguments(metrics.bits_reconstruction + metrics.bits_segmentation + metrics.bits_depth),
        )

        optimization.adaptive_clipping(self.optimizers(), losses)

    def calculate_bit_rate(self, batch: BatchType) -> Tuple[float, float, float]:
        inputs, _, _ = batch

        representations = self.split_channels(self.transformer(inputs))
        representations_reconstruction, representations_segmentation, representations_depth = representations

        bit_rate_reconstruction = sum(
            len(string) * 8
            for string
            in self.bits_reconstruction.compress(representations_reconstruction)
        ) / inputs.shape[0]

        bit_rate_segmentation = sum(
            len(string) * 8
            for string
            in self.bits_segmentation.compress(representations_segmentation)
        ) / inputs.shape[0]

        bit_rate_depth = sum(
            len(string) * 8
            for string
            in self.bits_depth.compress(representations_depth)
        ) / inputs.shape[0]

        return bit_rate_reconstruction, bit_rate_segmentation, bit_rate_depth

    def validation_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Validation {name}', value)

    def test_step(self, batch: BatchType, *args) -> None:  # type: ignore
        metrics = self.calculate_losses(batch)

        for name, value in metrics.items():
            self.log(f'Test {name}', value)

        bit_rate_reconstruction, bit_rate_segmentation, bit_rate_depth = self.calculate_bit_rate(batch)
        self.log('Test Bit Rate Reconstruction', bit_rate_reconstruction)
        self.log('Test Bit Rate Segmentation', bit_rate_segmentation)
        self.log('Test Bit Rate Depth', bit_rate_depth)

    def on_validation_start(self) -> None:
        self.segmentation_accuracy.reset()
        self.segmentation_iou.reset()
