import itertools

import torch
import torchvision.models.resnet as resnet_models
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Size, Tensor
from torch.nn import Conv2d, Embedding, Linear, Parameter, functional
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import vision_transformer
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models.vision_transformer import ViT_B_16_Weights

import lm_codec.functions as functions_lm
from image_codec import functions
from lm_codec.model_entropy import DistributionFreeModel, GaussianModel
from lm_codec.model_lm import LinearCosineCoefficient
from lm_codec.model_lm_codec import PriorAnalysis, PriorSynthesis


class ResNetAdHoc(LightningModule):
    def __init__(
        self,
        split_index: int,
        alpha: float,
        max_steps: int,
        num_classes: int,
        num_patches: int = 49,
        n_embd: int = 768,
        n_embd_out: int = 48,
        n_head: int = 12,
        label_smoothing: float = 0.0,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.alpha = alpha
        self.max_steps = max_steps
        self.split_index = split_index
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.n_embd = n_embd
        self.n_embd_out = n_embd_out
        self.n_head = n_head
        self.label_smoothing = label_smoothing

        match split_index:
            case 0:
                self.num_channels = 64
                self.patch_size = 8

            case 1:
                self.num_channels = 64
                self.patch_size = 8

            case 2:
                self.num_channels = 128
                self.patch_size = 4

            case 3:
                self.num_channels = 256
                self.patch_size = 2

            case 4:
                self.num_channels = 512
                self.patch_size = 1

        self.resnet = resnet_models.resnet34(ResNet34_Weights.IMAGENET1K_V1)

        self.model_prior = DistributionFreeModel()
        self.model_representation = GaussianModel()

        self.num_parameters = self.model_representation.num_parameters

        self.patcher = Conv2d(self.num_channels, n_embd, self.patch_size, self.patch_size)
        self.unpatcher = Linear(
            self.num_parameters * n_embd,
            self.num_parameters * self.num_channels * self.patch_size**2,
        )
        self.wpe = Embedding(num_patches, n_embd)

        self.analysis_prior = PriorAnalysis(n_embd, n_embd_out, n_head)
        self.synthesis_prior = PriorSynthesis(n_embd_out, n_embd, self.num_parameters, n_head)

        parameters_prior = self.model_prior.initial_prior(n_embd_out)
        self.parameters_prior = Parameter(parameters_prior)

        self.accuracy = MulticlassAccuracy(num_classes)

    @torch.compile
    def forward(
        self,
        x: Tensor,
        targets: Tensor,
        return_blocks: set[int] | None = None,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        if return_blocks is None:
            return_blocks = set()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        representations = []

        layers = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4, lambda x: x]

        for index, layer in enumerate(layers):
            if index in return_blocks:
                x = functions.quantize(x)
                representations.append(x)

            x = layer(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        distortion = functional.cross_entropy(x, targets, label_smoothing=self.label_smoothing)

        return x, distortion, representations

    def patchify(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape

        assert height % self.patch_size == 0 and width % self.patch_size == 0

        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size
        num_patches = num_patches_height * num_patches_width

        positions_height = torch.arange(num_patches_height, dtype=torch.long, device=x.device)
        positions_width = torch.arange(num_patches_width, dtype=torch.long, device=x.device)
        position_embeddings = positions_height[:, None] + num_patches_height * positions_width[None, :]
        position_embeddings = torch.flatten(position_embeddings)
        position_embeddings = self.wpe(position_embeddings)
        position_embeddings = torch.reshape(position_embeddings, shape=(num_patches, self.n_embd))

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.patcher(x)

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(batch_size, self.n_embd, num_patches_height * num_patches_width)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        x = x.permute(0, 2, 1)

        # add positional embeddings
        return x + position_embeddings

    def unpatchify(self, inputs: Tensor, shape: Size) -> Tensor:
        batch_size, num_channels, height, width = shape

        assert height % self.patch_size == 0 and width % self.patch_size == 0

        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size
        num_patches = num_patches_height * num_patches_width

        outputs = torch.reshape(inputs, (batch_size, num_patches, self.n_embd * self.num_parameters))
        outputs = self.unpatcher(outputs)
        outputs = torch.reshape(
            outputs,
            shape=(
                batch_size,
                num_patches_height,
                num_patches_width,
                self.patch_size,
                self.patch_size,
                num_channels,
                self.num_parameters,
            ),
        )
        outputs = torch.einsum('bhwpqcz->bchpwqz', outputs)
        return torch.reshape(outputs, shape=(batch_size, num_channels, height, width, self.num_parameters))

    @torch.compile
    def calculate_metrics(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, AttributeDict[Tensor]]:
        inputs, _ = batch
        _, _, height, width = inputs.shape
        block_size = height * width

        predictions, distortion, (representation, *_) = self.forward(*batch, {self.split_index})

        hyper_prior = self.patchify(representation)
        hyper_prior = self.analysis_prior(hyper_prior)
        hyper_prior = functions_lm.quantize(hyper_prior)

        likelihoods = self.model_prior.nll_discrete(hyper_prior, self.parameters_prior[None, None])
        bpt_prior = functions_lm.calculate_bpe(likelihoods, block_size)

        parameters_representation = self.synthesis_prior(hyper_prior)
        parameters_representation = self.unpatchify(parameters_representation, representation.shape)

        likelihoods = self.model_representation.nll_discrete(representation, parameters_representation)
        bpt_representation = functions_lm.calculate_bpe(likelihoods, block_size)

        bpt = bpt_prior + bpt_representation

        loss = distortion + self.alpha * bpt

        metrics = AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'BPT': bpt,
            'BPT Hyper Prior': bpt_prior,
            'BPT Representation': bpt_representation,
        })

        return predictions, metrics

    def configure_optimizers(self, betas: tuple[float, float] = (0.9, 0.95), weight_decay: float = 1e-4):
        groups = [
            {
                'params': itertools.chain(
                    self.analysis_prior.parameters(),
                    self.synthesis_prior.parameters(),
                    self.patcher.parameters(),
                    self.unpatcher.parameters(),
                    self.wpe.parameters(),
                    self.parameters(recurse=False),
                ),
                'weight_decay': 0.0,
            },
            {
                'params': self.resnet.parameters(),
                'weight_decay': weight_decay,
            },
        ]

        coefficient = LinearCosineCoefficient(self.max_steps)

        optimizer = AdamW(groups, coefficient.learning_rate_max, betas)

        scheduler = {
            'scheduler': LambdaLR(optimizer, coefficient.get_coefficient),
            'interval': 'step',
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, *args, **_) -> Tensor:
        _, metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()})

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        (_, targets), *_ = args

        predictions, metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Validation {name}': value for name, value in metrics.items()})

        self.accuracy(predictions, targets)
        self.log('Validation Accuracy', self.accuracy, on_step=False, on_epoch=True)

    def test_step(self, *args, **_) -> None:
        _, metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Testing {name}': value for name, value in metrics.items()})


class ViTAdHoc(LightningModule):
    def __init__(
        self,
        split_index: int,
        alpha: float,
        max_steps: int,
        n_embd_out: int = 24,
        label_smoothing: float = 0.11,
        **_,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.alpha = alpha
        self.max_steps = max_steps
        self.split_index = split_index
        self.n_embd_out = n_embd_out
        self.label_smoothing = label_smoothing

        self.vit = vision_transformer.vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)

        self.model_prior = DistributionFreeModel()
        self.model_representation = GaussianModel()

        self.num_parameters = self.model_representation.num_parameters

        self.analysis_prior = PriorAnalysis(
            self.vit.hidden_dim,
            n_embd_out,
        )
        self.synthesis_prior = PriorSynthesis(
            n_embd_out,
            self.vit.hidden_dim,
            self.num_parameters,
        )

        parameters_prior = self.model_prior.initial_prior(n_embd_out)
        self.parameters_prior = Parameter(parameters_prior)

        self.accuracy = MulticlassAccuracy(self.vit.num_classes)

    @torch.compile
    def forward(
        self,
        x: Tensor,
        targets: Tensor,
        return_blocks: set[int] | None = None,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        # reshape and permute the input tensor
        if return_blocks is None:
            return_blocks = set()
        x = self.vit._process_input(x)

        n, *_ = x.shape

        # expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        representations = []

        for index, layer in enumerate(self.vit.encoder.layers):
            if index in return_blocks:
                x = functions.quantize(x)
                representations.append(x)

            x = layer(x)

        x = self.vit.encoder.ln(x)
        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.vit.heads(x)

        distortion = functional.cross_entropy(x, targets, label_smoothing=self.label_smoothing)

        return x, distortion, representations

    @torch.compile
    def calculate_metrics(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, AttributeDict[Tensor]]:
        inputs, _ = batch
        _, _, height, width = inputs.shape
        block_size = height * width

        # self.vit.eval()
        predictions, distortion, (representation, *_) = self.forward(*batch, {self.split_index})

        hyper_prior = self.analysis_prior(representation)
        hyper_prior = functions.quantize(hyper_prior)

        likelihoods = self.model_prior.nll_discrete(hyper_prior, self.parameters_prior[None, None])
        bpt_prior = functions_lm.calculate_bpe(likelihoods, block_size)

        parameters_representation = self.synthesis_prior(hyper_prior)

        likelihoods = self.model_representation.nll_discrete(representation, parameters_representation)
        bpt_representation = functions_lm.calculate_bpe(likelihoods, block_size)

        bpt = bpt_prior + bpt_representation

        # loss = bpt_prior + bpt_representation
        loss = distortion + self.alpha * bpt

        metrics = AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'BPT': bpt,
            'BPT Hyper Prior': bpt_prior,
            'BPT Representation': bpt_representation,
        })

        return predictions, metrics

    def configure_optimizers(self, betas: tuple[float, float] = (0.9, 0.95), weight_decay: float = 0.3):
        groups = [
            {
                'params': itertools.chain(
                    self.analysis_prior.parameters(),
                    self.synthesis_prior.parameters(),
                    self.parameters(recurse=False),
                ),
                'weight_decay': 0.0,
            },
            {
                'params': self.vit.parameters(),
                'weight_decay': weight_decay,
            },
        ]

        coefficient = LinearCosineCoefficient(self.max_steps)

        optimizer = AdamW(groups, coefficient.learning_rate_max, betas)

        scheduler = {
            'scheduler': LambdaLR(optimizer, coefficient.get_coefficient),
            'interval': 'step',
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, *args, **_) -> Tensor:
        _, metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()})

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        (_, targets), *_ = args

        predictions, metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Validation {name}': value for name, value in metrics.items()})

        self.accuracy(predictions, targets)
        self.log('Validation Accuracy', self.accuracy, on_step=False, on_epoch=True)

    def test_step(self, *args, **_) -> None:
        _, metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Testing {name}': value for name, value in metrics.items()})
