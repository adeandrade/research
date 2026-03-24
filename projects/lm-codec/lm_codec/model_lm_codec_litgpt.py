import itertools

import torch
from lightning.fabric.utilities import load
from pytorch_lightning import LightningModule
from sfu_torch_lib.utils import AttributeDict
from torch import Tensor
from torch.nn import Parameter
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR

from litgpt import utils
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer
from lm_codec import functions
from lm_codec.model_codec import PriorAnalysis
from lm_codec.model_entropy import DistributionFreeModel, GaussianModel
from lm_codec.model_lm import LinearCosineCoefficient
from lm_codec.model_lm_codec import PriorSynthesis


class LMCodecAdHoc(LightningModule):
    def __init__(
        self,
        model_type_lm: str,
        split_index: int,
        alpha: float,
        max_steps: int,
        n_embd_out: int = 24,
        **_,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model_type_lm = model_type_lm
        self.alpha = alpha
        self.max_steps = max_steps
        self.split_index = split_index
        self.n_embd_out = n_embd_out

        self.lm, self.tokenizer = self.load_lm(model_type_lm)

        self.model_prior = DistributionFreeModel()
        self.model_representation = GaussianModel()

        self.num_parameters = self.model_representation.num_parameters

        self.analysis_prior = PriorAnalysis(
            self.lm.config.n_embd,
            n_embd_out,
            self.lm.config.n_head,
        )
        self.synthesis_prior = PriorSynthesis(
            n_embd_out,
            self.lm.config.n_embd,
            self.num_parameters,
            self.lm.config.n_head,
        )

        parameters_prior = self.model_prior.initial_prior(n_embd_out)
        self.parameters_prior = Parameter(parameters_prior)

    @classmethod
    def load_lm(cls, model_name: str) -> tuple[GPT, Tokenizer]:
        checkpoint_dir = utils.auto_download_checkpoint(model_name)

        config = Config.from_file(checkpoint_dir / 'model_config.yaml')

        model = GPT(config)

        state_dict = load._lazy_load(checkpoint_dir / 'lit_model.pth')
        state_dict = state_dict.get('model', state_dict)
        model.load_state_dict(state_dict)

        tokenizer = Tokenizer(checkpoint_dir)

        return model, tokenizer

    @torch.compile
    def forward(self, inputs: Tensor, return_blocks: set[int] | None = None, quantize: bool = True):
        if return_blocks is None:
            return_blocks = set()
        return self.lm(inputs, return_blocks=return_blocks, quantize=quantize)

    @torch.compile
    def calculate_metrics(self, batch: tuple[Tensor, Tensor]) -> AttributeDict[Tensor]:
        inputs, targets = batch
        _, block_size = inputs.shape

        logits, (representation, *_) = self.forward(inputs, {self.split_index}, True)

        distortion = utils.chunked_cross_entropy(logits, targets)

        hyper_prior = self.analysis_prior(representation)
        hyper_prior = functions.quantize(hyper_prior)

        likelihoods = self.model_prior.nll_discrete(hyper_prior, self.parameters_prior[None, None])
        bpt_prior = functions.calculate_bpe(likelihoods, block_size)

        parameters_representation = self.synthesis_prior(hyper_prior)

        likelihoods = self.model_representation.nll_discrete(representation, parameters_representation)
        bpt_representation = functions.calculate_bpe(likelihoods, block_size)

        bpt = bpt_prior + bpt_representation

        loss = distortion + self.alpha * bpt

        return AttributeDict({
            'Loss': loss,
            'Distortion': distortion,
            'BPT': bpt,
            'BPT Hyper Prior': bpt_prior,
            'BPT Representation': bpt_representation,
        })

    def configure_optimizers(self, betas: tuple[float, float] = (0.9, 0.95), weight_decay: float = 1e-1):
        groups = [
            {
                'params': itertools.chain(
                    self.analysis_prior.parameters(),
                    self.synthesis_prior.parameters(),
                    self.parameters(recurse=False),
                    (p for p in self.lm.parameters() if p.requires_grad and p.dim() < 2),
                ),
                'weight_decay': 0.0,
            },
            {
                'params': (p for p in self.lm.parameters() if p.requires_grad and p.dim() >= 2),
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
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Training {name}': value for name, value in metrics.items()})

        return metrics.loss

    def validation_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Validation {name}': value for name, value in metrics.items()})

    def test_step(self, *args, **_) -> None:
        metrics = self.calculate_metrics(args[0])
        self.log_dict({f'Testing {name}': value for name, value in metrics.items()})
