import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import Dropout, Embedding, LayerNorm, Linear, Module, ModuleDict, Sequential, functional, init
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2LMHeadModel

from lm_codec import functions
from lm_codec.model_layers import Block


class LinearCosineCoefficient:
    def __init__(
        self,
        max_steps: int,
        learning_rate_max: float = 6e-4,
        learning_rate_min: float = 6e-5,
        warmup_steps: int = 2000,
    ) -> None:
        self.max_steps = max_steps
        self.learning_rate_max = learning_rate_max
        self.learning_rate_min = learning_rate_min
        self.warmup_steps = warmup_steps

    def get_coefficient(self, t: int) -> float:
        if t < self.warmup_steps:
            return t / self.warmup_steps

        if t > self.max_steps:
            return self.learning_rate_min / self.learning_rate_max

        decay_ratio = (t - self.warmup_steps) / (self.max_steps - self.warmup_steps)

        assert 0 <= decay_ratio <= 1

        coefficient = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return self.learning_rate_min / self.learning_rate_max * (1 - coefficient) + coefficient


class GPT(Module):
    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = False,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    ):
        super().__init__()

        self.block_size = block_size
        self.n_layer = n_layer
        self.n_embd = n_embd

        self.transformer = ModuleDict({
            'wte': Embedding(vocab_size, n_embd),
            'wpe': Embedding(block_size, n_embd),
            'drop': Dropout(dropout),
            'h': Sequential(*[
                Block(
                    n_head,
                    n_embd,
                    bias,
                    dropout,
                )
                for _ in range(n_layer)
            ]),
            'ln_f': LayerNorm(n_embd, bias=bias),
        })

        self.lm_head = Linear(n_embd, vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type: str):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        # n_layer, n_head and n_embd are determined from model_type
        config_args: dict[str, Any] = {
            'gpt2': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},  # 124M params
            'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},  # 350M params
            'gpt2-large': {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},  # 774M params
            'gpt2-xl': {'n_layer': 48, 'n_head': 25, 'n_embd': 1600},  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        model = GPT(**config_args)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def embed(self, indices: Tensor) -> Tensor:
        # token embeddings of shape (b, t, n_embd)
        return self.transformer.wte(indices)  # type: ignore

    def predict(
        self,
        tok_emb: Tensor,
        targets: Tensor | None = None,
        return_blocks: set[int] | None = None,
        quantize: bool = True,
    ) -> tuple[Tensor, Tensor | None, list[Tensor]]:
        if return_blocks is None:
            return_blocks = set()
        _, block_size, _ = tok_emb.shape
        device = tok_emb.device

        assert block_size <= self.block_size, f'Sequence of length {block_size}, block size is only {self.block_size}'

        for return_block in return_blocks:
            assert 0 <= return_block < self.n_layer

        representations = []
        pos = torch.arange(0, block_size, dtype=torch.long, device=device)

        # forward the GPT model itself
        # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # type: ignore
        x = self.transformer.drop(tok_emb + pos_emb)  # type: ignore

        for index, block in enumerate(self.transformer.h):  # type: ignore
            if index in return_blocks:
                x = functions.quantize(x) if quantize else x
                representations.append(x)

            x = block(x)

        x = self.transformer.ln_f(x)  # type: ignore

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, representations

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        return_blocks: set[int] | None = None,
        quantize: bool = True,
    ) -> tuple[Tensor, Tensor | None, list[Tensor]]:
        if return_blocks is None:
            return_blocks = set()

        return self.predict(self.embed(idx), targets, return_blocks, quantize)

    @torch.no_grad
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        return_blocks: set[int] | None = None,
        quantize: bool = True,
    ) -> Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if return_blocks is None:
            return_blocks = set()
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]

            # forward the model to get the logits for the index in the sequence
            logits, *_ = self.forward(idx_cond, return_blocks=return_blocks, quantize=quantize)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # apply softmax to convert logits to (normalized) probabilities
            probs = functional.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, -max_new_tokens:]

    def configure_parameter_groups(self, weight_decay: float = 1e-1):
        parameters = [p for p in self.parameters() if p.requires_grad]

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in parameters if p.dim() >= 2]
        nodecay_params = [p for p in parameters if p.dim() < 2]

        return [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

    def configure_optimizers(
        self,
        max_steps: int,
        betas: tuple[float, float] = (0.9, 0.95),
        fused: bool = True,
    ):
        groups = self.configure_parameter_groups()

        coefficient = LinearCosineCoefficient(max_steps)

        optimizer = AdamW(groups, coefficient.learning_rate_max, betas, fused=fused)

        scheduler = {
            'scheduler': LambdaLR(optimizer, coefficient.get_coefficient),
            'interval': 'step',
            'frequency': 1,
        }

        return optimizer, scheduler
