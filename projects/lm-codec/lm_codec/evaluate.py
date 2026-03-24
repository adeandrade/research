import string

import defopt
import tiktoken
import torch
from sfu_torch_lib import state

from deepeval.benchmarks import LAMBADA
from deepeval.models.base_model import DeepEvalBaseLLM


class Evaluator(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer, split_indices: set[int] | None = None):
        if split_indices is None:
            split_indices = set()
        self.model = model
        self.tokenizer = tokenizer
        self.split_indices = split_indices

        self.translator = str.maketrans('', '', string.punctuation + '“”')

    def get_model_name(self):
        return self.model.__class__.__name__

    def load_model(self):
        return self.model.cuda().eval()

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        prompt_ids = self.tokenizer.encode_ordinary(prompt)

        inputs = torch.tensor([prompt_ids], dtype=torch.int64, device='cuda')

        generated_ids = model.generate(
            inputs,
            max_new_tokens=10,
            top_k=1,
            return_blocks=self.split_indices,
            quantize=True,
        )

        outputs = generated_ids.tolist()
        outputs = self.tokenizer.decode_batch(outputs)
        outputs = outputs[0]
        outputs = outputs.translate(self.translator)
        outputs = outputs.split()
        return outputs[0] if len(outputs) > 0 else ''

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


def evaluate(*, run_id: str) -> None:
    """
    Evaluate a language model using the LAMBADA test.

    :param run_id: run ID of a language model training run.
    """
    model = state.load_model(run_id, cache=True, overwrite=False)

    split_indices: set[int] = {model.split_index} if hasattr(model, 'split_index') else set()  # type: ignore

    tokenizer = tiktoken.get_encoding('gpt2')
    evaluator = Evaluator(model.lm, tokenizer, split_indices)

    benchmark = LAMBADA(n_shots=0)
    benchmark.evaluate(model=evaluator)


def main():
    defopt.run(evaluate)


if __name__ == '__main__':
    main()
