import os

import defopt
import torch
from pytorch_lightning import Trainer
from sfu_torch_lib import mlflow, slack, state
from sfu_torch_lib.mlflow import MLFlowLogger
from torch.utils.data import DataLoader

from litgpt.tokenizer import Tokenizer
from lm_codec.dataset_openwebtext import OpenWebText, OpenWebTextReTokenize


@slack.notify
@mlflow.install
def test(
    *,
    run_id_pretrained: str,
    run_id: str | None = None,
    dataset_type: str = 'openwebtext',
    batch_size: int = 12,
    block_size: int = 1024,
) -> None:
    """
    Tests a language model and its codec.

    :param dataset_type: dataset name
    :param run_id: experiment ID of an existing run to continue
    :param run_id_pretrained: experiment ID of a pretrained model to start with
    :param batch_size: batch size
    :param block_size: context size
    """
    model = state.load_model(run_id_pretrained)

    if dataset_type == 'openwebtext':
        num_documents = 4000 * batch_size

        if hasattr(model, 'tokenizer'):
            assert isinstance(model.tokenizer, Tokenizer)

            dataset = OpenWebTextReTokenize(model.tokenizer, 'validation', block_size, num_documents)

        else:
            dataset = OpenWebText('validation', block_size, num_documents)

    else:
        raise ValueError(f'Validation dataset {dataset_type} not supported.')

    logger = MLFlowLogger(
        tags={
            'model-type': model.__class__.__name__,
            'dataset': dataset.__class__.__name__,
            'job-id': os.environ.get('JOB_ID'),
        }
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    precision = (
        'bf16-mixed'
        if torch.cuda.is_bf16_supported()
        and torch.cuda.get_device_name()
        not in {
            'NVIDIA GeForce RTX 2080 Ti',
        }
        else '16-mixed'
    )

    trainer = Trainer(
        logger=logger,
        enable_progress_bar=False,
        accelerator='auto',
        detect_anomaly=False,
        precision=precision,
    )

    trainer.test(
        model=model,
        dataloaders=DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
    )


def main():
    defopt.run(test, no_negated_flags=True)


if __name__ == '__main__':
    main()
