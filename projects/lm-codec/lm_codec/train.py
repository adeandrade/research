import os

import defopt
import torch
from pytorch_lightning import Trainer
from sfu_torch_lib import mlflow, sampling, slack, state, utils
from sfu_torch_lib.mlflow import MLFlowLogger, MLFlowModelCheckpoint
from torch.utils.data import DataLoader

from lm_codec.dataset_openwebtext import OpenWebText, OpenWebTextReTokenize


@slack.notify
@mlflow.install
def train(
    *,
    model_type: str,
    dataset_type: str = 'openwebtext',
    dataset_type_validation: str = 'openwebtext',
    run_id: str | None = None,
    run_id_pretrained: str | None = None,
    model_type_lm: str | None = None,
    split_index: int | None = None,
    quantize: bool | None = None,
    alpha: float = 0.0001,
    gradient_clip_value: float | None = 1.0,
    batch_size: int = 12,
    accumulate_grad_batches: int = 40,
    block_size: int = 1024,
    patience: int = 4,
    max_steps: int = 50000,
    monitor_label: str = 'Loss',
) -> None:
    """
    Trains a language model and its codec.

    :param model_type: model class name (i.e.: lm_codec.model_lm_codec.LMCodec)
    :param dataset_type: dataset name
    :param dataset_type_validation: dataset type for validation
    :param run_id: experiment ID of an existing run to continue
    :param run_id_pretrained: experiment ID of a pretrained model to start with
    :param model_type_lm: type of language model
    :param split_index: split point index
    :param quantize: whether to quantize the split point
    :param alpha: lagrange multiplier for the information bottleneck
    :param gradient_clip_value: gradient clipping value
    :param batch_size: batch size
    :param accumulate_grad_batches: number of batches per step
    :param block_size: context size
    :param patience: patience for checkpointing
    :param max_steps: maximum number of steps
    :param monitor_label: metric to monitor for checkpointing
    """
    step_size = batch_size * accumulate_grad_batches

    model_class = utils.get_class(model_type)

    model = model_class(
        split_index=split_index,
        quantize=quantize,
        model_type_lm=model_type_lm,
        alpha=alpha,
        max_steps=max_steps,
    )

    if dataset_type == 'openwebtext':
        num_documents = 1000 * step_size

        if hasattr(model, 'tokenizer'):
            dataset_train = OpenWebTextReTokenize(model.tokenizer, 'train', block_size, num_documents)

        else:
            dataset_train = OpenWebText('train', block_size, num_documents)

    else:
        raise ValueError(f'Dataset {dataset_type} not supported.')

    if dataset_type_validation == 'openwebtext':
        num_documents = 100 * step_size

        if hasattr(model, 'tokenizer'):
            dataset_validation = OpenWebTextReTokenize(model.tokenizer, 'validation', block_size, num_documents)

        else:
            dataset_validation = OpenWebText('validation', block_size, num_documents)

    else:
        raise ValueError(f'Validation dataset {dataset_type_validation} not supported.')

    logger = MLFlowLogger(
        tags={
            'model-type': model_type,
            'dataset-train': dataset_train.__class__.__name__,
            'dataset-validation': dataset_validation.__class__.__name__,
            'job-id': os.environ.get('JOB_ID'),
        }
    )

    checkpoint_path, restart = state.get_resumable_checkpoint_path(run_id, run_id_pretrained)

    if checkpoint_path and restart:
        state.load_checkpoint_state(checkpoint_path, model)

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
        max_steps=max_steps,
        log_every_n_steps=sampling.get_num_steps(len(dataset_train), step_size),
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger,
        enable_progress_bar=False,
        accelerator='auto',
        gradient_clip_val=gradient_clip_value,
        detect_anomaly=False,
        precision=precision,
        callbacks=[
            MLFlowModelCheckpoint(
                monitor=f'Validation {monitor_label}',
                mode='min',
                patience=patience,
            ),
        ],
    )

    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        val_dataloaders=DataLoader(
            dataset=dataset_validation,
            batch_size=batch_size,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        ckpt_path=checkpoint_path if not restart else None,
    )


def main():
    defopt.run(train, no_negated_flags=True)


if __name__ == '__main__':
    main()
