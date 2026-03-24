import os

import defopt
import torch
from pytorch_lightning import Trainer
from sfu_torch_lib import io, mlflow, sampling, slack, state, utils
from sfu_torch_lib.mlflow import MLFlowLogger, MLFlowModelCheckpoint
from torch.utils.data import DataLoader, RandomSampler

from image_codec import processing
from image_codec.dataset_imagenet import Imagenet


@slack.notify
@mlflow.install
def train(
    *,
    model_type: str,
    dataset_type: str = 'imagenet',
    run_id: str | None = None,
    run_id_pretrained: str | None = None,
    split_index: int | None = None,
    alpha: float = 0.0001,
    gradient_clip_value: float | None = 1.0,
    batch_size: int = 16,
    accumulate_grad_batches: int = 40,
    patience: int = 5,
    max_steps: int = 600000,
    epoch_size: int = 100000,
    monitor_label: str = 'Loss',
    use_precision: bool = False,
    dataset_path_imagenet: str = 's3://datasets/imagenet.zip',
) -> None:
    """
    Trains an image classification model and its codec.

    :param model_type: model class name (i.e.: image_codec.model.ResNetAdHoc)
    :param dataset_type: dataset name
    :param run_id: experiment ID of an existing run to continue
    :param run_id_pretrained: experiment ID of a pretrained model to start with
    :param split_index: split point index
    :param alpha: lagrange multiplier for the information bottleneck
    :param gradient_clip_value: gradient clipping value
    :param batch_size: batch size
    :param accumulate_grad_batches: number of batches per step
    :param patience: patience for checkpointing
    :param max_steps: maximum number of steps
    :param epoch_size: number of training samples per epoch
    :param monitor_label: metric to monitor for checkpointing
    :param use_precision: use bfloat16 precision
    :param dataset_path_imagenet: path to ImageNet dataset
    """
    step_size = batch_size * accumulate_grad_batches

    if dataset_type == 'imagenet':
        dataset_path = io.localize_dataset(dataset_path_imagenet)

        transform_train = processing.create_detection_train_transform()
        dataset_train = Imagenet(dataset_path, transform_train, split='train')

        transform_test = processing.create_detection_test_transform()
        dataset_validation = Imagenet(dataset_path, transform_test, split='validation')

    else:
        raise ValueError(f'Dataset {dataset_type} not supported.')

    model_class = utils.get_class(model_type)

    model = model_class(
        num_channels_input=dataset_train.image_channels,
        split_index=split_index,
        alpha=alpha,
        max_steps=max_steps,
        num_classes=dataset_train.num_classes,
    )

    logger = MLFlowLogger(
        tags={
            'model-type': model_type,
            'dataset': dataset_train.__class__.__name__,
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
        logger=logger,
        enable_progress_bar=False,
        log_every_n_steps=sampling.get_num_steps(epoch_size, step_size),
        accelerator='auto',
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_value,
        detect_anomaly=False,
        precision=precision if use_precision else None,
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
            sampler=RandomSampler(dataset_train, num_samples=epoch_size),
            collate_fn=processing.collate_mixup(0.2, 1.0),
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        ),
        val_dataloaders=DataLoader(
            dataset=dataset_validation,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        ),
        ckpt_path=checkpoint_path if not restart else None,
    )


def main():
    defopt.run(train, no_negated_flags=True)


if __name__ == '__main__':
    main()
