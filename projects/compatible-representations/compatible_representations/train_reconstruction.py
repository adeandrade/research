import os
from typing import Optional

import defopt
import sfu_torch_lib.group_sampler as group_sampler
import sfu_torch_lib.io as io
import sfu_torch_lib.mlflow as mlflow
import sfu_torch_lib.sampling as sampling
import sfu_torch_lib.slack as slack
import sfu_torch_lib.state as state
import sfu_torch_lib.utils as utils
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sfu_torch_lib.group_sampler import GroupedBatchSampler
from sfu_torch_lib.mlflow import MLFlowLogger, MLFlowModelCheckpoint
from torch.utils.data import DataLoader, RandomSampler

import compatible_representations.processing_cityscapes as processing_cityscapes
import compatible_representations.processing_coco as processing_coco
from compatible_representations.dataset_cityscapes import Cityscapes
from compatible_representations.dataset_coco import COCO


@slack.notify
@mlflow.install
def train(
    *,
    model_type: str,
    dataset_type: str,
    dataset_validation_type: str = 'kodak',
    run_id: Optional[str] = None,
    run_id_pretrained: Optional[str] = None,
    run_id_encoder: Optional[str] = None,
    encoder_type: Optional[str] = None,
    dataset_path_vimeo: str = 's3://datasets/vimeo_septuplet.zip',
    dataset_path_cityscapes: str = 's3://datasets/cityscapes.zip',
    dataset_path_coco: str = 's3://datasets/coco-2017.zip',
    dataset_path_kodak: str = 's3://datasets/kodak.zip',
    alpha: float = 0.01,
    learning_rate: float = 0.0001,
    gradient_clip_value: Optional[float] = 1.0,
    batch_size: int = 10,
    patience: int = 5,
    max_num_epochs: int = 1000,
    num_aspect_ratio_groups: int = 3,
) -> None:
    """
    Trains a model.

    :param model_type: Model class name.
    :param dataset_type: Dataset type.
    :param dataset_validation_type: Dataset type for validation.
    :param run_id: Experiment ID of an existing run to continue.
    :param run_id_pretrained: Experiment ID of a pretrained model to start with.
    :param run_id_encoder: Run ID of an encoder.
    :param encoder_type: Encoder type.
    :param dataset_path_vimeo: Path to Vimeo dataset.
    :param dataset_path_cityscapes: Path to Cityscapes dataset.
    :param dataset_path_kodak: Path to Kodak dataset.
    :param alpha: Lagrange multiplier for the information bottleneck.
    :param learning_rate: Learning rate of the SGD optimizer.
    :param gradient_clip_value: Gradient clipping value.
    :param batch_size: Batch size.
    :param patience: Patience of the early-stopping algorithm.
    :param max_num_epochs: Maximum number of epochs.
    :param num_aspect_ratio_groups: Number of groups for the batches.
    """
    if dataset_type == 'cityscapes':
        dataset_path = io.localize_dataset(dataset_path_cityscapes)
        transform_train = processing_cityscapes.create_input_train_transformer(means=None, scales=None)
        dataset_train = Cityscapes(dataset_path, transform_train, split='train')

        train_batch_sampler = None
        collate_function = None
        shuffle = True

    elif dataset_type == 'coco':
        dataset_path = io.localize_dataset(dataset_path_coco)
        transform_train = processing_coco.create_input_train_transformer()
        dataset_train = COCO(dataset_path, transform_train, split='train')

        train_sampler = RandomSampler(dataset_train)
        group_ids = group_sampler.create_aspect_ratio_groups(dataset_train, num_aspect_ratio_groups)

        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)
        collate_function = processing_coco.identity
        shuffle = None

    else:
        raise ValueError(f'Dataset {dataset_type} not supported.')

    if dataset_validation_type == 'cityscapes':
        dataset_path = io.localize_dataset(dataset_path_cityscapes)
        transform_validation = processing_cityscapes.create_input_test_transformer(means=None, scales=None)
        dataset_validation = Cityscapes(dataset_path, transform_validation, split='validation')

        batch_size_validation = batch_size

    elif dataset_type == 'coco':
        dataset_path = io.localize_dataset(dataset_path_coco)
        transform_train = processing_coco.create_input_test_transformer()
        dataset_validation = COCO(dataset_path, transform_train, split='validation')

        batch_size_validation = 1

    else:
        raise ValueError(f'Validation dataset {dataset_validation_type} not supported.')

    model_class = utils.get_class(model_type)

    model = model_class(
        num_channels_input=dataset_train.image_channels,
        run_id_encoder=run_id_encoder,
        encoder_type=encoder_type,
        alpha=alpha,
        learning_rate=learning_rate,
    )

    logger = MLFlowLogger(
        tags={
            'model': model_class.__name__,
            'dataset': dataset_train.__class__.__name__,
            'job-id': os.environ.get('JOB_ID'),
        }
    )

    checkpoint_path, restart = state.get_resumable_checkpoint_path(run_id, run_id_pretrained)

    if checkpoint_path and restart:
        state.load_checkpoint_state(checkpoint_path, model)

    trainer = Trainer(
        max_epochs=max_num_epochs,
        logger=logger,
        enable_progress_bar=False,
        log_every_n_steps=sampling.get_num_steps(len(dataset_train), batch_size),
        accelerator='auto',
        gradient_clip_val=gradient_clip_value,
        detect_anomaly=False,
        callbacks=[
            EarlyStopping(
                monitor='Validation BPP',
                mode='min',
                patience=patience,
            ),
            MLFlowModelCheckpoint(
                monitor='Validation BPP',
                mode='min',
                patience=5,
            ),
        ],
    )

    torch.set_float32_matmul_precision('medium')

    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(
            dataset=dataset_train,
            batch_sampler=train_batch_sampler,
            batch_size=batch_size if train_batch_sampler is None else 1,
            collate_fn=collate_function,
            shuffle=shuffle,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        val_dataloaders=DataLoader(
            dataset=dataset_validation,
            batch_size=batch_size_validation,
            collate_fn=collate_function,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        ckpt_path=checkpoint_path if not restart else None,
    )


if __name__ == '__main__':
    defopt.run(train)
