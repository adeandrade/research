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
    run_id: Optional[str] = None,
    run_id_pretrained: Optional[str] = None,
    run_id_encoder: Optional[str] = None,
    encoder_type: Optional[str] = None,
    dataset_path_cityscapes: str = 's3://datasets/cityscapes.zip',
    dataset_path_coco: str = 's3://datasets/coco-2017.zip',
    alpha: float = 0.001,
    learning_rate: float = 0.0001,
    gradient_clip_value: Optional[float] = 1.0,
    batch_size: int = 10,
    patience: int = 50,
    max_num_epochs: int = 1000,
    num_aspect_ratio_groups: int = 5,
) -> None:
    """
    Trains a model.

    :param model_type: Model class name.
    :param dataset_type: Dataset type.
    :param run_id: Experiment ID of an existing run to continue.
    :param run_id_pretrained: Experiment ID of a pretrained model to start with.
    :param run_id_encoder: Experiment ID of a reconstruction model.
    :param encoder_type: Encoder type.
    :param alpha: The rate-distortion Lagrangian.
    :param learning_rate: Momentum of the Adam optimizer.
    :param batch_size: Batch size.
    :param patience: Patience of the early-stopping algorithm.
    :param max_num_epochs: Maximum number of epochs.
    :param num_aspect_ratio_groups: Number of groups for the batches.
    """
    if dataset_type == 'cityscapes':
        dataset_path = io.localize_dataset(dataset_path_cityscapes)

        transform_train = processing_cityscapes.create_segmentation_train_transformer(means=None, scales=None)
        dataset_train = Cityscapes(dataset_path, transform_train, split='train')
        transform_validation = processing_cityscapes.create_segmentation_test_transformer(means=None, scales=None)
        dataset_validation = Cityscapes(dataset_path, transform_validation, split='validation')

        train_batch_sampler = None
        collate_function = None
        image_channels = Cityscapes.image_channels
        num_classes = Cityscapes.num_classes
        ignore_index = Cityscapes.ignore_index
        shuffle = True

    elif dataset_type == 'coco':
        dataset_path = io.localize_dataset(dataset_path_coco)

        transform_train = processing_coco.create_segmentation_train_transformer()
        dataset_train = COCO(dataset_path, transform_train, split='train')
        transform_validation = processing_coco.create_segmentation_test_transformer()
        dataset_validation = COCO(dataset_path, transform_validation, split='validation')

        train_sampler = RandomSampler(dataset_train)
        group_ids = group_sampler.create_aspect_ratio_groups(dataset_train, num_aspect_ratio_groups)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

        collate_function = processing_coco.pad_segmentation
        image_channels = COCO.image_channels
        num_classes = COCO.num_classes_segmentation
        ignore_index = COCO.ignore_index
        batch_size = 1
        shuffle = None

    else:
        raise ValueError(f'Unknown dataset {dataset_type}')

    checkpoint_path, restart = state.get_resumable_checkpoint_path(run_id, run_id_pretrained)

    model_class = utils.get_class(model_type)

    model = model_class(
        run_id_encoder=run_id_encoder,
        encoder_type=encoder_type,
        num_channels_input=image_channels,
        num_classes=num_classes,
        ignore_index=ignore_index,
        alpha=alpha,
        learning_rate=learning_rate,
    )

    logger = MLFlowLogger(
        tags={
            'model': model_type,
            'dataset': dataset_type,
            'job-id': os.environ.get('JOB_ID'),
        }
    )

    if checkpoint_path and restart:
        state.load_checkpoint_state(checkpoint_path, model)

    trainer = Trainer(
        max_epochs=max_num_epochs,
        logger=logger,
        enable_progress_bar=False,
        gradient_clip_val=gradient_clip_value,
        log_every_n_steps=sampling.get_num_steps(len(dataset_train), batch_size),
        accelerator='auto',
        callbacks=[
            EarlyStopping(
                monitor='Train Segmentation IoU',
                mode='max',
                patience=patience,
            ),
            MLFlowModelCheckpoint(
                monitor='Train Segmentation IoU',
                mode='max',
                patience=patience,
            ),
        ],
    )

    torch.set_float32_matmul_precision('medium')

    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(
            dataset=dataset_train,
            batch_sampler=train_batch_sampler,
            batch_size=batch_size,
            collate_fn=collate_function,
            shuffle=shuffle,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        val_dataloaders=DataLoader(
            dataset=dataset_validation,
            batch_size=batch_size,
            collate_fn=collate_function,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        ckpt_path=checkpoint_path if not restart else None,
    )


if __name__ == '__main__':
    defopt.run(train)
