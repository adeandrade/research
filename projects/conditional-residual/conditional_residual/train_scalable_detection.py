import functools
import os
from typing import Optional

import defopt
import sfu_torch_lib.group_sampler as group_sampler
import sfu_torch_lib.io as io
import sfu_torch_lib.mlflow as mlflow
import sfu_torch_lib.sampling as sampling
import sfu_torch_lib.slack as slack
import sfu_torch_lib.state as state
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sfu_torch_lib.group_sampler import GroupedBatchSampler
from sfu_torch_lib.mlflow import MLFlowLogger, MLFlowModelCheckpoint
from torch.utils.data import DataLoader, RandomSampler

import conditional_residual.model_scalable as model_scalable
import conditional_residual.processing_coco as processing
from conditional_residual.dataset_coco import COCO


@slack.notify
@mlflow.install
def train(
        *,
        model_type: str,
        run_id_base: Optional[str] = None,
        run_id: Optional[str] = None,
        run_id_pretrained: Optional[str] = None,
        dataset_path: str = 's3://datasets/coco-2017.zip',
        run_id_reconstruction: Optional[str] = None,
        num_channels: Optional[int] = 256,
        alpha: Optional[float] = None,
        learning_rate: float = .0001,
        gradient_clip_value: Optional[float] = 1.,
        batch_size: int = 6,
        patience: int = 10,
        max_num_epochs: int = 100,
        num_aspect_ratio_groups: int = 3,
) -> None:
    """
    Trains a model.

    :param model_type: Class name of the model to train.
    :param run_id_base: Experiment ID of a segmentation model.
    :param run_id: Experiment ID of an existing run to continue.
    :param run_id_pretrained: Experiment ID of a pretrained model to start with.
    :param dataset_path: Path to dataset.
    :param run_id_reconstruction: Experiment ID of a reconstruction model.
    :param num_channels: Number of channels for enhancement.
    :param alpha: Scaling factor for entropy loss.
    :param learning_rate: Learning rate of the optimizer.
    :param gradient_clip_value: Gradient clipping value.
    :param batch_size: Batch size.
    :param patience: Patience of the early-stopping algorithm.
    :param max_num_epochs: Maximum number of epochs.
    :param num_aspect_ratio_groups: Number of groups for the batches.
    """
    model_class = getattr(model_scalable, model_type)

    model = model_class(
        image_channels=COCO.image_channels,
        alpha=alpha,
        learning_rate=learning_rate,
        run_id_base=run_id_base,
        num_channels=num_channels,
        run_id_reconstruction=run_id_reconstruction,
    )

    dataset_path = io.localize_dataset(dataset_path)

    transform_train = processing.create_input_train_transformer()
    dataset_train = COCO(dataset_path, transform_train, split='train')
    transform_validation = processing.create_input_test_transformer()
    dataset_validation = COCO(dataset_path, transform_validation, split='validation')

    train_sampler = RandomSampler(dataset_train)
    group_ids = group_sampler.create_aspect_ratio_groups(dataset_train, num_aspect_ratio_groups)
    train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    collate_function = functools.partial(processing.crop, divisor=16)

    logger = MLFlowLogger(tags={
        'model': model_class.__name__,
        'dataset': COCO.__name__,
        'job-id': os.environ.get('JOB_ID'),
    })

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
        callbacks=[
            EarlyStopping(
                monitor='Validation Loss',
                mode='min',
                patience=patience,
            ),
            MLFlowModelCheckpoint(
                monitor='Validation Loss',
                mode='min',
            ),
        ],
    )

    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(
            dataset=dataset_train,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_function,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        val_dataloaders=DataLoader(
            dataset=dataset_validation,
            batch_size=1,
            collate_fn=collate_function,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        ckpt_path=checkpoint_path if not restart else None,
    )


if __name__ == '__main__':
    defopt.run(train)
