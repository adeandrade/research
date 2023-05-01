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

import conditional_residual.processing_coco as processing
from conditional_residual.dataset_coco import COCO
from conditional_residual.model_baselines import DetectionCompressedReconstruction


@slack.notify
@mlflow.install
def train(
        *,
        run_id: Optional[str] = None,
        run_id_pretrained: Optional[str] = None,
        dataset_path: str = 's3://datasets/coco-2017.zip',
        num_channels: int = 32,
        alpha: float = .025,
        learning_rate: float = .0001,
        batch_size: int = 6,
        patience: int = 10,
        max_num_epochs: int = 100,
        num_aspect_ratio_groups: int = 3,
) -> None:
    """
    Trains a model.

    :param run_id: Experiment ID of an existing run to continue.
    :param run_id_pretrained: Experiment ID of a pretrained model to start with.
    :param dataset_path: Path to dataset.
    :param num_channels: Number of feature channels.
    :param alpha: Scaling factor for entropy loss.
    :param learning_rate: Momentum of the Adam optimizer.
    :param batch_size: Batch size.
    :param patience: Patience of the early-stopping algorithm.
    :param max_num_epochs: Maximum number of epochs.
    :param num_aspect_ratio_groups: Number of groups for the batches.
    """
    logger = MLFlowLogger(tags={
        'model': DetectionCompressedReconstruction.__name__,
        'dataset': COCO.__name__,
        'job-id': os.environ.get('JOB_ID'),
    })

    dataset_path = io.localize_dataset(dataset_path)

    transform_train = processing.create_train_transformer()
    dataset_train = COCO(dataset_path, transform_train, split='train')
    transform_validation = processing.create_test_transformer()
    dataset_validation = COCO(dataset_path, transform_validation, split='validation')

    train_sampler = RandomSampler(dataset_train)
    group_ids = group_sampler.create_aspect_ratio_groups(dataset_train, num_aspect_ratio_groups)
    train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    checkpoint_path, restart = state.get_resumable_checkpoint_path(run_id, run_id_pretrained)

    model = DetectionCompressedReconstruction(
        num_channels=num_channels,
        image_channels=COCO.image_channels,
        num_classes=COCO.num_classes_detection,
        alpha=alpha,
        learning_rate=learning_rate,
    )

    if checkpoint_path and restart:
        state.load_checkpoint_state(checkpoint_path, model)

    trainer = Trainer(
        max_epochs=max_num_epochs,
        logger=logger,
        enable_progress_bar=False,
        log_every_n_steps=sampling.get_num_steps(len(dataset_train), batch_size),
        accelerator='auto',
        callbacks=[
            EarlyStopping(
                monitor='Validation Mean Average Precision',
                mode='max',
                patience=patience,
            ),
            MLFlowModelCheckpoint(
                monitor='Validation Mean Average Precision',
                mode='max',
            ),
        ],
    )

    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(
            dataset=dataset_train,
            batch_sampler=train_batch_sampler,
            collate_fn=processing.collate,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        val_dataloaders=DataLoader(
            dataset=dataset_validation,
            batch_size=1,
            collate_fn=processing.collate,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        ckpt_path=checkpoint_path if not restart else None,
    )


if __name__ == '__main__':
    defopt.run(train)
