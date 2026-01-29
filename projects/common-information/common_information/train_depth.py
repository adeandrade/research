import os
from typing import Optional

import defopt
import sfu_torch_lib.io as io
import sfu_torch_lib.mlflow as mlflow
import sfu_torch_lib.sampling as sampling
import sfu_torch_lib.slack as slack
import sfu_torch_lib.state as state
import sfu_torch_lib.utils as utils
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sfu_torch_lib.mlflow import MLFlowLogger, MLFlowModelCheckpoint
from torch.utils.data import DataLoader

import common_information.processing_cityscapes as processing
from common_information.dataset_cityscapes import Cityscapes


@slack.notify
@mlflow.install
def train(
    *,
    model_type: str,
    run_id: Optional[str] = None,
    run_id_pretrained: Optional[str] = None,
    dataset_path: str = 's3://datasets/cityscapes.zip',
    downsample_factor: int = 2,
    learning_rate: float = 0.0001,
    batch_size: int = 10,
    patience: int = 200,
    max_num_epochs: int = 1000,
) -> None:
    """
    Trains a model.

    :param model_type: Model class name.
    :param run_id: Experiment ID of an existing run to continue.
    :param run_id_pretrained: Experiment ID of a pretrained model to start with.
    :param run_id_encoder: Experiment ID of a reconstruction model.
    :param dataset_path: Path to dataset.
    :param downsample_factor: Downsampling factor for the input.
    :param alpha: The rate-distortion Lagrangian.
    :param learning_rate: Learning rate of the Adam optimizer.
    :param batch_size: Batch size.
    :param patience: Patience of the early-stopping algorithm.
    :param max_num_epochs: Maximum number of epochs.
    """
    dataset_path = io.localize_dataset(dataset_path)

    transform_train = processing.create_depth_train_transformer(means=None, scales=None)
    dataset_train = Cityscapes(dataset_path, transform_train, split='train')

    transform_validation = processing.create_depth_test_transformer(means=None, scales=None)
    dataset_validation = Cityscapes(dataset_path, transform_validation, split='validation')

    checkpoint_path, restart = state.get_resumable_checkpoint_path(run_id, run_id_pretrained)

    model_class = utils.get_class(model_type)

    model = model_class(
        downsample_factor=downsample_factor,
        input_num_channels=Cityscapes.image_channels,
        learning_rate=learning_rate,
    )

    logger = MLFlowLogger(
        tags={
            'model': model_class.__name__,
            'dataset': Cityscapes.__name__,
            'job-id': os.environ.get('JOB_ID'),
        }
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
                monitor='Validation Loss',
                mode='min',
                patience=patience,
            ),
            MLFlowModelCheckpoint(
                monitor='Validation Loss',
                mode='min',
                patience=10,
            ),
        ],
    )

    torch.set_float32_matmul_precision('medium')

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
            batch_size=batch_size // 2,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        ckpt_path=checkpoint_path if not restart else None,
    )


def main():
    defopt.run(train)


if __name__ == '__main__':
    main()
