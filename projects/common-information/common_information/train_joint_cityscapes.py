import os

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

import common_information.model_base as model_base
import common_information.processing_cityscapes as processing
from common_information.dataset_cityscapes import Cityscapes


@slack.notify
@mlflow.install
def train(
    *,
    model_type: str,
    run_id: str | None = None,
    run_id_pretrained: str | None = None,
    dataset_path: str = 's3://datasets/cityscapes.zip',
    num_channels: int = 192,
    alpha: float = 0.001,
    beta: float = 1.0,
    learning_rate: float = 0.0001,
    gradient_clip_value: float | None = 1.0,
    batch_size: int = 4,
    accumulate_grad_batches: int = 3,
    patience: int = 50,
    max_num_epochs: int = 1000,
    monitor_label: str = 'Rate Distortion',
) -> None:
    """
    Trains a model.

    :param model_type: Model class name.
    :param run_id: Experiment ID of an existing run to continue.
    :param run_id_pretrained: Experiment ID of a pretrained model to start with.
    :param run_id_encoder: Experiment ID of a reconstruction model.
    :param dataset_path: Path to dataset.
    :param alpha: The rate-distortion Lagrangian.
    :param learning_rate: Momentum of the Adam optimizer.
    :param gradient_clip_value: Gradient clipping value.
    :param batch_size: Batch size.
    :param patience: Patience of the early-stopping algorithm.
    :param max_num_epochs: Maximum number of epochs.
    :param monitor_label: Monitor label.
    """
    step_size = batch_size * accumulate_grad_batches

    dataset_path = io.localize_dataset(dataset_path)

    transform_train = processing.create_joint_train_transformer(means=None, scales=None)
    dataset_train = Cityscapes(dataset_path, transform_train, split='train')

    transform_validation = processing.create_joint_test_transformer(means=None, scales=None)
    dataset_validation = Cityscapes(dataset_path, transform_validation, split='validation')

    checkpoint_path, restart = state.get_resumable_checkpoint_path(run_id, run_id_pretrained)

    model_class = utils.get_class(model_type, model_base)

    model = model_class(
        input_num_channels=Cityscapes.image_channels,
        num_classes=Cityscapes.num_classes,
        ignore_index=Cityscapes.ignore_index,
        latent_num_channels=num_channels,
        alpha=alpha,
        beta=beta,
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
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger,
        enable_progress_bar=False,
        log_every_n_steps=sampling.get_num_steps(len(dataset_train), step_size),
        accelerator='auto',
        gradient_clip_val=gradient_clip_value,
        callbacks=[
            EarlyStopping(
                monitor=f'Validation {monitor_label}',
                mode='min',
                patience=patience,
            ),
            MLFlowModelCheckpoint(
                monitor=f'Validation {monitor_label}',
                mode='min',
                patience=10,
            ),
        ],
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    defopt.run(train)


if __name__ == '__main__':
    main()
