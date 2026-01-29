import os

import defopt
import sfu_torch_lib.io as io
import sfu_torch_lib.mlflow as mlflow
import sfu_torch_lib.slack as slack
import sfu_torch_lib.state as state
import sfu_torch_lib.utils as utils
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sfu_torch_lib.mlflow import MLFlowLogger, MLFlowModelCheckpoint
from torch.utils.data import DataLoader

from common_information.dataset_discrete import DiscreteTensors


@slack.notify
@mlflow.install
def train(
    *,
    model_type: str,
    run_id: str | None = None,
    run_id_pretrained: str | None = None,
    dataset_path: str = 's3://datasets/discrete_tensors.pkl.gz',
    num_channels: int = 512,
    alpha: float = 0.0001,
    beta: float = 1.0,
    learning_rate: float = 0.0001,
    gradient_clip_value: float | None = 1.0,
    batch_size: int = 100,
    accumulate_grad_batches: int = 1,
    patience: int = 10,
    steps_per_epoch_training: int = 1000,
    steps_per_epoch_validation: int = 100,
    max_num_epochs: int = 100,
    monitor_label: str = 'Rate Distortion',
) -> None:
    """
    Trains a model.

    :param model_type: Class name of the model to train.
    :param run_id: Experiment ID of an existing run to continue.
    :param run_id_pretrained: Experiment ID of a pretrained model to start with.
    :param dataset_path: Path to dataset.
    :param alpha: Scaling factor for entropy loss.
    :param learning_rate: Learning rate of the optimizer.
    :param gradient_clip_value: Gradient clipping value.
    :param batch_size: Batch size.
    :param patience: Patience of the early-stopping algorithm.
    :param max_num_epochs: Maximum number of epochs.
    :param num_aspect_ratio_groups: Number of groups for the batches.
    """
    dataset_path = io.localize_dataset(dataset_path)

    dataset = DiscreteTensors(dataset_path)

    model_class = utils.get_class(model_type)

    model = model_class(
        input_num_channels=DiscreteTensors.channels,
        latent_num_channels=num_channels,
        alpha=alpha,
        beta=beta,
        learning_rate=learning_rate,
    )

    checkpoint_path, restart = state.get_resumable_checkpoint_path(run_id, run_id_pretrained)

    if checkpoint_path and restart:
        state.load_checkpoint_state(checkpoint_path, model)

    logger = MLFlowLogger(
        tags={
            'model': model_class.__name__,
            'dataset': DiscreteTensors.__name__,
            'job-id': os.environ.get('JOB_ID'),
        }
    )

    trainer = Trainer(
        max_epochs=max_num_epochs,
        logger=logger,
        enable_progress_bar=False,
        limit_train_batches=steps_per_epoch_training * accumulate_grad_batches,
        limit_val_batches=steps_per_epoch_validation * accumulate_grad_batches,
        log_every_n_steps=steps_per_epoch_training * accumulate_grad_batches // 10,
        accelerator='auto',
        gradient_clip_val=gradient_clip_value,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[
            EarlyStopping(
                monitor=f'Validation {monitor_label}',
                mode='min',
                patience=patience,
            ),
            MLFlowModelCheckpoint(
                monitor=f'Validation {monitor_label}',
                mode='min',
                patience=patience,
            ),
        ],
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
        ),
        val_dataloaders=DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
        ),
        ckpt_path=checkpoint_path if not restart else None,
    )


def main():
    defopt.run(train)


if __name__ == '__main__':
    main()
