import os
from typing import Optional

import defopt
import sfu_torch_lib.io as io
import sfu_torch_lib.mlflow as mlflow
import sfu_torch_lib.sampling as sampling
import sfu_torch_lib.slack as slack
import sfu_torch_lib.state as state
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sfu_torch_lib.mlflow import MLFlowLogger, MLFlowModelCheckpoint
from torch.utils.data import DataLoader

import conditional_residual.processing_cityscapes as processing
from conditional_residual.dataset_cityscapes import Cityscapes
from conditional_residual.model_baselines import SegmentationCompressedReconstruction


@slack.notify
@mlflow.install
def train(
        *,
        run_id: Optional[str] = None,
        run_id_pretrained: Optional[str] = None,
        run_id_reconstruction: Optional[str] = None,
        dataset_path: str = 's3://datasets/cityscapes.zip',
        num_channels: int = 32,
        alpha: float = .025,
        coder_learning_rate: float = .0001,
        classifier_learning_rate: float = .01,
        classifier_momentum: float = .9,
        classifier_weight_decay: float = 1e-4,
        batch_size: int = 10,
        patience: int = 100,
        max_num_epochs: int = 600,
) -> None:
    """
    Trains a model.

    :param run_id: Experiment ID of an existing run to continue.
    :param run_id_pretrained: Experiment ID of a pretrained model to start with.
    :param run_id_reconstruction: Experiment ID of a reconstruction model.
    :param dataset_path: Path to dataset.
    :param num_channels: Number of feature channels.
    :param alpha: Scaling factor for entropy loss.
    :param coder_learning_rate: Momentum of the Adam optimizer.
    :param classifier_learning_rate: Learning rate of the SGD optimizer.
    :param classifier_momentum: Momentum of the SGD optimizer.
    :param classifier_weight_decay: Regularization hyperparameter of the SGD optimizer.
    :param batch_size: Batch size.
    :param patience: Patience of the early-stopping algorithm.
    :param max_num_epochs: Maximum number of epochs.
    """
    logger = MLFlowLogger(tags={
        'model': SegmentationCompressedReconstruction.__name__,
        'dataset': Cityscapes.__name__,
        'job-id': os.environ.get('JOB_ID'),
    })

    dataset_path = io.localize_dataset(dataset_path)

    transform_train = processing.create_train_transformer(means=None, scales=None)
    dataset_train = Cityscapes(dataset_path, transform_train, split='train')
    transform_validation = processing.create_test_transformer(means=None, scales=None)
    dataset_validation = Cityscapes(dataset_path, transform_validation, split='validation')

    checkpoint_path, restart = state.get_resumable_checkpoint_path(run_id, run_id_pretrained)

    model = SegmentationCompressedReconstruction(
        num_channels=num_channels,
        image_channels=Cityscapes.image_channels,
        num_classes=Cityscapes.num_classes,
        ignore_index=Cityscapes.ignore_index,
        alpha=alpha,
        coder_learning_rate=coder_learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        classifier_momentum=classifier_momentum,
        classifier_weight_decay=classifier_weight_decay,
        run_id_reconstruction=run_id_reconstruction,
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
                monitor='Validation Segmentation IoU',
                mode='max',
                patience=patience,
            ),
            MLFlowModelCheckpoint(
                monitor='Validation Segmentation IoU',
                mode='max',
                patience=10,
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
            batch_size=batch_size // 2,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
        ckpt_path=checkpoint_path if not restart else None,
    )


if __name__ == '__main__':
    defopt.run(train)
