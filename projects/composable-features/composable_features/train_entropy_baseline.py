from typing import Tuple, Optional

import defopt
import sfu_torch_lib.io as io
import sfu_torch_lib.mlflow as mlflow
import sfu_torch_lib.slack as slack
import sfu_torch_lib.state as state
import sfu_torch_lib.utils as utils
import torch
from compressai.entropy_models import EntropyBottleneck as EntropyBottleneckModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sfu_torch_lib.mlflow import MLFlowLogger, MLFlowModelCheckpoint
from torch.utils.data import DataLoader

import composable_features.model_lst as model_lst
import composable_features.processing as processing
from composable_features.dataset_cityscapes import Cityscapes
from composable_features.model_entropy import AutoregressiveEntropyModel
from composable_features.model_splitter import EntropyBottleneckBaseline, Baseline


@slack.notify
@mlflow.log
def train(
        *,
        run_id: Optional[str] = None,
        run_id_backbone: Optional[str] = None,
        dataset_path: str = 's3://datasets/cityscapes.zip',
        data_size: Tuple[int, int] = (256, 512),
        num_features: int = 128,
        split_size: int = 128,
        reconstruction_size: int = 128,
        batch_size: int = 10,
        alpha: float = 1.,
        learning_rate: float = .0001,
        momentum: float = .937,
        patience: int = 30,
        max_num_epochs: int = 500,
) -> None:
    """
    Trains a model.

    :param run_id: Experiment ID of an existing model.
    :param run_id_backbone: Experiment ID of a backbone.
    :param dataset_path: Path to dataset.
    :param data_size: Desired size of the data.
    :param num_features: Number of feature channels.
    :param split_size: Number of channels dedicated to each component.
    :param reconstruction_size: Number of channels assigned for reconstruction.
    :param batch_size: Batch size.
    :param alpha: Multiplier of the entropy minimization objective.
    :param learning_rate: Learning rate of the SGD optimizer.
    :param momentum: Momentum of the SGD optimizer.
    :param patience: Patience of the early-stopping algorithm.
    :param max_num_epochs: Maximum number of epochs.
    """

    representation_size = reconstruction_size + 2 * split_size

    logger = MLFlowLogger(tags={'model': EntropyBottleneckBaseline.__name__, 'dataset': Cityscapes.__name__})

    dataset_path = io.localize(dataset_path, overwrite=False)

    transform_train = processing.create_classifier_transformer(data_size, means=None, variances=None)
    dataset_train = Cityscapes(dataset_path, transform_train, split='train')
    transform_validation = processing.create_saeed_transformer(data_size, means=None, variances=None)
    dataset_validation = Cityscapes(dataset_path, transform_validation, split='val')

    if run_id_backbone:
        transformer = state.load_model(run_id_backbone, Baseline).transformer
    else:
        transformer = model_lst.transformer(Cityscapes.image_channels, num_features, num_features)

    model = EntropyBottleneckBaseline(
        transformer=transformer,
        reconstructor=model_lst.anderson_lst_upsample(
            in_channels=reconstruction_size,
            out_channels=Cityscapes.image_channels,
        ),
        segmentation_classifier=model_lst.anderson_lst_upsample(
            in_channels=split_size,
            out_channels=Cityscapes.num_classes,
        ),
        depth_classifier=model_lst.anderson_lst_upsample(
            in_channels=split_size,
            out_channels=Cityscapes.depth_output_channels,
        ),
        entropy_model=EntropyBottleneckModel(
            channels=representation_size,
        ),
        bits_reconstruction=AutoregressiveEntropyModel(
            num_channels=reconstruction_size,
            factorized=True,
        ),
        bits_segmentation=AutoregressiveEntropyModel(
            num_channels=split_size,
            factorized=True,
        ),
        bits_depth=AutoregressiveEntropyModel(
            num_channels=split_size,
            factorized=True,
        ),
        split_size=split_size,
        reconstruction_size=reconstruction_size,
        num_classes=Cityscapes.num_classes,
        ignore_index=Cityscapes.ignore_index,
        alpha=alpha,
        learning_rate=learning_rate,
        momentum=momentum,
    )

    if run_id:
        state.load_state(run_id, model)

    trainer = Trainer(
        max_epochs=max_num_epochs,
        logger=logger,
        enable_progress_bar=False,
        log_every_n_steps=utils.get_num_steps(dataset_train, batch_size),
        gpus=torch.cuda.device_count(),
        callbacks=[
            EarlyStopping(
                monitor='Validation Loss Tasks',
                mode='min',
                patience=patience,
            ),
            MLFlowModelCheckpoint(
                monitor='Validation Loss Tasks',
                mode='min',
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
            num_workers=4,
        ),
        val_dataloaders=DataLoader(
            dataset=dataset_validation,
            batch_size=batch_size,
            num_workers=4,
        ),
    )


if __name__ == '__main__':
    defopt.run(train)
