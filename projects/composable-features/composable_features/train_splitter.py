from typing import Tuple, Optional

import defopt
import sfu_torch_lib.io as io
import sfu_torch_lib.mlflow as mlflow
import sfu_torch_lib.slack as slack
import sfu_torch_lib.state as state
import sfu_torch_lib.utils as utils
import torch
from compressai.entropy_models import EntropyBottleneck
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sfu_torch_lib.mlflow import MLFlowLogger, MLFlowModelCheckpoint
from torch.utils.data import DataLoader

import composable_features.model_lst as model_lst
import composable_features.processing as processing
from composable_features.dataset_cityscapes import Cityscapes
from composable_features.model_entropy import AutoregressiveEntropyModel
from composable_features.model_splitter import InformationSplitter, Baseline


@slack.notify
@mlflow.log
def train(
        *,
        run_id: Optional[str] = None,
        run_id_backbone: Optional[str] = None,
        dataset_path: str = 's3://datasets/cityscapes.zip',
        data_size: Tuple[int, int] = (256, 512),
        num_channels: int = 128,
        num_heads: int = 32,
        num_heads_reconstruction: int = 32,
        alpha: float = 1.,
        gamma: float = 100.,
        temperature: float = 1.,
        threshold: float = .1,
        learning_rate: float = .0001,
        momentum: float = .937,
        batch_size: int = 10,
        patience: int = 30,
        max_num_epochs: int = 500,
) -> None:
    """
    Trains a model.

    :param run_id: Experiment ID of an existing model.
    :param run_id_backbone: Experiment ID of a backbone model.
    :param dataset_path: Path to dataset.
    :param data_size: Desired size of the data.
    :param num_channels: Number of feature channels.
    :param num_heads: Number of heads to use.
    :param num_heads_reconstruction: Number of heads to use for reconstruction.
    :param alpha: Scaling factor for entropy loss.
    :param gamma: Scaling factor for channel selection penalty.
    :param temperature: Temperature parameter for the softmax.
    :param threshold: Threshold on the coefficient for the selection of a factor.
    :param learning_rate: Learning rate of the SGD optimizer.
    :param momentum: Momentum of the Adam optimizer.
    :param batch_size: Batch size.
    :param patience: Patience of the early-stopping algorithm.
    :param max_num_epochs: Maximum number of epochs.
    """

    in_features_kernel_encoder = int(3 * 3 * 2.5 * num_channels)

    logger = MLFlowLogger(tags={'model': InformationSplitter.__name__, 'dataset': Cityscapes.__name__})

    dataset_path = io.localize(dataset_path, overwrite=False)

    transform_train = processing.create_classifier_transformer(data_size, means=None, variances=None)
    dataset_train = Cityscapes(dataset_path, transform_train, split='train')
    transform_validation = processing.create_saeed_transformer(data_size, means=None, variances=None)
    dataset_validation = Cityscapes(dataset_path, transform_validation, split='val')

    if run_id_backbone:
        transformer = state.load_model(run_id_backbone, Baseline).transformer
    else:
        transformer = model_lst.transformer(Cityscapes.image_channels, num_channels, num_channels)

    model = InformationSplitter(
        transformer=transformer,
        reconstruction_model=model_lst.anderson_lst_upsample(num_heads_reconstruction, Cityscapes.image_channels),
        segmentation_model=model_lst.anderson_lst_upsample(num_heads, Cityscapes.num_classes),
        depth_model=model_lst.anderson_lst_upsample(num_heads, Cityscapes.depth_output_channels),
        entropy_model=EntropyBottleneck(channels=num_channels),
        kernel_encoder=model_lst.kernel_encoder(in_features=in_features_kernel_encoder),
        bits_model=AutoregressiveEntropyModel(num_channels=num_channels, factorized=True),
        num_classes=Cityscapes.num_classes,
        ignore_index=Cityscapes.ignore_index,
        alpha=alpha,
        gamma=gamma,
        temperature=temperature,
        threshold=threshold,
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
