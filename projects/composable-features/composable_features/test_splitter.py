from typing import Tuple

import defopt
import mlflow
import sfu_torch_lib.io as io
import sfu_torch_lib.mlflow as mlflow_lib
import sfu_torch_lib.state as state
import torch
from pytorch_lightning import Trainer
from sfu_torch_lib.mlflow import MLFlowLogger
from torch.utils.data import DataLoader

import composable_features.model_splitter as model_splitter
import composable_features.processing as processing
from composable_features.dataset_cityscapes import Cityscapes


@mlflow_lib.log
def evaluate(
        *,
        run_id: str,
        dataset_path: str = 's3://datasets/cityscapes.zip',
        split: str = 'val',
        data_size: Tuple[int, int] = (256, 512),
        batch_size: int = 10,
) -> None:
    """
    Trains a model.

    :param run_id: Experiment ID of an existing model.
    :param dataset_path: Path to dataset.
    :param split: Dataset split.
    :param data_size: Desired size of the data.
    :param batch_size: Batch size.
    """

    model_type = mlflow.get_run(run_id).data.tags['model']

    logger = MLFlowLogger(tags={'model': model_type, 'dataset': Cityscapes.__name__})

    dataset_path = io.localize(dataset_path, overwrite=False)

    transform = processing.create_saeed_transformer(data_size, means=None, variances=None)
    dataset = Cityscapes(dataset_path, transform, split=split)

    model = state.load_model(run_id, getattr(model_splitter, model_type))

    trainer = Trainer(
        logger=logger,
        enable_progress_bar=False,
        gpus=torch.cuda.device_count(),
    )

    trainer.test(
        model=model,
        dataloaders=DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=4,
        ),
    )


if __name__ == '__main__':
    defopt.run(evaluate)
