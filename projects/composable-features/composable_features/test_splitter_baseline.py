from typing import Tuple

import defopt
import sfu_torch_lib.io as io
import sfu_torch_lib.mlflow as mlflow
import sfu_torch_lib.state as state
import torch
from pytorch_lightning import Trainer
from sfu_torch_lib.mlflow import MLFlowLogger
from torch.utils.data import DataLoader

import composable_features.processing as processing
from composable_features.dataset_cityscapes import Cityscapes
from composable_features.model_splitter import Baseline


@mlflow.log
def evaluate(
        *,
        run_id: str,
        dataset_path: str = 's3://datasets/cityscapes.zip',
        data_size: Tuple[int, int] = (256, 512),
        batch_size: int = 10,
) -> None:
    """
    Evaluates a model.

    :param run_id: Experiment ID of a existing model.
    :param dataset_path: Path to dataset.
    :param data_size: Desired size of the data.
    :param batch_size: Batch size.
    """

    logger = MLFlowLogger(tags={'model': Baseline.__name__, 'dataset': Cityscapes.__name__})

    dataset_path = io.localize(dataset_path, overwrite=False)

    transform = processing.create_saeed_transformer(data_size, means=None, variances=None)
    dataset = Cityscapes(dataset_path, transform, split='val')

    model = state.load_model(run_id, Baseline)

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
