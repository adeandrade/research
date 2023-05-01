import importlib
from typing import Optional

import defopt
import sfu_torch_lib.io as io
import sfu_torch_lib.mlflow as mlflow
import sfu_torch_lib.slack as slack
import sfu_torch_lib.state as state
from pytorch_lightning import Trainer
from sfu_torch_lib.mlflow import MLFlowLogger
from torch.utils.data import DataLoader

import conditional_residual.processing_cityscapes as processing
from conditional_residual.dataset_cityscapes import Cityscapes
from conditional_residual.model_baselines import Segmentation


@slack.notify
@mlflow.install
def train(
        *,
        model_type: str,
        run_id_pretrained: str,
        run_id: Optional[str] = None,
        dataset_path: str = 's3://datasets/cityscapes.zip',
        batch_size: int = 10,
        split: str = 'validation',
) -> None:
    """
    Trains a model.

    :param model_type: Module and class to use as model.
    :param run_id: Experiment ID of an existing run to continue.
    :param run_id_pretrained: Experiment ID of a pretrained model to start with.
    :param dataset_path: Path to dataset.
    :param batch_size: Batch size.
    :param split: Dataset split to use.
    """
    dataset_path = io.localize_dataset(dataset_path)

    transform = processing.create_test_transformer(means=None, scales=None)
    dataset = Cityscapes(dataset_path, transform, split=split)

    if run_id_pretrained:
        module_name, class_name = model_type.rsplit('.', maxsplit=1)
        model_class = getattr(importlib.import_module(module_name), class_name)
        model = state.load_model(run_id_pretrained, model_class)

    else:
        model = Segmentation(
            num_classes=Cityscapes.num_classes,
            ignore_index=Cityscapes.ignore_index,
            learning_rate=0.,
            momentum=0.,
            weight_decay=0.,
        )

    logger = MLFlowLogger(tags={'model': model.__class__.__name__, 'dataset': Cityscapes.__name__})

    trainer = Trainer(
        logger=logger,
        enable_progress_bar=False,
        accelerator='auto',
    )

    trainer.test(
        model=model,
        dataloaders=DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
    )


if __name__ == '__main__':
    defopt.run(train)
