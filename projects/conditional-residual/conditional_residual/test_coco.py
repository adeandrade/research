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

import conditional_residual.processing_coco as processing
from conditional_residual.dataset_coco import COCO
from conditional_residual.model_baselines import Detection


@slack.notify
@mlflow.install
def train(
        *,
        model_type: Optional[str] = None,
        run_id_pretrained: Optional[str] = None,
        run_id: Optional[str] = None,
        dataset_path: str = 's3://datasets/coco-2017.zip',
        split: str = 'validation',
) -> None:
    """
    Evaluates a model.

    :param model_type: Module and class to use as model.
    :param run_id: Experiment ID of an existing run to continue.
    :param run_id_pretrained: Experiment ID of a pretrained model to start with.
    :param dataset_path: Path to dataset.
    :param split: Dataset split to use.
    """
    dataset_path = io.localize_dataset(dataset_path)

    transform = processing.create_test_transformer()
    dataset = COCO(dataset_path, transform, split=split)

    if run_id_pretrained:
        module_name, class_name = model_type.rsplit('.', maxsplit=1)
        model_class = getattr(importlib.import_module(module_name), class_name)
        model = state.load_model(run_id_pretrained, model_class)

    else:
        model = Detection(
            num_classes=COCO.num_classes_detection,
            learning_rate=0.,
        )

    logger = MLFlowLogger(tags={'model': model.__class__.__name__, 'dataset': COCO.__name__})

    trainer = Trainer(
        logger=logger,
        enable_progress_bar=False,
        accelerator='auto',
    )

    trainer.test(
        model=model,
        dataloaders=DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=processing.collate,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        ),
    )


if __name__ == '__main__':
    defopt.run(train)
