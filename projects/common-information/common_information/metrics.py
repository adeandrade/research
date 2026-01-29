import contextlib
from io import StringIO
from typing import Any, Literal

import numpy as np
import torch
from lightning_utilities import apply_to_collection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MeanAveragePrecisionBase


class MeanAveragePrecision(MeanAveragePrecisionBase):
    def compute(self) -> Tensor:
        return super().compute()['map']


class KeypointMeanAveragePrecision(Metric):
    detection_boxes: list[Tensor]
    detection_labels: list[Tensor]
    detection_keypoints: list[Tensor]
    detection_scores: list[Tensor]

    groundtruth_boxes: list[Tensor]
    groundtruth_labels: list[Tensor]
    groundtruth_keypoints: list[Tensor]
    groundtruth_areas: list[Tensor]

    def __init__(
        self,
        average: Literal['macro', 'micro'] = 'macro',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.average = average

        self.add_state('detection_boxes', default=[])
        self.add_state('detection_labels', default=[])
        self.add_state('detection_keypoints', default=[])
        self.add_state('detection_scores', default=[])

        self.add_state('groundtruth_boxes', default=[])
        self.add_state('groundtruth_labels', default=[])
        self.add_state('groundtruth_keypoints', default=[])
        self.add_state('groundtruth_areas', default=[])

    def update(self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]) -> None:
        for item in preds:
            self.detection_boxes.append(item['boxes'])
            self.detection_labels.append(item['labels'])
            self.detection_keypoints.append(item['keypoints'])
            self.detection_scores.append(item['scores'])

        for item in target:
            self.groundtruth_boxes.append(item['boxes'])
            self.groundtruth_labels.append(item['labels'])
            self.groundtruth_keypoints.append(item['keypoints'])
            self.groundtruth_areas.append(item['area'])

    def get_categories(self) -> list[int]:
        if len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0:
            classes = torch.cat(self.detection_labels + self.groundtruth_labels).unique().cpu().tolist()
        else:
            classes = []

        return classes

    def get_coco_dataset(
        self,
        boxes: list[Tensor],
        labels: list[Tensor],
        keypoints: list[Tensor],
        scores: list[Tensor] | None = None,
        areas: list[Tensor] | None = None,
    ) -> dict[str, Any]:
        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        for image_id, image_labels in enumerate(labels):
            image_boxes = boxes[image_id].cpu().tolist()
            image_labels = image_labels.cpu().tolist()

            images.append({'id': image_id})

            for i, image_label in enumerate(image_labels):
                assert isinstance(image_label, int), f'Type {type(image_label)} for label of box {i}, sample {image_id}'

                image_box = image_boxes[i]
                assert len(image_box) == 4, f'Got {len(image_box)} elements for box {i}, sample {image_id}'

                image_keypoints = keypoints[image_id][i]
                image_keypoints = torch.flatten(image_keypoints)
                image_keypoints = image_keypoints.cpu().tolist()

                area = image_box[2] * image_box[3] if areas is None else areas[image_id][i].cpu().item()

                annotation = {
                    'image_id': image_id,
                    'id': annotation_id,
                    'category_id': image_label,
                    'bbox': image_box,
                    'keypoints': image_keypoints,
                    'area': area,
                    'iscrowd': 0,
                }

                if scores is None:
                    num_keypoints = keypoints[image_id][i]
                    num_keypoints = num_keypoints[:, 2] > 0
                    num_keypoints = torch.sum(num_keypoints.int()).item()

                    annotation['num_keypoints'] = num_keypoints

                else:
                    score = scores[image_id][i].cpu().item()
                    assert isinstance(score, float), f'Type {type(score)} for score of box {i}, sample {image_id}'

                    annotation['score'] = score

                annotations.append(annotation)
                annotation_id += 1

        categories = [{'id': x, 'name': str(x)} for x in self.get_categories()]

        dataset = {
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }

        return dataset

    def get_coco(self) -> tuple[COCO, COCO]:
        if self.average == 'micro':
            # for micro averaging we set everything to be the same class
            groundtruth_labels = apply_to_collection(self.groundtruth_labels, Tensor, lambda x: torch.zeros_like(x))
            detection_labels = apply_to_collection(self.detection_labels, Tensor, lambda x: torch.zeros_like(x))
        else:
            groundtruth_labels = self.groundtruth_labels
            detection_labels = self.detection_labels

        coco_target, coco_preds = COCO(), COCO()

        coco_preds.dataset = self.get_coco_dataset(
            self.detection_boxes,
            detection_labels,
            self.detection_keypoints,
            scores=self.detection_scores,
        )  # type: ignore
        coco_target.dataset = self.get_coco_dataset(
            self.groundtruth_boxes,
            groundtruth_labels,
            self.groundtruth_keypoints,
            areas=self.groundtruth_areas,
        )  # type: ignore

        with contextlib.redirect_stdout(StringIO()):
            coco_target.createIndex()
            coco_preds.createIndex()

        return coco_preds, coco_target

    def stats_to_tensor_dict(self, stats: np.ndarray) -> dict[str, Tensor]:
        return {
            'MAP': torch.tensor([stats[0]], dtype=torch.float32),
            'MAP 50': torch.tensor([stats[1]], dtype=torch.float32),
            'MAP 75': torch.tensor([stats[2]], dtype=torch.float32),
            'MAP Medium': torch.tensor([stats[3]], dtype=torch.float32),
            'MAP Large': torch.tensor([stats[4]], dtype=torch.float32),
            'MAR': torch.tensor([stats[5]], dtype=torch.float32),
            'MAR 50': torch.tensor([stats[6]], dtype=torch.float32),
            'MAR 75': torch.tensor([stats[7]], dtype=torch.float32),
            'MAR Medium': torch.tensor([stats[8]], dtype=torch.float32),
            'MAR Large': torch.tensor([stats[9]], dtype=torch.float32),
        }

    def compute(self) -> Tensor:
        coco_preds, coco_target = self.get_coco()

        with contextlib.redirect_stdout(StringIO()):
            coco_eval = COCOeval(coco_target, coco_preds, iouType='keypoints')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        results = self.stats_to_tensor_dict(coco_eval.stats)
        results = results['MAP']

        return results
