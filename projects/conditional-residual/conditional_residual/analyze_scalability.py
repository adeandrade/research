import collections
import dataclasses
import operator
from typing import Sequence, List, Callable, Tuple, Optional

import defopt
import matplotlib.pyplot as pyplot
import numpy as np
import pandas
import seaborn
from mlflow import MlflowClient
from mlflow.entities import Metric
from pandas import DataFrame

import conditional_residual.functions as functions


@dataclasses.dataclass
class Experiment:
    name: str
    loss_label: str
    metrics: List[str]
    run_ids: List[str]
    calculate_bpp: Callable
    calculate_psnr: Callable
    mode: str = 'min'
    lengths: Optional[Sequence[Optional[int]]] = None


EXPERIMENTS_SEGMENTATION = (
    Experiment(
        name='RD Curve',
        loss_label='Validation Segmentation IoU',
        mode='max',
        metrics=['Validation Bits', 'Validation Segmentation IoU'],
        run_ids=[
            'f2f59803700742229f2ba8c12876070d',
            'f0159a0fc5e741a8a784071fc3dca1dc',
            '178f110a4ad14dad926c2c923080aad8',
            'f72a2363957145f190c12cfc8edaa661',
        ],
        calculate_bpp=lambda bits, _: bits / 2048 / 1024,
        calculate_psnr=lambda _, psnr: psnr,
    ),
)


EXPERIMENTS_DETECTION = (
    Experiment(
        name='RD Curve',
        loss_label='Validation Mean Average Precision',
        mode='max',
        metrics=['Validation Bits', 'Validation Mean Average Precision'],
        run_ids=[
            '7f7c14a792fd4887981dc944c3a6bc15',
            'b30b3b6db0174a4bb433ead517f2feac',
            '96c4b0ae08304967bb145d4249e8c4ad',
            '85e1db0bb82f42839fa84aea16b85bcd',
        ],
        calculate_bpp=lambda bits, _: bits / 892220.1,
        calculate_psnr=lambda _, psnr: psnr,
    ),
)


EXPERIMENTS_CITYSCAPES = (
    Experiment(
        name='Upper Baseline',
        loss_label='Validation Loss',
        metrics=['Validation Bits', 'Validation PSNR'],
        run_ids=[
            'e119118e4e73439c97c177734ce6b521',
            '9b4a736104a74a8aa8d4d1666e28a2e1',
            'debfb861e0da433fb43a33138c653e9c',
            '77cbe1e5e8564ed992af30c63544b2de',
            'c3865d0d7d3641c9aa5dbc7871450c2f',
            'd2576fe62874408e85e0c76349cb77f8',
        ],
        calculate_bpp=lambda bits, _: bits / 2048 / 1024,
        calculate_psnr=lambda _, psnr: psnr,
    ),
    Experiment(
        name='Residual',
        loss_label='Validation Loss',
        metrics=['Validation Bits', 'Validation PSNR'],
        run_ids=[
            'af5b8bbb56f14b5ea87ac395c08e5bb5',
            'fc3cc1ccc6524098b26fb206a4a92131',
            '18c2c2e620c84e6d8ab4d9453d87bae6',
            '4f98b28496ca475386a3528821312c4e',
            '58f0b854089743f9a36ca69b1340a156',
        ],
        calculate_bpp=lambda bits, _: bits / 2048 / 1024,
        calculate_psnr=lambda _, psnr: psnr,
    ),
    Experiment(
        name='Conditional',
        loss_label='Validation Loss',
        metrics=['Validation Bits', 'Validation PSNR'],
        run_ids=[
            'a15efab119fb426f84895f4bf900b67f',
            '5c2d29cf25404c889d1b7ad84c7df2a6',
            'dab6226a94084149bb7af400ef6b1502',
            'afc4698eea3f4c1c860a3da9aa46207d',
            '5e74b64366ea42f1b0ddc2008eab63b3',
        ],
        calculate_bpp=lambda bits, _: bits / 2048 / 1024,
        calculate_psnr=lambda _, psnr: psnr,
    ),
)


EXPERIMENTS_COCO = (
    Experiment(
        name='Upper Baseline',
        loss_label='Validation Loss',
        metrics=['Validation Bits', 'Validation PSNR'],
        run_ids=[
            'b238a0acbce040f0a4a21fbf75439d33',
            'a862f0df7ba7479ea7a409554650663f',
            '111d65a11fd942ac96ed0eaf953950a0',
            'ca01069661b341418fd9918d9865a1c5',
            '9baf5caf42444b429761e832c5c4916d',
        ],
        calculate_bpp=lambda bits, _: bits / 269560.3,
        calculate_psnr=lambda _, psnr: psnr,
    ),
    Experiment(
        name='Residual',
        loss_label='Validation Loss',
        metrics=['Validation Bits', 'Validation PSNR'],
        run_ids=[
            '103120da8f074861bbeea580502841ba',
            'b68f0cbf927b429a9a17bca15ca9db37',
            'a4c68c757bb2423a9a34aca1ffac709f',
            '808c99210d46492286326a66dc8d4232',
            'f03f6ea433f64bb392b931f7ab827e1c',
        ],
        calculate_bpp=lambda bits, _: bits / 269560.3,
        calculate_psnr=lambda _, psnr: psnr,
    ),
    Experiment(
        name='Conditional',
        loss_label='Validation Loss',
        metrics=['Validation Bits', 'Validation PSNR'],
        run_ids=[
            '9760231f7eaf4e4d8074d4e3be3a318c',
            '7eade7e99cb14f0e939227b14fa6b3a4',
            '50802779b89f407293acb44577ab563b',
            '5f8c113d5a304fb6b786116c2abd5bc3',
            '50a7b12af80a453eb396e91d1721ce45',
        ],
        calculate_bpp=lambda bits, _: bits / 269560.3,
        calculate_psnr=lambda _, psnr: psnr,
    ),
)


class Experiments(Sequence[Experiment]):
    def __init__(self, data: Sequence[Experiment]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Experiment:
        return self.data[index]

    @staticmethod
    def from_string(string: str) -> 'Experiments':
        return Experiments([
            Experiment(
                name=group,
                loss_label=loss_label,
                metrics=metrics.split(','),
                run_ids=run_ids.split(','),
                calculate_bpp=lambda bpp, _: bpp,
                calculate_psnr=lambda _, psnr: psnr,
            )
            for group, loss_label, metrics, run_ids
            in (
                group.split(':')
                for group in string.split(';')
            )
        ])


def sort_metrics(*metric_lists: Sequence[Metric]) -> List[List[float]]:
    metrics_collected = collections.defaultdict(list)

    for list_index, metrics in enumerate(metric_lists):
        for metric in metrics:
            if list_index >= len(metrics_collected[metric.step]):
                metrics_collected[metric.step].append(metric.value)
            else:
                metrics_collected[metric.step][-1] = metric.value

    metrics_sorted = sorted(metrics_collected.items(), key=operator.itemgetter(0))

    metrics_values = [values for _, values in metrics_sorted]

    return metrics_values


def find_rate_distortion(
        run_id: str,
        loss_label: str,
        metric_labels: List[str],
        mode: str = 'min',
        length: Optional[int] = None,
) -> List[float]:

    client = MlflowClient()

    losses = client.get_metric_history(run_id, loss_label)
    metrics = [client.get_metric_history(run_id, metric_label) for metric_label in metric_labels]

    best_loss = np.inf if mode == 'min' else -np.inf
    best_metric = [np.inf] * len(metric_labels)

    for loss, *metric in sort_metrics(losses, *metrics)[:length]:
        if mode == 'min' and loss <= best_loss:
            best_loss = loss
            best_metric = metric

        elif mode == 'max' and loss >= best_loss:
            best_loss = loss
            best_metric = metric

    return best_metric


def create_dataframe(experiments: Experiments) -> DataFrame:
    data = [
        (experiment.name, experiment.calculate_bpp(*metrics), experiment.calculate_psnr(*metrics))
        for experiment, metrics
        in (
            (
                experiment,
                find_rate_distortion(
                    run_id=run_id,
                    loss_label=experiment.loss_label,
                    metric_labels=experiment.metrics,
                    mode=experiment.mode,
                    length=experiment.lengths[index] if experiment.lengths else None,
                ),
            )
            for experiment in experiments
            for index, run_id in enumerate(experiment.run_ids)
        )
    ]

    dataframe = DataFrame(data, columns=['Method', 'BPP', 'PSNR'])

    return dataframe


def plot(dataframe: DataFrame, baseline: Optional[float] = None, label: str = 'PSNR') -> None:
    dataframe = dataframe.rename(columns={'PSNR': label})

    seaborn.set_theme(style='darkgrid')

    dashes = {method: (2, 1) if method.endswith('Baseline') else '' for method in dataframe['Method'].unique()}

    ax = seaborn.lineplot(dataframe, x='BPP', y=label, hue='Method', marker='o', dashes=dashes, style='Method')

    if baseline is not None:
        ax.axhline(baseline, linestyle='--', color='purple', label='Uncompressed')
        pyplot.subplots_adjust(bottom=.41, wspace=0., hspace=0.)
    else:
        pyplot.subplots_adjust(wspace=0., hspace=0.)

    pyplot.legend(title='Method', loc='lower right')

    pyplot.savefig('/Users/anderson/segmentation-base.eps', bbox_inches='tight', pad_inches=0.)
    pyplot.show()


def get_segmentation_rate_distortion(run_id: str) -> Tuple[float, float]:
    bits, iou = find_rate_distortion(
        run_id=run_id,
        loss_label='Validation Segmentation IoU',
        metric_labels=['Validation Bits', 'Validation Segmentation IoU'],
        mode='max',
    )

    bpp = bits / 1024 / 2048

    return bpp, iou


def get_detection_rate_distortion(run_id: str) -> Tuple[float, float]:
    bits, mean_average_precision = find_rate_distortion(
        run_id=run_id,
        loss_label='Validation Mean Average Precision',
        metric_labels=['Validation Bits', 'Validation Mean Average Precision'],
        mode='max',
    )

    bpp = bits / 892220.1

    return bpp, mean_average_precision


def add_lower_bound(dataframe: DataFrame, run_id: str) -> DataFrame:
    bpp, _ = get_detection_rate_distortion(run_id)

    lower_bound = dataframe.loc[dataframe['Method'] == 'Upper Baseline'].copy()
    lower_bound.loc[:, 'Method'] = 'Lower Baseline'

    dataframe = pandas.concat((dataframe, lower_bound))

    dataframe.loc[dataframe['Method'] != 'Upper Baseline', 'BPP'] += bpp

    return dataframe


def compute_bd_rates(dataframe: DataFrame) -> None:
    rate_lower_bound = dataframe.loc[dataframe['Method'] == 'Lower Baseline', 'BPP']
    distortion_lower_bound = dataframe.loc[dataframe['Method'] == 'Lower Baseline', 'PSNR']

    rate_upper_bound = dataframe.loc[dataframe['Method'] == 'Upper Baseline', 'BPP']
    distortion_upper_bound = dataframe.loc[dataframe['Method'] == 'Upper Baseline', 'PSNR']

    rate_conditional = dataframe.loc[dataframe['Method'] == 'Conditional', 'BPP']
    distortion_conditional = dataframe.loc[dataframe['Method'] == 'Conditional', 'PSNR']

    rate_residual = dataframe.loc[dataframe['Method'] == 'Residual', 'BPP']
    distortion_residual = dataframe.loc[dataframe['Method'] == 'Residual', 'PSNR']

    bd_rate_conditional = functions.bj_delta(
        rate_1=rate_lower_bound,
        psnr_1=distortion_lower_bound,
        rate_2=rate_conditional,
        psnr_2=distortion_conditional,
        mode=1,
    )

    bd_rate_residual = functions.bj_delta(
        rate_1=rate_lower_bound,
        psnr_1=distortion_lower_bound,
        rate_2=rate_residual,
        psnr_2=distortion_residual,
        mode=1,
    )

    bd_rate_upper_bound = functions.bj_delta(
        rate_1=rate_lower_bound,
        psnr_1=distortion_lower_bound,
        rate_2=rate_upper_bound,
        psnr_2=distortion_upper_bound,
        mode=1,
    )

    ratio_conditional = bd_rate_conditional / bd_rate_upper_bound
    ratio_residual = bd_rate_residual / bd_rate_upper_bound

    print(f'Conditional BD-Rate: {bd_rate_conditional}')
    print(f'Residual BD-Rate: {bd_rate_residual}')
    print(f'Upper Baseline BD-Rate: {bd_rate_upper_bound}')

    print(f'Conditional Ratio: {ratio_conditional}')
    print(f'Residual Ratio: {ratio_residual}')


def main(
        *,
        experiments: Experiments = EXPERIMENTS_COCO,
        lower_bound_run_id: Optional[str] = None,
        baseline_run_id: Optional[str] = None,
        report_bd_rates: bool = False,
        distortion_label: str = 'PSNR',
) -> None:

    dataframe = create_dataframe(experiments)

    if lower_bound_run_id:
        dataframe = add_lower_bound(dataframe, lower_bound_run_id)

    if report_bd_rates:
        compute_bd_rates(dataframe)

    if baseline_run_id:
        _, baseline = get_detection_rate_distortion(baseline_run_id)
    else:
        baseline = None

    plot(dataframe, baseline, distortion_label)


if __name__ == '__main__':
    defopt.run(main, parsers={Experiments: Experiments.from_string})
