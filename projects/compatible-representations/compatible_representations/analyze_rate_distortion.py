import collections
import operator
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import defopt
import matplotlib.pyplot as pyplot
import numpy as np
import pandas
import seaborn
import sfu_torch_lib.utils as utils
from mlflow import MlflowClient
from mlflow.entities import Metric
from pandas import DataFrame

import compatible_representations.experiments as experiments
import compatible_representations.functions as functions


@dataclass
class Curve:
    name: str
    metrics: List[str]
    run_ids: List[str]
    calculate_loss: Callable
    calculate_bpp: Callable
    calculate_distortion: Callable
    mode: str = 'min'
    lengths: Optional[Sequence[Optional[int]]] = None


@dataclass
class Point:
    run_id: str
    metrics: List[str]
    calculate_loss: Callable
    calculate_bpp: Callable
    calculate_distortion: Callable
    mode: str = 'min'
    length: Optional[int] = None


@dataclass
class Plot:
    experiments: Sequence[Curve]
    distortion_label: str
    filename: str
    rate_label: str = 'BPP'
    anchor: Optional[str] = None
    inverse: bool = True
    baseline: Optional[Point] = None
    lower_bound: Optional[Point] = None


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
    calculate_loss: Callable,
    metric_labels: List[str],
    mode: str = 'min',
    length: Optional[int] = None,
) -> List[float]:
    client = MlflowClient()

    metrics = [client.get_metric_history(run_id, metric_label) for metric_label in metric_labels]
    losses = [
        Metric(
            'Loss',
            calculate_loss(*(argument.value for argument in arguments)),
            arguments[0].timestamp,
            arguments[0].step,
        )
        for arguments in zip(*metrics)
    ]

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


def create_dataframe(experiments: Sequence[Curve]) -> DataFrame:
    data = [
        (
            experiment.name,
            experiment.calculate_bpp(*metrics),
            experiment.calculate_distortion(*metrics),
        )
        for experiment, metrics in (
            (
                experiment,
                find_rate_distortion(
                    run_id=run_id,
                    calculate_loss=experiment.calculate_loss,
                    metric_labels=experiment.metrics,
                    mode=experiment.mode,
                    length=experiment.lengths[index] if experiment.lengths else None,
                ),
            )
            for experiment in experiments
            for index, run_id in enumerate(experiment.run_ids)
        )
    ]

    dataframe = DataFrame(data, columns=['Method', 'Rate', 'Distortion'])

    return dataframe


def plot_results(
    dataframe: DataFrame,
    path: str,
    distortion_label: str,
    rate_label: str,
    baseline: Optional[float] = None,
) -> None:
    dataframe = dataframe.rename(columns={'Distortion': distortion_label})
    dataframe = dataframe.rename(columns={'Rate': rate_label})

    seaborn.set_theme(style='darkgrid')

    dashes = {
        method: (2, 1) if method.endswith('Baseline') or method.endswith('Standalone') else ''
        for method in dataframe['Method'].unique()
    }

    pyplot.figure(figsize=(6.4, 3.0))

    ax = seaborn.lineplot(
        data=dataframe,
        x=rate_label,
        y=distortion_label,
        hue='Method',
        marker='o',
        dashes=dashes,
        style='Method',
    )

    if baseline is not None:
        ax.axhline(baseline, linestyle='--', color='purple', label='Uncompressed')

    pyplot.legend(title='Method')

    pyplot.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=300)


def add_lower_bound(dataframe: DataFrame, experiment: Point) -> DataFrame:
    metrics = find_rate_distortion(
        run_id=experiment.run_id,
        calculate_loss=experiment.calculate_loss,
        metric_labels=experiment.metrics,
        mode=experiment.mode,
        length=experiment.length,
    )

    bpp = experiment.calculate_bpp(metrics)

    lower_bound = dataframe.loc[dataframe['Method'] == 'Upper Baseline'].copy()
    lower_bound.loc[:, 'Method'] = 'Lower Baseline'

    dataframe = pandas.concat((dataframe, lower_bound))

    dataframe.loc[dataframe['Method'] != 'Upper Baseline', 'BPP'] += bpp

    return dataframe


def get_baseline(experiment: Point) -> float:
    metrics = find_rate_distortion(
        run_id=experiment.run_id,
        calculate_loss=experiment.calculate_loss,
        metric_labels=experiment.metrics,
        mode=experiment.mode,
        length=experiment.length,
    )

    distortion = experiment.calculate_distortion(*metrics)

    return distortion


def compute_bd_rates(dataframe: DataFrame, anchor_label: str, inverse: bool) -> None:
    rate_anchor = dataframe.loc[dataframe['Method'] == anchor_label, 'Rate']
    distortion_anchor = dataframe.loc[dataframe['Method'] == anchor_label, 'Distortion']

    labels = set(dataframe['Method']) - {anchor_label}

    for label in labels:
        rate_target = dataframe.loc[dataframe['Method'] == label, 'Rate']
        distortion_target = dataframe.loc[dataframe['Method'] == label, 'Distortion']

        if inverse:
            bd_rate = functions.bj_delta(
                rate_1=rate_anchor,
                psnr_1=distortion_anchor,
                rate_2=rate_target,
                psnr_2=distortion_target,
            )

        else:
            bd_rate = functions.bj_delta(
                rate_1=rate_target,
                psnr_1=distortion_target,
                rate_2=rate_anchor,
                psnr_2=distortion_anchor,
            )

        print(f'{label} BD-Rate: {bd_rate}')


def main(
    plot_label: str,
    *,
    format: str = 'png',
    path: str = '/home/adeandrade/code/research/projects/compatible-representations/results/{}.{}',
) -> None:
    plot = utils.get_class(plot_label.upper(), experiments)

    dataframe = create_dataframe(plot.experiments)

    if plot.lower_bound:
        dataframe = add_lower_bound(dataframe, plot.lower_bound)

    if plot.anchor:
        compute_bd_rates(dataframe, plot.anchor, plot.inverse)

    baseline = get_baseline(plot.baseline) if plot.baseline else None

    path = path.format(plot.filename, format)

    plot_results(dataframe, path, plot.distortion_label, plot.rate_label, baseline)


if __name__ == '__main__':
    defopt.run(main)
