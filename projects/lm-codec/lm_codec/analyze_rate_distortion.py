import collections
import itertools
import operator
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import defopt
import numpy as np
import pandas
import scipy
import seaborn
from matplotlib import pyplot
from mlflow import MlflowClient
from mlflow.entities import Metric
from pandas import DataFrame
from sfu_torch_lib import utils

from lm_codec import functions


@dataclass
class Curve:
    name: str
    metrics: list[str]
    run_ids: list[str]
    calculate_loss: Callable | None
    calculate_bpp: Callable
    calculate_distortion: Callable
    mode: str = 'min'
    lengths: Sequence[int | None] | None = None


@dataclass
class Point:
    run_id: str
    metrics: list[str]
    calculate_loss: Callable
    calculate_bpp: Callable
    calculate_distortion: Callable
    mode: str = 'min'
    length: int | None = None


@dataclass
class PointData:
    bpp: float | None = None
    distortion: float | None = None


@dataclass
class Plot:
    experiments: Sequence[Curve]
    distortion_label: str
    filename: str
    rate_label: str = 'Rate'
    legend_label: str = 'Method'
    anchor: str | None = None
    pearson: bool = False
    normalize_distortion: bool = False
    normalize_rate: bool = False
    baseline_distortion: Point | PointData | None = None
    baseline_bpp: Point | PointData | None = None
    lower_bound: Point | None = None


def confidence_interval(data: np.ndarray, confidence: float = 0.99) -> float:
    standard_error = scipy.stats.sem(data)
    return standard_error * scipy.stats.t.ppf((1 + confidence) / 2, len(data) - 1)


def normalize(vector):
    maximum = np.max(vector)
    minimum = np.min(vector)

    return (vector - minimum) / (maximum - minimum)


def sort_metrics(*metric_lists: Sequence[Metric]) -> list[list[float]]:
    metrics_collected = collections.defaultdict(list)

    for list_index, metrics in enumerate(metric_lists):
        for metric in metrics:
            if list_index >= len(metrics_collected[metric.step]):
                metrics_collected[metric.step].append(metric.value)
            else:
                metrics_collected[metric.step][-1] = metric.value

    metrics_sorted = sorted(metrics_collected.items(), key=operator.itemgetter(0))

    return [values for _, values in metrics_sorted]


def find_best_metrics(
    run_id: str,
    calculate_loss: Callable,
    metric_labels: list[str],
    mode: str = 'min',
    length: int | None = None,
) -> list[float]:
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
        if mode == 'min' and loss <= best_loss or mode == 'max' and loss >= best_loss:
            best_loss = loss
            best_metric = metric

    return best_metric


def zip_with_repeat[T](*sequences: Sequence[T]) -> Iterable[Iterable[T]]:
    return zip(*(itertools.repeat(sequence[0]) if len(sequence) == 1 else sequence for sequence in sequences))


def get_all_metrics(run_id: str, metric_labels: list[str]) -> list[list[float]]:
    client = MlflowClient()

    return [
        [float(metric.value) for metric in step_metrics]
        for step_metrics in zip_with_repeat(
            *(
                client.get_metric_history(
                    run_id,
                    metric_label,
                )
                for metric_label in metric_labels
            )
        )
    ]


def get_run_metrics(run_id: str, experiment: Curve, index: int) -> list[list[float]]:
    if experiment.calculate_loss:
        metrics = [
            find_best_metrics(
                run_id=run_id,
                calculate_loss=experiment.calculate_loss,
                metric_labels=experiment.metrics,
                mode=experiment.mode,
                length=experiment.lengths[index] if experiment.lengths else None,
            )
        ]

    else:
        metrics = get_all_metrics(run_id, experiment.metrics)

    return metrics


def create_dataframe(experiments: Sequence[Curve]) -> DataFrame:
    data = [
        (
            experiment.name,
            experiment.calculate_bpp(*metrics),
            experiment.calculate_distortion(*metrics),
        )
        for experiment in experiments
        for index, run_id in enumerate(experiment.run_ids)
        for metrics in get_run_metrics(run_id, experiment, index)
    ]

    return DataFrame(data, columns=['Method', 'Rate', 'Distortion'])


def plot_results(
    dataframe: DataFrame,
    path: str,
    distortion_label: str,
    rate_label: str,
    baseline_distortion: float | None = None,
    baseline_bpp: float | None = None,
    legend_label: str = 'Method',
) -> None:
    dataframe = dataframe.rename(columns={'Distortion': distortion_label})
    dataframe = dataframe.rename(columns={'Rate': rate_label})

    seaborn.set_theme(style='darkgrid', font='serif', font_scale=0.74)

    dashes = {
        method: (2, 1)
        if method.endswith((
            'Baseline',
            'Standalone',
        ))
        else ''
        for method in dataframe['Method'].unique()
    }

    if baseline_bpp is None:
        pyplot.figure(figsize=(6.4, 3.0))
        ax_1 = None

    else:
        _, (ax_1, ax_2) = pyplot.subplots(
            ncols=2,
            nrows=1,
            sharey=True,
            width_ratios=(0.9, 0.1),
            figsize=(6.4, 3.0),
        )

    ax = seaborn.lineplot(
        data=dataframe,
        x=rate_label,
        y=distortion_label,
        hue='Method',
        marker='o',
        dashes=dashes,  # type: ignore
        style='Method',
        errorbar='sd',
        ax=ax_1,
    )

    if baseline_distortion is not None:
        ax.axhline(baseline_distortion, linestyle='--', color='purple', label='Uncompressed')

    if baseline_bpp is not None:
        ax_2.axhline(baseline_distortion, linestyle='--', color='purple')
        ax_2.axvline(baseline_bpp, linestyle='--', color='purple')
        ax_2.set_xticks([baseline_bpp])
        ax_2.set_xticklabels([f'{x:,}' for x in ax_2.get_xticks()])

    pyplot.subplots_adjust(wspace=0.02, hspace=0)
    ax.legend(title=legend_label)

    pyplot.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=300)


def add_lower_bound(dataframe: DataFrame, experiment: Point) -> DataFrame:
    metrics = find_best_metrics(
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


def get_distortion(experiment: Point | PointData) -> float:
    if isinstance(experiment, PointData):
        assert experiment.distortion is not None
        return experiment.distortion

    metrics = find_best_metrics(
        run_id=experiment.run_id,
        calculate_loss=experiment.calculate_loss,
        metric_labels=experiment.metrics,
        mode=experiment.mode,
        length=experiment.length,
    )

    return experiment.calculate_distortion(*metrics)


def get_bpp(experiment: Point | PointData) -> float:
    if isinstance(experiment, PointData):
        assert experiment.bpp is not None
        return experiment.bpp

    metrics = find_best_metrics(
        run_id=experiment.run_id,
        calculate_loss=experiment.calculate_loss,
        metric_labels=experiment.metrics,
        mode=experiment.mode,
        length=experiment.length,
    )

    return experiment.calculate_bpp(*metrics)


def compute_pearson(dataframe: DataFrame) -> DataFrame:
    labels = set(dataframe['Method'])

    for label in labels:
        rate = dataframe.loc[dataframe['Method'] == label, 'Rate']
        distortion = dataframe.loc[dataframe['Method'] == label, 'Distortion']

        pearson = np.corrcoef(rate, distortion)[0, 1]

        dataframe.loc[dataframe['Method'] == label, 'Method'] = f'{label}: {round(pearson, 2)}'

        print(f'{label} Pearson Correlation: {pearson}')

    return dataframe


def compute_bd_rates(dataframe: DataFrame, anchor_label: str, piecewise: bool = False) -> DataFrame:
    rate_anchor = dataframe.loc[dataframe['Method'] == anchor_label, 'Rate']
    distortion_anchor = dataframe.loc[dataframe['Method'] == anchor_label, 'Distortion']

    labels = set(dataframe['Method']) - {anchor_label}

    for label in labels:
        rate_target = dataframe.loc[dataframe['Method'] == label, 'Rate']
        distortion_target = dataframe.loc[dataframe['Method'] == label, 'Distortion']

        bd_rate = functions.bd_rate(
            rate_1=rate_anchor,
            psnr_1=distortion_anchor,
            rate_2=rate_target,
            psnr_2=distortion_target,
            piecewise=piecewise,
        )

        dataframe.loc[dataframe['Method'] == label, 'Method'] = f'{label}: {round(bd_rate, 2)}'

        print(f'{label} BD-Rate: {bd_rate}')

    return dataframe


def analyze_rate_distortion(
    plot_type: str,
    *,
    format: str = 'pdf',
    path: str = f'{os.environ.get("PROJECT_DIR", ".")}/results/{{filename}}.{{format}}',
    error_stats: bool = False,
    verbose: bool = False,
) -> None:
    """
    Fetches metrics from a `Plot` object organizing MLFlow runs and plots it.

    :param plot_type: `Plot` object to work with (i.e.: lm_codec.experiments.RATE_DISTORTION_HYPER_PRIOR)
    :param format: plot format
    :param path: directory where to save the plot
    :param error_stats: whether to display the condifence interval of distortion for each method and rate.
    :param verbose: whether to display `DataFrame` with results
    """
    plot = utils.get_class(plot_type)

    dataframe = create_dataframe(plot.experiments)

    if plot.pearson:
        dataframe = compute_pearson(dataframe)

    if plot.lower_bound:
        dataframe = add_lower_bound(dataframe, plot.lower_bound)

    if plot.anchor:
        dataframe = compute_bd_rates(dataframe, plot.anchor)

    baseline_distortion = get_distortion(plot.baseline_distortion) if plot.baseline_distortion else None
    baseline_bpp = get_bpp(plot.baseline_bpp) if plot.baseline_bpp else None

    if error_stats:
        print(dataframe.groupby(['Method', 'Rate']).agg({'Distortion': ['mean', confidence_interval]}).to_string())

    if verbose:
        print(dataframe.to_string())

    if plot.normalize_distortion:
        dataframe['Distortion'] = dataframe.groupby('Method')['Distortion'].transform(lambda x: normalize(x.values))

    if plot.normalize_rate:
        dataframe['Rate'] = dataframe.groupby('Method')['Rate'].transform(lambda x: normalize(x.values))

    path = path.format(filename=plot.filename, format=format)

    plot_results(
        dataframe,
        path,
        plot.distortion_label,
        plot.rate_label,
        baseline_distortion,
        baseline_bpp,
        plot.legend_label,
    )


def main():
    defopt.run(analyze_rate_distortion)


if __name__ == '__main__':
    main()
