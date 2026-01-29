import collections
import operator
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import defopt
import matplotlib.pyplot as pyplot
import numpy as np
import pandas
import seaborn
import sfu_torch_lib.utils as utils
from mlflow import MlflowClient
from mlflow.entities import Metric
from pandas import DataFrame

import common_information.functions as functions


@dataclass
class Curve:
    name: str
    metrics: list[str]
    run_ids: list[str]
    calculate_loss: Callable
    calculate_bpp: Callable
    calculate_distortion: Callable
    mode: str = 'min'
    lengths: Sequence[int | None] | None = None


@dataclass
class CurveDistortions:
    name: str
    metrics: list[str]
    run_ids: list[str]
    calculate_loss: Callable
    calculate_bpp: Callable
    calculate_distortions: list[Callable]
    mode: str = 'min'
    lengths: Sequence[int | None] | None = None


@dataclass
class CurveData:
    name: str
    data: Sequence[tuple[float, float]]


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
class Plot:
    experiments: Sequence[Curve | CurveDistortions | CurveData]
    distortion_label: str
    filename: str
    rate_label: str = 'Rate (BPP)'
    legend_label: str = 'Method: BD-Rate (%)'
    anchor: str | None = None
    baseline: Point | float | None = None
    lower_bound: Point | None = None
    normalize: bool = False


def normalize_points(points: list[float]) -> list[float]:
    minimum = min(*points)
    maximum = max(*points)

    points = [(point - minimum) / (maximum - minimum) for point in points]

    return points


def sort_metrics(*metric_lists: Sequence[Metric]) -> list[list[float]]:
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
        if mode == 'min' and loss <= best_loss:
            best_loss = loss
            best_metric = metric

        elif mode == 'max' and loss >= best_loss:
            best_loss = loss
            best_metric = metric

    return best_metric


def get_data(experiment: Curve | CurveData | CurveDistortions) -> list[tuple[str, float, float | list[float]]]:
    if isinstance(experiment, CurveData):
        return [
            (
                experiment.name,
                rate,
                distortion,
            )
            for rate, distortion in experiment.data
        ]

    elif isinstance(experiment, CurveDistortions):
        return [
            (
                experiment.name,
                experiment.calculate_bpp(*metrics),
                [f(*metrics) for f in experiment.calculate_distortions],
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
                for index, run_id in enumerate(experiment.run_ids)
            )
        ]

    else:
        return [
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
                for index, run_id in enumerate(experiment.run_ids)
            )
        ]


def create_dataframe(experiments: Sequence[Curve | CurveData | CurveDistortions], normalize: bool = False) -> DataFrame:
    data = [data for experiment in experiments for data in get_data(experiment)]

    if isinstance(data[0][2], list):
        function = normalize_points if normalize else lambda x: x

        distortions = [
            sum(vector)
            for vector in zip(*[function(points) for points in zip(*[distortions for (*_, distortions) in data])])
        ]

        data = [(method, rate, distortion) for ((method, rate, *_), distortion) in zip(data, distortions)]

    dataframe = DataFrame(data, columns=['Method', 'Rate', 'Distortion'])  # type: ignore

    return dataframe


def plot_results(
    dataframe: DataFrame,
    path: str,
    distortion_label: str,
    rate_label: str,
    baseline: float | None = None,
    legend_label: str = 'Method',
) -> None:
    dataframe = dataframe.rename(columns={'Distortion': distortion_label})
    dataframe = dataframe.rename(columns={'Rate': rate_label})

    seaborn.set_theme(style='darkgrid', font='serif', font_scale=0.85)

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
        dashes=dashes,  # type: ignore
        style='Method',
    )

    if baseline is not None:
        ax.axhline(baseline, linestyle='--', color='purple', label='Uncompressed')

    pyplot.legend(title=legend_label)

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


def get_baseline(experiment: Point | float) -> float:
    if not isinstance(experiment, Point):
        return experiment

    metrics = find_rate_distortion(
        run_id=experiment.run_id,
        calculate_loss=experiment.calculate_loss,
        metric_labels=experiment.metrics,
        mode=experiment.mode,
        length=experiment.length,
    )

    distortion = experiment.calculate_distortion(*metrics)

    return distortion


def compute_bd_rates(dataframe: DataFrame, anchor_label: str) -> DataFrame:
    rate_anchor = dataframe.loc[dataframe['Method'] == anchor_label, 'Rate']
    distortion_anchor = dataframe.loc[dataframe['Method'] == anchor_label, 'Distortion']

    labels = set(dataframe['Method']) - {anchor_label}

    for label in labels:
        rate_target = dataframe.loc[dataframe['Method'] == label, 'Rate']
        distortion_target = dataframe.loc[dataframe['Method'] == label, 'Distortion']

        bd_rate = functions.bd_rate(
            rate_anchor,
            distortion_anchor,
            rate_target,
            distortion_target,
            piecewise=False,
        )

        dataframe.loc[dataframe['Method'] == label, 'Method'] = f'{label}: {round(bd_rate, 2)}'

        print(f'{label} BD-Rate: {bd_rate}')

    return dataframe


def print_table(plot: Plot) -> None:
    for experiment in plot.experiments:
        if not isinstance(experiment, (Curve, CurveDistortions)):
            continue

        data = []

        for index, run_id in enumerate(experiment.run_ids):
            metrics = find_rate_distortion(
                run_id=run_id,
                calculate_loss=experiment.calculate_loss,
                metric_labels=experiment.metrics,
                mode=experiment.mode,
                length=experiment.lengths[index] if experiment.lengths else None,
            )

            if len(experiment.metrics) == 6:
                _, r_a, r_b, r_c, d_a, d_b = metrics

            elif len(experiment.metrics) == 5:
                _, r_a, r_b, d_a, d_b = metrics
                r_c = 0.0

            elif len(experiment.metrics) == 4:
                _, r_c, d_a, d_b = metrics
                r_a = r_b = 0.0

            else:
                raise ValueError('Expected 4-6 metrics for detailed output.')

            r_t = r_a + r_b + r_c
            r_r = r_a + r_b + 2 * r_c

            metrics = [r_a, r_b, r_c, r_t, r_r, d_a, d_b]
            metrics = [round(metric, 3) for metric in metrics]

            data.append(metrics)

        data = sorted(data, key=lambda array: array[3], reverse=True)
        data = zip([1, 10, 25, 50, 75, 100], data)
        data = [' & '.join(str(value) for value in ([experiment.name, alpha] + row)) + ' \\\\' for (alpha, row) in data]

        print(f'{"\n".join(data)}\n\\midrule')


def create_plot(
    plot_type: str,
    *,
    format: str = 'pdf',
    path: str = f'{os.environ["PROJECT_DIR"]}/results/{{filename}}.{{format}}',
    verbose: bool = False,
) -> None:
    plot = utils.get_class(plot_type)

    assert isinstance(plot, Plot)

    dataframe = create_dataframe(plot.experiments, plot.normalize)

    if plot.lower_bound:
        dataframe = add_lower_bound(dataframe, plot.lower_bound)

    if plot.anchor:
        dataframe = compute_bd_rates(dataframe, plot.anchor)

    if verbose:
        print_table(plot)

    baseline = get_baseline(plot.baseline) if plot.baseline else None

    path = path.format(filename=plot.filename, format=format)

    plot_results(dataframe, path, plot.distortion_label, plot.rate_label, baseline, plot.legend_label)


def main():
    defopt.run(create_plot)


if __name__ == '__main__':
    main()
