import os
from dataclasses import dataclass

import defopt
import numpy as np
import pandas
import sfu_torch_lib.utils as utils
from pandas import DataFrame

import common_information.analyze_rate_distortion as rd
from common_information.analyze_rate_distortion import Curve, CurveData, Point


@dataclass
class PlotMI:
    joint: Curve
    independent: Curve
    experiments: list[Curve | CurveData]
    distortion_label: str
    filename: str
    rate_label: str = 'Common Information (BPP)'
    legend_label: str = 'Method: BD-Rate (%)'
    baseline: Point | None = None


def interpolate_point(points: np.ndarray, x: float) -> float:
    _, num_points = points.shape

    assert len(points[0]) >= 2

    index = np.searchsorted(points[0], x)

    if index == 0:
        first, second = points[:, 0], points[:, 1]
    elif index == num_points:
        first, second = points[:, -2], points[:, -1]
    else:
        first, second = points[:, index - 1], points[:, index]

    y = (second[1] - first[1]) / (second[0] - first[0])
    y = first[1] + (x - first[0]) * y

    return y


def sort_points(points: np.ndarray, axis: int = 0) -> np.ndarray:
    return points[:, np.argsort(points[axis])]


def interpolate_points(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    points = sort_points(points)

    combined_points = [(x, interpolate_point(points, x)) for x, _ in targets.T]
    combined_points = np.asarray(combined_points).T

    return combined_points


def create_mutual_information_curve(joint_curve: Curve, independent_curve: Curve) -> DataFrame:
    joint = rd.create_dataframe([joint_curve])
    joint = joint[['Distortion', 'Rate']].to_numpy().T

    independent = rd.create_dataframe([independent_curve])
    independent = independent[['Distortion', 'Rate']].to_numpy().T

    joint_interpolated = interpolate_points(joint, independent)
    independent_interpolated = interpolate_points(independent, joint)

    joint_all = np.concatenate((joint, joint_interpolated), axis=1)
    joint_all = sort_points(joint_all, axis=0)

    independent_all = np.concatenate((independent, independent_interpolated), axis=1)
    independent_all = sort_points(independent_all, axis=0)

    rate = np.maximum(independent_all[1] - joint_all[1], 0.001)

    mutual_information = np.stack((rate, independent_all[0]))[:, [1, 3, 4, -2]]

    dataframe = []

    for rate, distortion in mutual_information.T:
        dataframe.append(('Mutual Information (Empirical)', rate, distortion))

    dataframe = DataFrame(dataframe, columns=('Method', 'Rate', 'Distortion'))  # type: ignore

    return dataframe


def create_plot(
    plot_type: str,
    *,
    format: str = 'pdf',
    path: str = f'{os.environ["PROJECT_DIR"]}/results/{{filename}}.{{format}}',
) -> None:
    plot = utils.get_class(plot_type)

    assert isinstance(plot, PlotMI)

    dataframe = pandas.concat([
        create_mutual_information_curve(plot.joint, plot.independent),
        rd.create_dataframe(plot.experiments),
    ])

    rd.compute_bd_rates(dataframe, 'Mutual Information (Empirical)')

    baseline = rd.get_baseline(plot.baseline) if plot.baseline else None

    path = path.format(filename=plot.filename, format=format)

    rd.plot_results(dataframe, path, plot.distortion_label, plot.rate_label, baseline, plot.legend_label)


def main():
    defopt.run(create_plot)


if __name__ == '__main__':
    main()
