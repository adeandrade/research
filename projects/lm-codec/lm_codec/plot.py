import os

import defopt
from pandas import DataFrame

from lm_codec import analyze_rate_distortion

LAMBADA_HYPER_PRIOR = [
    ('Deep Factorized Density Model', 0.2824, 355.93),
    ('Deep Factorized Density Model', 0.2723, 273.06),
    ('Deep Factorized Density Model', 0.2391, 214.98),
    ('Deep Factorized Density Model', 0.2389, 207.36),
    ('Fourier Basis Density Model', 0.2946, 541.67),
    ('Fourier Basis Density Model', 0.2800, 464.49),
    ('Fourier Basis Density Model', 0.2212, 434.07),
    ('Fourier Basis Density Model', 0.2069, 422.32),
    ('Direct-Access Entropy Model', 0.2663, 263.62),
    ('Direct-Access Entropy Model', 0.2554, 215.04),
    ('Direct-Access Entropy Model', 0.2296, 165.94),
    ('Direct-Access Entropy Model', 0.2084, 160.16),
]


LAMBADA_LAYERS = [
    ('Split Point 3', 0.2610, 230.16),
    ('Split Point 3', 0.2558, 189.74),
    ('Split Point 3', 0.2513, 152.25),
    ('Split Point 3', 0.2414, 148.68),
    ('Split Point 6', 0.2824, 355.93),
    ('Split Point 6', 0.2723, 273.06),
    ('Split Point 6', 0.2391, 214.98),
    ('Split Point 6', 0.2389, 207.36),
    ('Split Point 9', 0.2649, 492.08),
    ('Split Point 9', 0.2503, 388.11),
    ('Split Point 9', 0.2305, 288.44),
    ('Split Point 9', 0.2245, 268.34),
]


def plot(path: str = f'{os.environ["PROJECT_DIR"]}/results/{{filename}}.{{format}}'):
    """
    Plots results obtained manually.

    :param path: directory where to save the plot
    """
    lambada_hyper_prior = DataFrame(LAMBADA_HYPER_PRIOR, columns=('Method', 'Distortion', 'Rate'))  # type: ignore
    lambada_hyper_prior = analyze_rate_distortion.compute_bd_rates(
        lambada_hyper_prior,
        'Deep Factorized Density Model',
        piecewise=True,
    )

    analyze_rate_distortion.plot_results(
        lambada_hyper_prior,
        path.format(filename='lambada-hp', format='pdf'),
        'LAMBADA Accuracy',
        'Bitrate (BPT)',
        0.2961,
        768 * 16,
    )

    analyze_rate_distortion.plot_results(
        DataFrame(LAMBADA_LAYERS, columns=('Method', 'Distortion', 'Rate')),  # type: ignore
        path.format(filename='lambada-layers', format='pdf'),
        'LAMBADA Accuracy',
        'Bitrate (BPT)',
        0.2961,
        768 * 16,
    )


def main():
    defopt.run(plot)


if __name__ == '__name__':
    main()
