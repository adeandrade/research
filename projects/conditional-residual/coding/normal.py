import math
from typing import Sequence, Tuple

import numba
import numpy as np
import scipy

import coding.distribution as distribution
from coding.range import PRECISION
from coding.type import Floats


SCALE_MIN = .11
SCALE_MAX = 256.
NUM_SCALES = 64
TAIL_MASS = 1e-9

SQUARE_2_PI = 2.50662827463100050242e0
EXP_MINUS_2 = 0.13533528323661269189
SQUARE_ROOT_2_RECIPROCAL = 2 ** -.5


# Approximation for 0 <= |y - 0.5| <= 3/8
P0 = (
    -5.99633501014107895267e1,
    9.80010754185999661536e1,
    -5.66762857469070293439e1,
    1.39312609387279679503e1,
    -1.23916583867381258016e0,
)

Q0 = (
    1.95448858338141759834e0,
    4.67627912898881538453e0,
    8.63602421390890590575e1,
    -2.25462687854119370527e2,
    2.00260212380060660359e2,
    -8.20372256168333339912e1,
    1.59056225126211695515e1,
    -1.18331621121330003142e0,
)

# Approximation for interval z = sqrt(-2 log y ) between 2 and 8
# i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14
P1 = (
    4.05544892305962419923e0,
    3.15251094599893866154e1,
    5.71628192246421288162e1,
    4.40805073893200834700e1,
    1.46849561928858024014e1,
    2.18663306850790267539e0,
    -1.40256079171354495875e-1,
    -3.50424626827848203418e-2,
    -8.57456785154685413611e-4,
)

Q1 = (
    1.57799883256466749731e1,
    4.53907635128879210584e1,
    4.13172038254672030440e1,
    1.50425385692907503408e1,
    2.50464946208309415979e0,
    -1.42182922854787788574e-1,
    -3.80806407691578277194e-2,
    -9.33259480895457427372e-4,
)


# Approximation for interval z = sqrt(-2 log y ) between 8 and 64
# i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890
P2 = (
    3.23774891776946035970e0,
    6.91522889068984211695e0,
    3.93881025292474443415e0,
    1.33303460815807542389e0,
    2.01485389549179081538e-1,
    1.23716634817820021358e-2,
    3.01581553508235416007e-4,
    2.65806974686737550832e-6,
    6.23974539184983293730e-9,
)

Q2 = (
    6.02427039364742014255e0,
    3.67983563856160859403e0,
    1.37702099489081330271e0,
    2.16236993594496635890e-1,
    1.34204006088543189037e-2,
    3.28014464682127739104e-4,
    2.89247864745380683936e-6,
    6.79019408009981274425e-9,
)


@numba.njit
def polevl(x: float, coefficients: Sequence[float], n: int) -> float:
    answer = coefficients[0]

    for index in range(n):
        answer = answer * x + coefficients[index + 1]

    return answer


@numba.njit
def p1evl(x: float, coefficients: Sequence[float], n: int) -> float:
    answer = x + coefficients[0]

    for index in range(n - 1):
        answer = answer * x + coefficients[index + 1]

    return answer


@numba.njit
def ppf(y0: float) -> float:
    if y0 == 0.0:
        return -np.Infinity
    elif y0 == 1.0:
        return np.Infinity
    if y0 < 0.0 or y0 > 1.0:
        raise ValueError('Cumulative probability out of range')

    code = 1
    y = y0

    if y > (1.0 - EXP_MINUS_2):
        y = 1.0 - y
        code = 0

    if y > EXP_MINUS_2:
        y = y - 0.5
        y2 = y * y
        x = y + y * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8))
        x = x * SQUARE_2_PI

        return x

    x = math.sqrt(-2.0 * math.log(y))
    x0 = x - math.log(x) / x

    z = 1.0 / x

    # y > exp(-32) = 1.2664165549e-14
    if x < 8.0:
        x1 = z * polevl(z, P1, 8) / p1evl(z, Q1, 8)
    else:
        x1 = z * polevl(z, P2, 8) / p1evl(z, Q2, 8)

    x = x0 - x1

    if code != 0:
        x = -x

    return x


@numba.vectorize
def calculate_standardized_cumulative(values: Floats) -> Floats:
    return numba.float32(.5) * math.erfc(values * numba.float32(-(2 ** -.5)))


def get_scale_table(minimum: float = SCALE_MIN, maximum: float = SCALE_MAX, num_scales: int = NUM_SCALES) -> np.ndarray:
    scale_table = np.linspace(np.log(minimum), math.log(maximum), num_scales, dtype=np.float32)
    scale_table = np.exp(scale_table)

    return scale_table


def calculate_cdfs(scale_table: np.ndarray, tail_mass: float = TAIL_MASS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    multiplier = -scipy.stats.norm.ppf(tail_mass / 2)
    pmf_centers = np.ceil(scale_table * multiplier).astype(np.int32)
    pmf_sizes = 2 * pmf_centers + 1
    pmf_max_size = np.max(pmf_sizes)

    samples = np.arange(pmf_max_size, dtype=np.int32)
    samples = np.abs(samples - pmf_centers[:, None])
    samples = samples.astype(np.float32)

    upper = calculate_standardized_cumulative((.5 - samples) / scale_table[:, None])
    lower = calculate_standardized_cumulative((-.5 - samples) / scale_table[:, None])
    pmfs = upper - lower

    tail_mass = 2 * lower[:, 0]

    cdfs = distribution.pmfs_to_quantized_cdfs(pmfs, pmf_sizes, tail_mass, pmf_max_size)
    cdf_sizes = pmf_sizes + 2
    offsets = -pmf_centers

    return cdfs, cdf_sizes, offsets


@numba.njit
def calculate_cdf(
        scale: float,
        precision: int = PRECISION,
        precision_tail_mass: int = 2,
) -> Tuple[np.ndarray, int]:

    max_probability = (1 << precision) - 1
    tail_mass = precision_tail_mass / max_probability
    length = math.floor(scale * ppf(1 - tail_mass))

    samples = np.arange(-length, length, dtype=np.float32) if length else np.array([0], dtype=np.float32)

    pmf = calculate_standardized_cumulative((samples + .5) / scale)
    pmf -= calculate_standardized_cumulative((samples - .5) / scale)
    pmf = pmf / np.sum(pmf) * (max_probability - precision_tail_mass)
    pmf = np.floor(pmf)

    cdf = distribution.cumsum(pmf, max_probability)

    offset = -length

    return cdf, offset
