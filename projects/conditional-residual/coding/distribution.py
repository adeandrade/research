import numba
import numpy as np

import coding.quantization as quantization
from coding.range import PRECISION


@numba.njit
def pmf_to_quantized_cdf(pmf: np.ndarray, precision: int = PRECISION) -> np.ndarray:
    cdf = np.empty((len(pmf) + 1,), dtype=np.int32)
    cdf[0] = 0

    probability_scale = 1 << precision

    total = 0
    for index in range(len(pmf)):
        cdf[index + 1] = quantization.round_away_zero(pmf[index] * probability_scale)
        total += cdf[index + 1]

    cdf = (cdf * probability_scale / total).astype(np.int32)
    cdf = np.cumsum(cdf)
    cdf[-1] = probability_scale

    for i in range(len(cdf) - 1):
        if cdf[i] == cdf[i + 1]:
            best_probability = np.Infinity
            best_index = -1

            for j in range(len(cdf) - 1):
                probability = cdf[j + 1] - cdf[j]

                if 1 < probability < best_probability:
                    best_probability = probability
                    best_index = j

            if best_index < i:
                for j in range(best_index + 1, i + 1):
                    cdf[j] -= 1
            else:
                for j in range(i + 1, best_index + 1):
                    cdf[j] += 1

    return cdf


@numba.njit
def pmfs_to_quantized_cdfs(pmfs: np.ndarray, sizes: np.ndarray, tail_mass: np.ndarray, max_length: int) -> np.ndarray:
    cdfs = np.zeros((len(sizes), max_length + 2), dtype=np.int32)

    for index in range(len(pmfs)):
        pmf_with_tail = np.concatenate((pmfs[index, :sizes[index]], np.array([tail_mass[index]])), axis=0)
        cdf = pmf_to_quantized_cdf(pmf_with_tail)
        cdfs[index, :len(cdf)] = cdf

    return cdfs


@numba.njit
def calculate_cdf_index(scale: float, table: np.array, lower_bound: float) -> int:
    scale = numba.float32(max(abs(scale), lower_bound))

    index = len(table) - np.sum(scale <= table[:-1]) - 1

    return index


@numba.njit
def calculate_cdf_indices(scales: np.ndarray, table: np.array, lower_bound: float) -> np.ndarray:
    indices = np.empty_like(scales, dtype=np.int32)

    for index in range(len(scales)):
        indices[index] = calculate_cdf_index(scales[index], table, lower_bound)

    return indices


@numba.njit
def cumsum(values: np.ndarray, max_probability: int, dtype: np.dtype = np.int32) -> np.ndarray:
    size = len(values)

    cdf = np.empty((size + 2,), dtype=dtype)
    cdf[0] = 0

    for index in range(size):
        cdf[index + 1] = values[index] + cdf[index]

    cdf[-1] = max_probability

    return cdf
