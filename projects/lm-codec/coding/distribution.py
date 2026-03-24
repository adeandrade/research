import numba
import numpy as np

from coding.range import PRECISION


@numba.njit
def pmf_to_quantized_cdf(pmf: np.ndarray, precision: int = PRECISION) -> np.ndarray:
    size = len(pmf)
    scale = 1 << precision

    pmf = np.round(pmf * scale)
    pmf = (scale / np.sum(pmf) * pmf).astype(np.int32)

    cdf = cumsum(pmf, scale)

    for i in range(size):
        if cdf[i] == cdf[i + 1]:
            best_probability = np.inf
            best_index = -1

            for j in range(size):
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
def pmfs_to_quantized_cdfs(
    pmfs: np.ndarray,
    sizes: np.ndarray,
    tail_masses: np.ndarray,
    max_length: int,
    precision: int = PRECISION,
) -> np.ndarray:
    cdfs = np.zeros((len(sizes), max_length + 2), dtype=np.int32)

    for index in range(len(pmfs)):
        pmf_with_tail = np.concatenate((pmfs[index, : sizes[index]], tail_masses[index : index + 1]), axis=0)
        cdf = pmf_to_quantized_cdf(pmf_with_tail, precision)
        cdfs[index, : len(cdf)] = cdf

    return cdfs


@numba.njit
def calculate_cdf_index(scale: float, table: np.ndarray):
    return len(table) - np.sum(scale <= table[:-1]) - 1


@numba.njit
def calculate_cdf_indices(scales: np.ndarray, table: np.ndarray) -> np.ndarray:
    indices = np.empty_like(scales, dtype=np.int32)

    for index in range(len(scales)):
        indices[index] = calculate_cdf_index(scales[index], table)

    return indices


@numba.njit
def cumsum(values: np.ndarray, max_probability: int, dtype: type = np.int32) -> np.ndarray:
    size = len(values)

    cdf = np.empty((size + 1,), dtype=dtype)
    cdf[0] = 0

    for index in range(size):
        cdf[index + 1] = values[index] + cdf[index]

    cdf[-1] = max_probability

    return cdf
