import numba
import numpy as np
from numba.typed import List

import coding.range as coding
from coding.type import StatesType


@numba.njit
def encode_symbol(
    symbol: int,
    cdf_index: int,
    cdfs: np.ndarray,
    cdf_sizes: np.ndarray,
    offsets: np.ndarray,
) -> StatesType:
    symbol = symbol - offsets[cdf_index]
    cdf = cdfs[cdf_index][: cdf_sizes[cdf_index]]

    return coding.encode_symbol(symbol, cdf)


@numba.njit
def encode_symbols(
    symbols: np.ndarray,
    indices: np.ndarray,
    cdfs: np.ndarray,
    cdf_sizes: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    states = List()  # type: ignore

    for symbol, cdf_index in zip(symbols.flatten(), indices.flatten()):
        states.extend(encode_symbol(symbol, cdf_index, cdfs, cdf_sizes, offsets))

    return coding.encode_states(states)


@numba.njit
def decode_symbol(
    state: int,
    position: int,
    cdf_index: int,
    codes: np.ndarray,
    cdfs: np.ndarray,
    cdf_sizes: np.ndarray,
    offsets: np.ndarray,
) -> tuple[int, int, int]:
    cdf = cdfs[cdf_index][: cdf_sizes[cdf_index]]

    output, state_new, position_new = coding.decode_symbol(state, position, codes, cdf)

    output = output + offsets[cdf_index]

    return output, state_new, position_new


@numba.njit
def decode_symbols(
    state: int,
    position: int,
    cdf_indices: np.ndarray,
    codes: np.ndarray,
    cdfs: np.ndarray,
    cdf_sizes: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    symbols = np.empty_like(cdf_indices)

    for index, cdf_index in enumerate(cdf_indices):
        symbols[index], state, position = decode_symbol(state, position, cdf_index, codes, cdfs, cdf_sizes, offsets)

    return symbols, state, position
