from typing import Tuple

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
    cdf = cdfs[cdf_index][:cdf_sizes[cdf_index]]

    states = coding.encode_symbol(symbol, cdf)

    return states


@numba.njit
def encode_symbols(
        symbols: np.ndarray,
        indices: np.ndarray,
        cdfs: np.ndarray,
        cdf_sizes: np.ndarray,
        offsets: np.ndarray,
) -> np.ndarray:

    states = List()

    for symbol, cdf_index in zip(symbols, indices):
        states.extend(encode_symbol(symbol, cdf_index, cdfs, cdf_sizes, offsets))

    codes = coding.encode_states(states)

    return codes


@numba.njit
def decode_symbol(
    state: int,
    position: int,
    cdf_index: int,
    codes: np.ndarray,
    cdfs: np.ndarray,
    cdf_sizes: np.ndarray,
    offsets: np.ndarray,
) -> Tuple[int, int, int]:

    cdf = cdfs[cdf_index][:cdf_sizes[cdf_index]]

    output, state_new, position_new = coding.decode_symbol(state, position, codes, cdf)

    output = output + offsets[cdf_index]

    return output, state_new, position_new
