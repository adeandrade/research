from typing import Tuple, Optional

import numba
import numpy as np
from numba.typed import List

from coding.type import StatesType


INITIAL_STATE = 2 ** 31
PRECISION = 16
BYPASS_PRECISION = 4


@numba.njit
def encode_cumulative_probability(
        cumulative_probability: int,
        probability: int,
        state: int,
        scale_bits: int,
) -> Tuple[Optional[int], int]:

    data, state_new = None, state

    state_maximum = ((INITIAL_STATE >> scale_bits) << 32) * probability
    if state >= state_maximum:
        data = state & 0xffffffff
        state_new >>= 32

    state_new = ((state_new // probability) << scale_bits) + (state_new % probability) + cumulative_probability

    return data, state_new


@numba.njit
def encode_bits(cumulative_probability: int, state: int, num_bits: int) -> Tuple[Optional[int], int]:
    probability = 1 << (16 - num_bits)

    data, state_new = None, state

    state_maximum = ((INITIAL_STATE >> 16) << 32) * probability
    if state >= state_maximum:
        data = state & 0xffffffff
        state_new >>= 32

    state_new = (state_new << num_bits) | cumulative_probability

    return data, state_new


@numba.njit
def encode_states(states: StatesType) -> np.ndarray:
    codes = np.empty((len(states),), dtype=np.int64)
    index = len(codes) - 1
    state = INITIAL_STATE

    while len(states) > 0:
        cumulative_probability, probability, bypass = states.pop()

        if bypass:
            code, state = encode_bits(cumulative_probability, state, BYPASS_PRECISION)
        else:
            code, state = encode_cumulative_probability(cumulative_probability, probability, state, PRECISION)

        if code is not None:
            codes[index] = code
            index -= 1

    codes[index - 1] = state & 0xffffffff
    state >>= 32
    codes[index] = state & 0xffffffff

    codes = codes[index - 1:]

    return codes


@numba.njit
def encode_symbol(symbol: int, cdf: np.ndarray) -> StatesType:
    states = List()

    symbol_max = len(cdf) - 2

    if symbol < 0:
        symbol_raw = -2 * symbol - 1
        symbol = symbol_max
    elif symbol >= symbol_max:
        symbol_raw = 2 * (symbol - symbol_max)
        symbol = symbol_max
    else:
        symbol_raw = 0

    # assert 0 <= symbol < len(cdf) - 1

    states.append((cdf[symbol], cdf[symbol + 1] - cdf[symbol], False))

    # handle out-of-bound symbols
    if symbol == symbol_max:
        num_bypasses = 0
        while (symbol_raw >> (num_bypasses * BYPASS_PRECISION)) != 0:
            num_bypasses += 1

        value = num_bypasses
        max_bypass_value = (1 << BYPASS_PRECISION) - 1
        while value >= max_bypass_value:
            states.append((numba.int32(max_bypass_value), numba.int32(0), True))
            value -= max_bypass_value

        states.append((numba.int32(value), numba.int32(0), True))

        for index in range(num_bypasses):
            value = (symbol_raw >> (index * BYPASS_PRECISION)) & max_bypass_value
            states.append((numba.int32(value), numba.int32(0), True))

    return states


@numba.jit
def decode_cumulative_probability(state: int, scale_bits: int) -> int:
    return state & ((1 << scale_bits) - 1)


@numba.njit
def advance_state(
        state: int,
        position: int,
        codes: np.ndarray,
        cumulative_probability: int,
        probability: int,
        scale_bits: int,
) -> Tuple[int, int]:

    mask = (1 << scale_bits) - 1

    state_new = probability * (state >> scale_bits) + (state & mask) - cumulative_probability
    position_new = position

    if state_new < INITIAL_STATE:
        state_new = (state_new << 32) | codes[position]
        position_new += 1

    return state_new, position_new


@numba.njit
def decode_bits(state: int, position: int, codes: np.ndarray, num_bits: int) -> Tuple[int, int, int]:
    value = state & ((1 << num_bits) - 1)

    state_new = state >> num_bits
    position_new = position

    if state_new < INITIAL_STATE:
        state_new = (state_new << 32) | codes[position]
        position_new += 1

    return value, state_new, position_new


@numba.njit
def find_index(element: int, sequence: np.ndarray) -> int:
    for index, value in enumerate(sequence):
        if value > element:
            return index - 1

    return len(sequence)


@numba.njit
def decode_symbol(
    state: int,
    position: int,
    codes: np.ndarray,
    cdf: np.ndarray,
) -> Tuple[int, int, int]:

    # assert (state == -1 and position == -1) or (state != -1 and position != -1)

    if state == -1:
        state = (codes[0] << 0) | (codes[1] << 32)
        position = 2

    max_value = len(cdf) - 2

    cumulative_probability = decode_cumulative_probability(state, PRECISION)

    value = find_index(cumulative_probability, cdf)
    # assert value < len(cdf)

    cdf, pmf = cdf[value], cdf[value + 1] - cdf[value]

    state_new, position_new = advance_state(state, position, codes, cdf, pmf, PRECISION)

    if value == max_value:
        max_bypass_value = (1 << BYPASS_PRECISION) - 1
        value_new, state_new, position_new = decode_bits(state_new, position_new, codes, BYPASS_PRECISION)

        num_bypasses = value_new
        while value_new == max_bypass_value:
            value_new, state_new, position_new = decode_bits(state_new, position_new, codes, BYPASS_PRECISION)
            num_bypasses += value_new

        value_raw = 0
        for index in range(num_bypasses):
            value_new, state_new, position_new = decode_bits(state_new, position_new, codes, BYPASS_PRECISION)
            # assert value_new <= MAX_BYPASS_VALUE
            value_raw |= value_new << (index * BYPASS_PRECISION)

        value = value_raw >> 1
        value = -value - 1 if value_raw & 1 else value + max_value

    output = value

    return output, state_new, position_new
