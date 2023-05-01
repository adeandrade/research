import math

import numba

from coding.type import Floats, Ints


@numba.vectorize
def round_away_zero(value: Floats) -> Ints:
    return numba.int32(math.copysign(math.floor(abs(value) + .5), value))


@numba.vectorize
def round_toward_zero(value: Floats) -> Ints:
    return numba.int32(math.copysign(math.ceil(abs(value) - .5), value))
