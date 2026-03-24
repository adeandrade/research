import math

import numba


@numba.vectorize
def round_away_zero(value):
    return numba.int32(math.copysign(math.floor(abs(value) + 0.5), value))


@numba.vectorize
def round_toward_zero(value):
    return numba.int32(math.copysign(math.ceil(abs(value) - 0.5), value))
