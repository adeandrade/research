from typing import List

import numba
import numpy as np


ONE_BYTES = tuple(bytes([integer]) for integer in range(256))


@numba.njit
def int_to_byte_array(integer: int) -> List[bytes]:
    byte_array = [ONE_BYTES[integer & 0xff]]

    integer >>= 8
    byte_array.append(ONE_BYTES[integer & 0xff])

    integer >>= 8
    byte_array.append(ONE_BYTES[integer & 0xff])

    integer >>= 8
    byte_array.append(ONE_BYTES[integer & 0xff])

    return byte_array


@numba.njit
def int_array_to_bytes(integers: np.ndarray) -> bytes:
    return b''.join([byte for integer in integers for byte in int_to_byte_array(integer)])


@numba.njit
def byte_array_to_int(byte_string: bytes) -> int:
    integer = int(byte_string[0])

    integer |= int(byte_string[1]) << 8
    integer |= int(byte_string[2]) << 16
    integer |= int(byte_string[3]) << 24

    return integer


@numba.njit
def bytes_to_int_array(byte_string: bytes) -> np.ndarray:
    return np.array(
        [
            byte_array_to_int(byte_string[index:index + 4])
            for index in range(0, len(byte_string), 4)
        ],
        dtype=np.int64,
    )
