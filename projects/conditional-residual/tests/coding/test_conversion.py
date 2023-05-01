import numpy as np

import coding.conversion as conversion


def test_int_to_byte_conversion():
    int_array = np.array([524288, 214748364])
    expected = b'\x00\x00\x08\x00\xcc\xcc\xcc\x0c'

    actual = conversion.int_array_to_bytes(int_array)

    assert actual == expected


def test_byte_to_int_conversion():
    byte_string = b'\x00\x00\x08\x00\xcc\xcc\xcc\x0c'
    expected = np.array([524288, 214748364])

    actual = conversion.bytes_to_int_array(byte_string)

    assert actual.tolist() == expected.tolist()
