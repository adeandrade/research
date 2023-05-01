import numpy as np
import torch
from compressai.entropy_models import GaussianConditional

import coding.normal as normal
from coding.normal import SCALE_MIN, TAIL_MASS


def test_scale_table():
    scale_table = normal.get_scale_table()

    expected = torch.tensor(scale_table)
    expected_compressai = GaussianConditional._prepare_scale_table(scale_table.tolist())
    actual = scale_table

    assert np.all(expected.numpy() == actual)
    assert torch.all(expected == torch.from_numpy(actual))
    assert torch.all(expected == torch.tensor(actual))
    assert actual.tolist() == expected.tolist()
    assert actual.tolist() == expected_compressai.tolist()


def test_state():
    scale_table = normal.get_scale_table()

    actual_cdfs, actual_cdf_sizes, actual_offsets = normal.calculate_cdfs(scale_table, TAIL_MASS)

    expected_model = GaussianConditional(
        scale_table=list(scale_table),
        scale_bound=SCALE_MIN,
    )
    expected_model.update()

    actual_cdfs = torch.from_numpy(actual_cdfs)
    torch.sum(actual_cdfs == expected_model.quantized_cdf)

    assert actual_cdfs.tolist() == expected_model.quantized_cdf.tolist()
    assert actual_cdf_sizes.tolist() == expected_model.cdf_length.tolist()
    assert actual_offsets.tolist() == expected_model.offset.tolist()
