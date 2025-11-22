
import torch
import pytest
from brainmamba.models.selective_ssm import SelectiveSSM, SelectiveSSMBlock

@pytest.mark.parametrize("use_parallel_scan", [True, False])
def test_selective_ssm_shape(use_parallel_scan):
    batch_size = 2
    seq_len = 16
    d_model = 8
    d_state = 4

    model = SelectiveSSM(
        d_model=d_model,
        d_state=d_state,
        use_parallel_scan=use_parallel_scan
    )

    u = torch.randn(batch_size, seq_len, d_model)
    y = model(u)

    assert y.shape == (batch_size, seq_len, d_model)

def test_selective_ssm_block_shape():
    batch_size = 2
    seq_len = 16
    d_model = 8
    d_state = 4

    model = SelectiveSSMBlock(
        d_model=d_model,
        d_state=d_state
    )

    x = torch.randn(batch_size, seq_len, d_model)
    y = model(x)

    assert y.shape == (batch_size, seq_len, d_model)
