
import torch
from brainmamba.models.btmamba import BTMamba, CrossVariateMLP, VariateEncoder

def test_cross_variate_mlp():
    batch_size = 2
    num_variates = 10
    seq_len = 16
    d_model = 10 # num_variates must equal d_model for simple case

    model = CrossVariateMLP(d_model=d_model)
    x = torch.randn(batch_size, num_variates, seq_len)
    y = model(x)

    assert y.shape == (batch_size, num_variates, seq_len)

def test_variate_encoder_shape():
    batch_size = 2
    num_variates = 5
    seq_len = 16
    d_model = 8

    # seq_len != d_model, should trigger projection logic
    # Must provide input_dim (which corresponds to seq_len for VariateEncoder projection)
    model = VariateEncoder(d_model=d_model, n_layers=1, input_dim=seq_len)
    x = torch.randn(batch_size, num_variates, seq_len)

    y = model(x)
    assert y.shape == (batch_size, num_variates, d_model)

def test_btmamba_full():
    batch_size = 2
    num_variates = 5
    seq_len = 16
    d_model = 8

    # Must provide input_dim (num_variates) and seq_len if they differ from d_model
    model = BTMamba(d_model=d_model, n_layers=1, input_dim=num_variates, seq_len=seq_len)
    x = torch.randn(batch_size, num_variates, seq_len)

    # Test with node encodings return
    node_enc, brain_enc = model(x, return_node_encodings=True)

    assert node_enc.shape == (batch_size, d_model, d_model) # projected num_variates -> d_model
    assert brain_enc.shape == (batch_size, d_model)
