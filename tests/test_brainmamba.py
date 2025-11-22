
import torch
from brainmamba.models.brainmamba import BrainMamba

def test_brainmamba_forward():
    batch_size = 2
    num_nodes = 10
    seq_len = 16
    d_model = 8
    num_classes = 2

    model = BrainMamba(
        d_model=d_model,
        num_classes=num_classes,
        n_ts_layers=1,
        n_mpnn_layers=1,
        n_ssm_layers=1,
        input_dim=num_nodes,
        seq_len=seq_len
    )

    timeseries = torch.randn(batch_size, num_nodes, seq_len)

    # Test standard forward
    logits = model(timeseries)
    assert logits.shape == (batch_size, num_classes)

    # Test with MI loss
    logits, mi_loss = model(timeseries, return_mi_loss=True)
    assert logits.shape == (batch_size, num_classes)
    assert isinstance(mi_loss, torch.Tensor)
    assert mi_loss.shape == ()

def test_brainmamba_inference():
    batch_size = 2
    num_nodes = 10
    seq_len = 16
    d_model = 8

    model = BrainMamba(d_model=d_model, input_dim=num_nodes, seq_len=seq_len)
    timeseries = torch.randn(batch_size, num_nodes, seq_len)

    preds, probs = model.inference(timeseries)
    assert preds.shape == (batch_size,)
    assert probs.shape == (batch_size, 2)
