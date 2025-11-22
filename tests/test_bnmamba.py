
import torch
from brainmamba.models.bnmamba import BNMamba, MessagePassingLayer

def test_mpnn_layer():
    batch_size = 2
    num_nodes = 10
    d_model = 8

    model = MessagePassingLayer(d_model=d_model)

    x = torch.randn(batch_size, num_nodes, d_model)
    adj = torch.rand(batch_size, num_nodes, num_nodes)
    adj = (adj > 0.5).float()

    y = model(x, adj)
    assert y.shape == (batch_size, num_nodes, d_model)

def test_bnmamba_full():
    batch_size = 2
    num_nodes = 10
    d_model = 8

    model = BNMamba(d_model=d_model, n_mpnn_layers=1, n_ssm_layers=1)

    adj = torch.rand(batch_size, num_nodes, num_nodes)

    graph_enc = model(adj)
    assert graph_enc.shape == (batch_size, d_model)

    node_enc, graph_enc = model(adj, return_node_encodings=True)
    assert node_enc.shape == (batch_size, num_nodes, d_model)
    assert graph_enc.shape == (batch_size, d_model)
