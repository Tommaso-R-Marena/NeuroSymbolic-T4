"""Additional tests for enhanced features."""

import pytest
import torch
import math
from neurosymbolic.neural import PositionalEncoding
from neurosymbolic.integration import NeurosymbolicSystem

def test_positional_encoding_2d():
    dim = 256
    H, W = 14, 14
    pe = PositionalEncoding(dim)
    x = torch.zeros(1, dim, H, W)
    output = pe(x)

    assert output.shape == (1, dim, H, W)
    assert not torch.allclose(output, x)

    # Check spatial variation
    assert not torch.allclose(output[:, :, 0, 0], output[:, :, 0, 1])
    assert not torch.allclose(output[:, :, 0, 0], output[:, :, 1, 0])

def test_ablation_flags():
    # Test disable_gnn
    model_no_gnn = NeurosymbolicSystem(disable_gnn=True)
    assert model_no_gnn.reasoner.use_gnn is False

    # Test disable_fpn
    model_no_fpn = NeurosymbolicSystem(disable_fpn=True)
    assert model_no_fpn.perception.use_fpn is False

def test_reproducibility():
    from neurosymbolic import set_seed

    x = torch.ones(1, 3, 224, 224)

    set_seed(42)
    model1 = NeurosymbolicSystem()
    out1 = model1.perception(x)["concepts"]

    set_seed(42)
    model2 = NeurosymbolicSystem()
    out2 = model2.perception(x)["concepts"]

    assert torch.allclose(out1, out2)
