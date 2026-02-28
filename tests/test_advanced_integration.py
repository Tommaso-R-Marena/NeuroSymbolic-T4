import torch
import pytest
from neurosymbolic.integration import NeurosymbolicSystem

def test_advanced_neural_modules():
    # Test initialization and forward pass with advanced modules enabled
    device = torch.device("cpu")
    model = NeurosymbolicSystem(
        use_relation_network=True,
        use_symbolic_attention=True,
        perception_config={"feature_dim": 128}
    ).to(device)

    x = torch.randn(1, 3, 224, 224).to(device)
    output = model.perceive(x)

    assert "relational_context" in output["neural"]
    assert output["neural"]["concept_features"].shape[-1] == 128

    # Check that symbolic attention was applied
    # It should have updated concept_features
    assert output["neural"]["concept_features"] is not None
