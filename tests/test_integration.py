"""Tests for neurosymbolic integration."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurosymbolic import NeurosymbolicSystem


class TestNeurosymbolicSystem:
    @pytest.fixture
    def model(self):
        return NeurosymbolicSystem(
            perception_config={
                "backbone": "efficientnet_b0",
                "feature_dim": 256,
                "num_concepts": 50,
            }
        )
    
    def test_perceive(self, model):
        x = torch.randn(2, 3, 224, 224)
        output = model.perceive(x, threshold=0.6)
        
        assert "neural" in output
        assert "symbolic" in output
        assert len(output["symbolic"]) == 2
    
    def test_reason(self, model):
        symbolic_scene = [("car", 0.9), ("moving", 0.8)]
        reasoning = model.reason(symbolic_scene, object_id="obj1")
        
        assert "num_derived" in reasoning
        assert "derived_facts" in reasoning
        assert "all_facts" in reasoning
    
    def test_forward(self, model):
        x = torch.randn(2, 3, 224, 224)
        output = model.forward(x, threshold=0.6)
        
        assert "perception" in output
        assert "reasoning" in output
        assert len(output["reasoning"]) == 2
    
    def test_query(self, model):
        x = torch.randn(1, 3, 224, 224)
        query = ("vehicle", ("obj0",))
        
        proofs = model.query(x, query, threshold=0.5)
        assert isinstance(proofs, list)
    
    def test_explain_prediction(self, model):
        x = torch.randn(1, 3, 224, 224)
        fact = ("vehicle", ("obj0",))
        
        explanations = model.explain_prediction(x, fact, threshold=0.5)
        assert isinstance(explanations, list)