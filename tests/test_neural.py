"""Tests for neural perception module."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurosymbolic.neural import PerceptionModule, AttentionPool


class TestAttentionPool:
    def test_forward(self):
        pool = AttentionPool(dim=256, num_heads=8)
        x = torch.randn(2, 10, 256)
        out = pool(x)
        assert out.shape == (2, 256)


class TestPerceptionModule:
    @pytest.fixture
    def model(self):
        return PerceptionModule(
            backbone="efficientnet_b0",
            feature_dim=512,
            num_concepts=100,
        )
    
    def test_forward(self, model):
        x = torch.randn(2, 3, 224, 224)
        output = model.forward(x)
        
        assert "features" in output
        assert "concepts" in output
        assert "bboxes" in output
        assert "confidence" in output
        
        assert output["features"].shape == (2, 512)
        assert output["concepts"].shape == (2, 100)
        assert output["bboxes"].shape == (2, 4)
        assert output["confidence"].shape == (2, 1)
    
    def test_extract_symbolic_scene(self, model):
        x = torch.randn(2, 3, 224, 224)
        scene = model.extract_symbolic_scene(x, threshold=0.5)
        
        assert isinstance(scene, list)
        for obj in scene:
            assert "concepts" in obj
            assert "bbox" in obj
            assert "confidence" in obj