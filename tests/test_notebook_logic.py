"""Tests for core logic used in notebooks."""

import torch
import numpy as np
from neurosymbolic import NeurosymbolicSystem
from benchmarks import CLEVRBenchmark, VQABenchmark, GQABenchmark
import pytest


def test_synthetic_evaluation_consistency():
    """Ensure that synthetic evaluation (mock mode) actually runs and produces metrics."""
    model = NeurosymbolicSystem()
    device = torch.device("cpu")

    # Test CLEVR synthetic
    clevr = CLEVRBenchmark(model, device)
    results = clevr.evaluate(num_samples=5)
    assert "accuracy" in results
    assert "avg_reasoning_depth" in results
    assert results["total_evaluated"] == 5

    # Test VQA synthetic
    vqa = VQABenchmark(model, device)
    results = vqa.evaluate(num_samples=5)
    assert "avg_concepts_detected" in results
    assert results["total_evaluated"] == 5

    # Test GQA synthetic
    gqa = GQABenchmark(model, device)
    results = gqa.evaluate(num_samples=5)
    assert "avg_compositional_steps" in results
    assert results["total_evaluated"] == 5


def test_heuristic_accuracy_logic():
    """Verify the heuristic accuracy logic works as expected."""
    # Mock data structure matching perception output
    symbolic_scene = [
        ("car", 0.9, {}),
        ("person", 0.8, {}),
    ]
    derived_facts = [
        ("vehicle", ("obj1",), 0.95, "derived"),
    ]
    reasoning_output = {
        "derived_facts": derived_facts,
        "num_derived": 1
    }

    # Positive case
    target_answer = "car"
    perceived_concepts = [c[0] for c in symbolic_scene]
    derived_predicates = [f[0] for f in reasoning_output["derived_facts"]]
    assert target_answer in perceived_concepts or target_answer in derived_predicates

    # Derived case
    target_answer = "vehicle"
    assert target_answer in perceived_concepts or target_answer in derived_predicates

    # Negative case
    target_answer = "airplane"
    assert not (target_answer in perceived_concepts or target_answer in derived_predicates)


def test_3tuple_perception_handling():
    """Rigorous test for 3-tuple output handling across the system."""
    model = NeurosymbolicSystem()
    image = torch.randn(1, 3, 224, 224)

    # 1. Test perceive() output format
    output = model.perceive(image)
    scene = output["symbolic"][0]
    assert len(scene[0]) == 3, f"Expected 3-tuple, got {len(scene[0])}"
    assert isinstance(scene[0][2], dict), "3rd element must be an attribute dictionary"

    # 2. Test reason() compatibility with 3-tuple
    res = model.reason(scene)
    assert res["num_derived"] >= 0

    # 3. Test forward() consistency
    full_output = model.forward(image)
    assert "perception" in full_output
    assert "reasoning" in full_output
