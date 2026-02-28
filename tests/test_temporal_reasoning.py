import pytest
from neurosymbolic.symbolic import SymbolicReasoner

def test_enhanced_temporal_reasoning():
    reasoner = SymbolicReasoner(use_gnn=False, confidence_threshold=0.1)

    # Standard persistence
    reasoner.add_fact("moving", ("obj1",), 1.0, persistence=1.0)
    # High persistence
    reasoner.add_fact("is_apple", ("obj2",), 1.0, persistence=10.0)

    # Step 10 times
    for _ in range(10):
        reasoner.temporal_forward_chain()

    # obj1 should have decayed more than obj2
    fact1 = [f for f in reasoner.facts if f.arguments == ("obj1",)][0]
    fact2 = [f for f in reasoner.facts if f.arguments == ("obj2",)][0]

    assert fact2.confidence > fact1.confidence
    assert fact2.confidence > 0.8  # Should remain high due to persistence

def test_temporal_cleanup():
    reasoner = SymbolicReasoner(use_gnn=False, confidence_threshold=0.5)

    # Low confidence fact that will decay below threshold quickly
    reasoner.add_fact("glitch", ("obj1",), 0.6, persistence=0.5)

    for _ in range(10):
        reasoner.temporal_forward_chain()

    # Should have been cleaned up
    assert len([f for f in reasoner.facts if f.predicate == "glitch"]) == 0
