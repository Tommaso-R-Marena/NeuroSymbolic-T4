"""Tests for symbolic reasoning engine."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurosymbolic.symbolic import SymbolicReasoner, Fact, Rule


class TestFact:
    def test_creation(self):
        fact = Fact("car", ("obj1",), 0.9)
        assert fact.predicate == "car"
        assert fact.arguments == ("obj1",)
        assert fact.confidence == 0.9
    
    def test_equality(self):
        fact1 = Fact("car", ("obj1",), 0.9)
        fact2 = Fact("car", ("obj1",), 0.8)
        assert fact1 == fact2  # Confidence doesn't affect equality


class TestRule:
    def test_creation(self):
        rule = Rule(
            head=("vehicle", ("?x",)),
            body=[("car", ("?x",))],
            confidence=1.0
        )
        assert rule.head == ("vehicle", ("?x",))
        assert len(rule.body) == 1


class TestSymbolicReasoner:
    @pytest.fixture
    def reasoner(self):
        return SymbolicReasoner(confidence_threshold=0.3)
    
    def test_add_fact(self, reasoner):
        reasoner.add_fact("car", ("obj1",), 0.9)
        assert len(reasoner.facts) == 1
    
    def test_query(self, reasoner):
        reasoner.add_fact("car", ("obj1",), 0.9)
        result = reasoner.query("car", ("obj1",))
        assert result == 0.9
    
    def test_forward_chain(self, reasoner):
        # Add facts
        reasoner.add_fact("car", ("obj1",), 0.9)
        
        # Add rule: vehicle(X) :- car(X)
        reasoner.add_rule(
            head=("vehicle", ("?x",)),
            body=[("car", ("?x",))],
            confidence=1.0
        )
        
        # Forward chain
        num_derived = reasoner.forward_chain()
        assert num_derived >= 1
        
        # Check derived fact
        result = reasoner.query("vehicle", ("obj1",))
        assert result is not None
        assert result >= 0.3  # Above threshold
    
    def test_backward_chain(self, reasoner):
        reasoner.add_fact("car", ("obj1",), 0.9)
        reasoner.add_rule(
            head=("vehicle", ("?x",)),
            body=[("car", ("?x",))],
            confidence=1.0
        )
        
        proofs = reasoner.backward_chain(("vehicle", ("obj1",)))
        assert len(proofs) > 0
        assert proofs[0]["confidence"] >= 0.3
    
    def test_explain(self, reasoner):
        reasoner.add_fact("car", ("obj1",), 0.9)
        reasoner.add_rule(
            head=("vehicle", ("?x",)),
            body=[("car", ("?x",))],
            confidence=1.0
        )
        
        explanations = reasoner.explain(("vehicle", ("obj1",)))
        assert len(explanations) > 0
        assert isinstance(explanations[0], str)