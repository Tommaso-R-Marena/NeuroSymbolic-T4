import pytest
from neurosymbolic.symbolic import SymbolicReasoner, Fact

def test_rule_induction_generalization():
    reasoner = SymbolicReasoner(use_gnn=False)

    # Example: A red apple is fruit
    # Fact(predicate, arguments, confidence, source)
    conclusion = Fact("fruit", ("apple1",), 1.0)
    premises = [
        Fact("red", ("apple1",), 1.0),
        Fact("apple", ("apple1",), 1.0)
    ]

    reasoner.learn_rule_from_examples([(conclusion, premises)])

    # Should have learned: fruit(?v0) :- red(?v0), apple(?v0)
    assert len(reasoner.rules) > 0
    learned_rule = reasoner.rules[-1]
    assert learned_rule.head[0] == "fruit"
    assert len(learned_rule.body) == 2

    # Check if variables are used correctly
    head_var = learned_rule.head[1][0]
    assert head_var.startswith("?")
    for pred, args in learned_rule.body:
        assert args[0] == head_var

def test_rule_induction_refinement():
    reasoner = SymbolicReasoner(use_gnn=False)

    conclusion = Fact("fruit", ("apple1",), 1.0)
    premises = [
        Fact("red", ("apple1",), 1.0),
        Fact("apple", ("apple1",), 1.0)
    ]

    # Learn it once
    reasoner.learn_rule_from_examples([(conclusion, premises)])
    initial_rules_count = len(reasoner.rules)

    # Learn same rule again with another example
    conclusion2 = Fact("fruit", ("apple2",), 1.0)
    premises2 = [
        Fact("red", ("apple2",), 1.0),
        Fact("apple", ("apple2",), 1.0)
    ]
    reasoner.learn_rule_from_examples([(conclusion2, premises2)])

    # Should not have added a new rule, but updated existing
    assert len(reasoner.rules) == initial_rules_count
    assert reasoner.rules[0].usage_count > 0
