"""
Advanced Reasoning Demo for NeuroSymbolic-T4.

This script demonstrates advanced features of the symbolic engine:
1. Rule Induction: Learning new rules from examples.
2. Hierarchical Reasoning: Reasoning across multiple abstraction levels.
3. Temporal Reasoning: Reasoning with facts that persist or decay over time.
"""

import torch
from neurosymbolic.symbolic import SymbolicReasoner, Fact, Rule

def demo_rule_induction():
    print("\n--- Step 1: Rule Induction ---")
    reasoner = SymbolicReasoner(use_gnn=False)

    # Example 1: If something is a car and it's moving, it might be dangerous
    conclusion1 = Fact("dangerous", ("obj1",), confidence=0.9)
    premises1 = [
        Fact("car", ("obj1",), confidence=1.0),
        Fact("moving", ("obj1",), confidence=0.8)
    ]

    # Example 2: If something is a truck and it's moving, it might be dangerous
    conclusion2 = Fact("dangerous", ("obj2",), confidence=0.95)
    premises2 = [
        Fact("truck", ("obj2",), confidence=1.0),
        Fact("moving", ("obj2",), confidence=0.9)
    ]

    # Learn from examples
    print("Learning rules from examples...")
    reasoner.learn_rule_from_examples([
        (conclusion1, premises1),
        (conclusion2, premises2)
    ])

    print(f"Learned {len(reasoner.rules)} rules:")
    for rule in reasoner.rules:
        print(f"  {rule}")

def demo_hierarchical_reasoning():
    print("\n--- Step 2: Hierarchical Reasoning ---")
    reasoner = SymbolicReasoner(use_gnn=False)

    # Level 0 facts: Concrete detections
    reasoner.hierarchical.add_fact_to_level(Fact("sedan", ("car1",), 1.0), level=0)
    reasoner.hierarchical.add_fact_to_level(Fact("suv", ("car2",), 1.0), level=0)

    # Level 0 -> 1 abstraction: Specific cars are Vehicles
    car_rule = Rule(("vehicle", ("?x",)), [("sedan", ("?x",))], confidence=1.0)
    suv_rule = Rule(("vehicle", ("?x",)), [("suv", ("?x",))], confidence=1.0)
    reasoner.hierarchical.add_abstraction_rule(level=0, rule=car_rule)
    reasoner.hierarchical.add_abstraction_rule(level=0, rule=suv_rule)

    # Level 1 -> 2 abstraction: Vehicles are Transportation
    trans_rule = Rule(("transport", ("?x",)), [("vehicle", ("?x",))], confidence=0.9)
    reasoner.hierarchical.add_abstraction_rule(level=1, rule=trans_rule)

    print("Performing hierarchical abstraction...")
    # Answer a high-level query
    query = ("transport", ("?x",))
    proofs = reasoner.hierarchical.reason_hierarchically(query, max_level=2)

    print(f"Found {len(proofs)} proofs for query transport(?x):")
    for proof in proofs:
        print(f"  Binding: {proof['binding']}, Conf: {proof['confidence']:.3f}, Level: {proof['hierarchical_level']}")

def demo_temporal_reasoning():
    print("\n--- Step 3: Temporal Reasoning ---")
    reasoner = SymbolicReasoner(use_gnn=False)

    # Fact with high persistence (e.g., identity)
    reasoner.add_fact("is_car", ("obj1",), confidence=1.0, persistence=10.0)
    # Fact with standard persistence (e.g., transient state)
    reasoner.add_fact("is_turning", ("obj1",), confidence=1.0, persistence=1.0)

    print("Initial facts:")
    for fact in reasoner.facts:
        print(f"  {fact}")

    # Advance time steps
    for step in range(1, 11):
        reasoner.temporal_forward_chain()
        if step % 5 == 0:
            print(f"\nAfter {step} time steps:")
            for fact in reasoner.facts:
                print(f"  {fact}")

if __name__ == "__main__":
    print("=== NeuroSymbolic-T4 Advanced Reasoning Demo ===")
    demo_rule_induction()
    demo_hierarchical_reasoning()
    demo_temporal_reasoning()
    print("\nDemo completed successfully!")
