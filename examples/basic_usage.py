"""Basic usage examples for NeuroSymbolic-T4."""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurosymbolic import NeurosymbolicSystem


def example_perception():
    """Example: Neural perception."""
    print("\n" + "="*50)
    print("Example 1: Neural Perception")
    print("="*50)
    
    # Initialize system
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeurosymbolicSystem().to(device)
    model.eval()
    
    # Create dummy image
    image = torch.randn(1, 3, 224, 224).to(device)
    
    # Perceive
    with torch.no_grad():
        output = model.perceive(image, threshold=0.6)
    
    print("\nDetected concepts:")
    for concept, confidence in output["symbolic"][0]:
        print(f"  - {concept}: {confidence:.3f}")


def example_reasoning():
    """Example: Symbolic reasoning."""
    print("\n" + "="*50)
    print("Example 2: Symbolic Reasoning")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeurosymbolicSystem().to(device)
    model.eval()
    
    # Create dummy image
    image = torch.randn(1, 3, 224, 224).to(device)
    
    # Full forward pass (perception + reasoning)
    with torch.no_grad():
        output = model.forward(image, threshold=0.6)
    
    reasoning = output["reasoning"][0]
    print(f"\nDerived {reasoning['num_derived']} new facts via reasoning")
    
    if reasoning["derived_facts"]:
        print("\nDerived facts:")
        for pred, args, conf in reasoning["derived_facts"][:5]:
            print(f"  - {pred}{args}: {conf:.3f}")


def example_query():
    """Example: Querying the system."""
    print("\n" + "="*50)
    print("Example 3: Query with Explanation")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeurosymbolicSystem().to(device)
    model.eval()
    
    # Create dummy image
    image = torch.randn(1, 3, 224, 224).to(device)
    
    # Query: Is there something dangerous?
    query = ("dangerous", ("obj0",))
    
    with torch.no_grad():
        proofs = model.query(image, query, threshold=0.5)
    
    print(f"\nQuery: {query[0]}{query[1]}")
    print(f"Found {len(proofs)} proofs")
    
    if proofs:
        print("\nTop proof:")
        proof = proofs[0]
        print(f"Confidence: {proof['confidence']:.3f}")
        print("Proof steps:")
        for step in proof["proof"]:
            print(f"  - {step}")


def example_explanation():
    """Example: Explaining predictions."""
    print("\n" + "="*50)
    print("Example 4: Explainable Predictions")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeurosymbolicSystem().to(device)
    model.eval()
    
    # Create dummy image
    image = torch.randn(1, 3, 224, 224).to(device)
    
    # Get prediction and explanation
    fact = ("vehicle", ("obj0",))
    
    with torch.no_grad():
        explanations = model.explain_prediction(image, fact, threshold=0.5)
    
    print(f"\nExplaining: {fact[0]}{fact[1]}")
    if explanations:
        print("\n" + explanations[0])
    else:
        print("No explanation found (fact may not hold)")


def example_custom_rules():
    """Example: Adding custom rules."""
    print("\n" + "="*50)
    print("Example 5: Custom Reasoning Rules")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeurosymbolicSystem().to(device)
    model.eval()
    
    # Add custom rule: If something is large and red, it's important
    model.reasoner.add_rule(
        head=("important", ("?x",)),
        body=[("large", ("?x",)), ("red", ("?x",))],
        confidence=0.9
    )
    
    # Manually add facts for demonstration
    model.reasoner.add_fact("large", ("obj1",), 0.9)
    model.reasoner.add_fact("red", ("obj1",), 0.85)
    
    # Forward chain
    num_derived = model.reasoner.forward_chain()
    print(f"\nDerived {num_derived} new facts")
    
    # Query
    result = model.reasoner.query("important", ("obj1",))
    if result:
        print(f"\nimportant(obj1): {result:.3f}")
        
        # Explain
        explanations = model.reasoner.explain(("important", ("obj1",)))
        if explanations:
            print("\n" + explanations[0])


if __name__ == "__main__":
    print("NeuroSymbolic-T4 Examples")
    print("========================\n")
    
    # Run all examples
    example_perception()
    example_reasoning()
    example_query()
    example_explanation()
    example_custom_rules()
    
    print("\n" + "="*50)
    print("All examples completed!")
    print("="*50)