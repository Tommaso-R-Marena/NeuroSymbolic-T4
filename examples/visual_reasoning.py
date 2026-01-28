"""Visual reasoning example with scene understanding."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurosymbolic import NeurosymbolicSystem


def visual_reasoning_demo():
    """Demonstrate visual reasoning capabilities."""
    print("Visual Reasoning Demo")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Initialize system
    model = NeurosymbolicSystem().to(device)
    model.eval()
    
    # Add scene understanding rules
    print("Adding scene understanding rules...")
    
    # Rule: Indoor scenes typically have furniture
    model.reasoner.add_rule(
        head=("indoor_scene", ("?s",)),
        body=[("furniture", ("?s",))],
        confidence=0.8
    )
    
    # Rule: Outdoor scenes typically have trees or buildings
    model.reasoner.add_rule(
        head=("outdoor_scene", ("?s",)),
        body=[("tree", ("?s",))],
        confidence=0.7
    )
    
    model.reasoner.add_rule(
        head=("outdoor_scene", ("?s",)),
        body=[("building", ("?s",))],
        confidence=0.6
    )
    
    # Rule: Urban scene has vehicles and buildings
    model.reasoner.add_rule(
        head=("urban_scene", ("?s",)),
        body=[("vehicle", ("?s",)), ("building", ("?s",))],
        confidence=0.9
    )
    
    # Simulate multiple scenes
    print("\nAnalyzing scenes...\n")
    
    for scene_id in range(3):
        print(f"Scene {scene_id + 1}:")
        print("-" * 40)
        
        # Generate random scene
        image = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model.forward(image, threshold=0.55)
        
        # Display detected concepts
        symbolic = output["perception"]["symbolic"][0]
        if symbolic:
            print("Detected concepts:")
            for concept, conf in sorted(symbolic, key=lambda x: x[1], reverse=True)[:8]:
                print(f"  {concept:15s}: {conf:.3f}")
        
        # Display derived scene properties
        reasoning = output["reasoning"][0]
        if reasoning["derived_facts"]:
            print("\nDerived scene properties:")
            for pred, args, conf in reasoning["derived_facts"][:5]:
                if "scene" in pred:
                    print(f"  {pred:15s}: {conf:.3f}")
        
        print()


if __name__ == "__main__":
    visual_reasoning_demo()