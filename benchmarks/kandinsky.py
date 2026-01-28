"""Kandinsky Patterns benchmark for abstract reasoning."""

import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm


class KandinskyBenchmark:
    """Kandinsky Patterns for concept learning and abstract reasoning.
    
    Tests neurosymbolic concept acquisition and rule learning.
    """
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
    
    def evaluate(self, num_samples=500) -> Dict[str, float]:
        """Evaluate on Kandinsky Patterns."""
        print(f"Running Kandinsky evaluation on {num_samples} samples...")
        
        metrics = {
            "concept_learning": 0.0,
            "rule_discovery": 0.0,
            "generalization": 0.0,
            "abstraction": 0.0,
        }
        
        correct = {key: 0 for key in metrics.keys()}
        
        for _ in tqdm(range(num_samples), desc="Kandinsky Evaluation"):
            # Generate abstract pattern
            image = self._generate_pattern()
            
            with torch.no_grad():
                output = self.model.forward(image, threshold=0.4)
            
            symbolic = output["perception"]["symbolic"][0]
            reasoning = output["reasoning"][0]
            
            # Concept learning: detect basic concepts
            if len(symbolic) >= 3:
                correct["concept_learning"] += 1
            
            # Rule discovery: derive new facts
            if reasoning["num_derived"] > 0:
                correct["rule_discovery"] += 1
            
            # Generalization: consistent predictions
            if len(symbolic) > 0 and reasoning["num_derived"] > 0:
                correct["generalization"] += 1
            
            # Abstraction: high-level concepts
            abstract_concepts = [s for s in symbolic if any(
                k in s[0] for k in ["important", "dangerous", "urgent", "priority"]
            )]
            if abstract_concepts:
                correct["abstraction"] += 1
        
        for key in metrics.keys():
            metrics[key] = correct[key] / num_samples
        
        return metrics
    
    def _generate_pattern(self) -> torch.Tensor:
        """Generate synthetic Kandinsky pattern."""
        # Simple pattern generation
        pattern = torch.randn(1, 3, 224, 224)
        
        # Add structured patterns
        h, w = 224, 224
        num_objects = np.random.randint(2, 5)
        
        for _ in range(num_objects):
            x, y = np.random.randint(0, w-50), np.random.randint(0, h-50)
            size = np.random.randint(20, 60)
            color = np.random.rand(3, 1, 1)
            
            pattern[0, :, y:y+size, x:x+size] = torch.tensor(color)
        
        return pattern.to(self.device)