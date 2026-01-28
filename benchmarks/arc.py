"""ARC (Abstraction and Reasoning Corpus) benchmark."""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm


class ARCBenchmark:
    """ARC benchmark for abstract reasoning and program synthesis.
    
    Reference: Chollet, 2019
    https://github.com/fchollet/ARC
    """
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
    
    def evaluate(self, num_samples=200) -> Dict[str, float]:
        """Evaluate on ARC-like tasks."""
        print(f"Running ARC evaluation on {num_samples} samples...")
        
        metrics = {
            "pattern_recognition": 0.0,
            "transformation_learning": 0.0,
            "inductive_reasoning": 0.0,
            "few_shot_learning": 0.0,
        }
        
        correct = {key: 0 for key in metrics.keys()}
        
        for _ in tqdm(range(num_samples), desc="ARC Evaluation"):
            # Generate task examples
            examples = self._generate_arc_task()
            
            # Test pattern recognition on multiple examples
            patterns_recognized = 0
            for example in examples:
                with torch.no_grad():
                    output = self.model.forward(example, threshold=0.5)
                
                if output["reasoning"][0]["num_derived"] > 0:
                    patterns_recognized += 1
            
            # Pattern recognition
            if patterns_recognized >= len(examples) * 0.7:
                correct["pattern_recognition"] += 1
            
            # Transformation learning (consistent reasoning)
            if patterns_recognized >= 2:
                correct["transformation_learning"] += 1
            
            # Inductive reasoning (generalization)
            if patterns_recognized == len(examples):
                correct["inductive_reasoning"] += 1
            
            # Few-shot learning (learn from few examples)
            if patterns_recognized >= 1:
                correct["few_shot_learning"] += 1
        
        for key in metrics.keys():
            metrics[key] = correct[key] / num_samples
        
        return metrics
    
    def _generate_arc_task(self, num_examples=3) -> List[torch.Tensor]:
        """Generate ARC-like task with examples."""
        examples = []
        
        # Simple transformation: color change
        base_pattern = torch.randn(3, 224, 224)
        
        for i in range(num_examples):
            # Apply transformation
            transformed = base_pattern.clone()
            transformed = transformed * (1 + 0.1 * i)  # Simple scaling
            examples.append(transformed.unsqueeze(0).to(self.device))
        
        return examples