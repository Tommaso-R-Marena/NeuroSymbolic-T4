"""Integration layer combining neural perception and symbolic reasoning."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .neural import PerceptionModule
from .symbolic import SymbolicReasoner, Fact, Rule


class NeurosymbolicSystem(nn.Module):
    """Complete neurosymbolic AI system.
    
    Integrates neural perception with symbolic reasoning for:
    - Explainable predictions
    - Logical consistency
    - Abstract reasoning
    - Knowledge-guided learning
    """
    
    def __init__(
        self,
        perception_config: Optional[Dict] = None,
        reasoning_config: Optional[Dict] = None,
        concept_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Initialize perception module
        perception_config = perception_config or {}
        self.perception = PerceptionModule(**perception_config)
        
        # Initialize symbolic reasoner
        reasoning_config = reasoning_config or {}
        self.reasoner = SymbolicReasoner(**reasoning_config)
        
        # Concept vocabulary
        self.concept_names = concept_names or self._default_concepts()
        self.concept_to_idx = {name: idx for idx, name in enumerate(self.concept_names)}
        
        # Initialize domain knowledge
        self._initialize_knowledge()
        
    def _default_concepts(self) -> List[str]:
        """Default concept vocabulary."""
        return [
            # Objects
            "person", "car", "building", "tree", "animal", "furniture",
            "vehicle", "electronics", "food", "clothing",
            # Attributes
            "red", "blue", "green", "large", "small", "round", "square",
            "moving", "static", "indoor", "outdoor",
            # Spatial relations
            "above", "below", "left", "right", "near", "far", "inside", "outside",
            # Actions
            "walking", "sitting", "standing", "running", "holding",
            # Abstract
            "dangerous", "safe", "important", "urgent", "normal",
        ] + [f"concept_{i}" for i in range(50)]  # Pad to 100
    
    def _initialize_knowledge(self):
        """Initialize domain knowledge and rules."""
        # Taxonomic rules
        self.reasoner.add_rule(
            head=("vehicle", ("?x",)),
            body=[("car", ("?x",))],
            confidence=1.0
        )
        
        self.reasoner.add_rule(
            head=("living_thing", ("?x",)),
            body=[("person", ("?x",))],
            confidence=1.0
        )
        
        self.reasoner.add_rule(
            head=("living_thing", ("?x",)),
            body=[("animal", ("?x",))],
            confidence=1.0
        )
        
        # Spatial reasoning rules
        self.reasoner.add_rule(
            head=("near", ("?x", "?y")),
            body=[("near", ("?y", "?x"))],
            confidence=1.0
        )
        
        # Safety rules
        self.reasoner.add_rule(
            head=("dangerous", ("?x",)),
            body=[("vehicle", ("?x",)), ("moving", ("?x",))],
            confidence=0.8
        )
        
        # Action rules
        self.reasoner.add_rule(
            head=("moving", ("?x",)),
            body=[("walking", ("?x",))],
            confidence=0.9
        )
        
        self.reasoner.add_rule(
            head=("moving", ("?x",)),
            body=[("running", ("?x",))],
            confidence=1.0
        )
    
    def perceive(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """Extract symbolic representation from sensory input.
        
        Args:
            x: Input tensor (e.g., images)
            threshold: Confidence threshold for concept detection
            
        Returns:
            Dictionary with neural features and symbolic scene
        """
        # Neural perception
        with torch.cuda.amp.autocast(enabled=True):
            perception_output = self.perception(x)
        
        # Extract symbolic scene
        concepts = perception_output["concepts"]
        B = concepts.shape[0]
        
        symbolic_scenes = []
        for i in range(B):
            scene = self._concepts_to_symbols(concepts[i], threshold)
            symbolic_scenes.append(scene)
        
        return {
            "neural": perception_output,
            "symbolic": symbolic_scenes,
        }
    
    def _concepts_to_symbols(self, concept_probs: torch.Tensor, threshold: float) -> List[Tuple[str, float]]:
        """Convert concept probabilities to symbolic predicates."""
        symbols = []
        probs = concept_probs.detach().cpu().numpy()
        
        for idx, prob in enumerate(probs):
            if prob >= threshold and idx < len(self.concept_names):
                concept_name = self.concept_names[idx]
                symbols.append((concept_name, float(prob)))
        
        return symbols
    
    def reason(self, symbolic_scene: List[Tuple[str, float]], 
               object_id: str = "obj1") -> Dict[str, Any]:
        """Perform symbolic reasoning on perceived scene.
        
        Args:
            symbolic_scene: List of (concept, confidence) tuples
            object_id: Identifier for the object
            
        Returns:
            Reasoning results with derived facts and explanations
        """
        # Clear previous facts
        self.reasoner.clear()
        self._initialize_knowledge()  # Re-add rules
        
        # Add perceived concepts as facts
        for concept, confidence in symbolic_scene:
            self.reasoner.add_fact(concept, (object_id,), confidence)
        
        # Forward chaining to derive new facts
        num_derived = self.reasoner.forward_chain(max_iterations=5)
        
        # Collect derived facts
        derived_facts = [
            (fact.predicate, fact.arguments, fact.confidence)
            for fact in self.reasoner.facts
            if not any(concept == fact.predicate and (object_id,) == fact.arguments 
                      for concept, _ in symbolic_scene)
        ]
        
        return {
            "num_derived": num_derived,
            "derived_facts": derived_facts,
            "all_facts": list(self.reasoner.facts),
        }
    
    def forward(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """Complete forward pass: perception + reasoning.
        
        Args:
            x: Input tensor
            threshold: Confidence threshold
            
        Returns:
            Complete neurosymbolic output
        """
        # Perception
        perception_output = self.perceive(x, threshold)
        
        # Reasoning on each scene
        reasoning_outputs = []
        for i, scene in enumerate(perception_output["symbolic"]):
            reasoning = self.reason(scene, object_id=f"obj{i}")
            reasoning_outputs.append(reasoning)
        
        return {
            "perception": perception_output,
            "reasoning": reasoning_outputs,
        }
    
    def query(self, x: torch.Tensor, query: Tuple[str, Tuple[str, ...]], 
             threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Query the system with a logical goal.
        
        Args:
            x: Input tensor
            query: Query in form (predicate, arguments)
            threshold: Confidence threshold
            
        Returns:
            Proofs and explanations
        """
        # Perceive and reason
        self.forward(x, threshold)
        
        # Backward chaining to answer query
        proofs = self.reasoner.backward_chain(query, max_depth=5)
        
        return proofs
    
    def explain_prediction(self, x: torch.Tensor, 
                          fact: Tuple[str, Tuple[str, ...]],
                          threshold: float = 0.5) -> List[str]:
        """Explain why a fact holds for given input.
        
        Args:
            x: Input tensor
            fact: Fact to explain
            threshold: Confidence threshold
            
        Returns:
            List of natural language explanations
        """
        # Process input
        self.forward(x, threshold)
        
        # Generate explanations
        return self.reasoner.explain(fact, max_depth=3)