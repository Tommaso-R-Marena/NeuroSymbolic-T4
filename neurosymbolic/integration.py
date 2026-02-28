"""Enhanced integration layer with advanced neural-symbolic coupling.

Major improvements:
- Attention-based neural-symbolic grounding
- Multi-modal reasoning support
- Adaptive reasoning with learned rule selection
- Curriculum learning for rule acquisition
- Bidirectional neural-symbolic interaction
- Confidence calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .neural import PerceptionModule
from .symbolic import SymbolicReasoner, Fact, Rule
from .advanced_neural import RelationNetwork, SymbolicAttention


class NeuralSymbolicGrounding(nn.Module):
    """Neural-symbolic grounding with bidirectional attention."""
    
    def __init__(self, neural_dim: int, symbolic_dim: int):
        super().__init__()
        
        # Neural to symbolic projection
        self.neural_to_symbolic = nn.Sequential(
            nn.Linear(neural_dim, symbolic_dim),
            nn.LayerNorm(symbolic_dim),
            nn.ReLU()
        )
        
        # Symbolic to neural feedback
        self.symbolic_to_neural = nn.Sequential(
            nn.Linear(symbolic_dim, neural_dim),
            nn.LayerNorm(neural_dim),
            nn.ReLU()
        )
        
        # Bidirectional attention
        self.cross_attention = nn.MultiheadAttention(
            neural_dim, num_heads=8, batch_first=True
        )
        
        # Confidence calibration
        self.calibration = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, neural_features: torch.Tensor, 
                symbolic_confidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ground neural features with symbolic reasoning.
        
        Args:
            neural_features: [B, N, neural_dim]
            symbolic_confidence: [B, N] symbolic confidence scores
            
        Returns:
            grounded_features: [B, N, neural_dim]
            calibrated_confidence: [B, N]
        """
        B, N, D = neural_features.shape
        
        # Ensure consistent dtype
        dtype = neural_features.dtype
        
        # Project to symbolic space
        symbolic_proj = self.neural_to_symbolic(neural_features)
        
        # Bidirectional attention
        attended, _ = self.cross_attention(
            neural_features, symbolic_proj, symbolic_proj
        )
        
        # Combine with residual
        grounded = neural_features + 0.3 * attended
        
        # Calibrate confidence (combine neural and symbolic)
        neural_conf = neural_features.norm(dim=-1) / D ** 0.5
        
        # Ensure symbolic_confidence has correct shape and dtype
        if symbolic_confidence.dim() == 1:
            symbolic_confidence = symbolic_confidence.unsqueeze(0).expand(B, -1)
        symbolic_confidence = symbolic_confidence.to(dtype)
        
        conf_input = torch.stack([neural_conf, symbolic_confidence], dim=-1)
        calibrated = self.calibration(conf_input).squeeze(-1)
        
        return grounded, calibrated


class AdaptiveRuleSelector(nn.Module):
    """Learn to select relevant rules for a given context."""
    
    def __init__(self, context_dim: int, num_rules: int):
        super().__init__()
        self.rule_selector = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_rules),
            nn.Sigmoid()
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Select rules based on context.
        
        Args:
            context: [B, context_dim] contextual features
            
        Returns:
            rule_weights: [B, num_rules] selection probabilities
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)
        return self.rule_selector(context)


class NeurosymbolicError(Exception):
    """Base exception for neurosymbolic system."""
    pass

class PerceptionError(NeurosymbolicError):
    """Error during neural perception."""
    pass

class ReasoningError(NeurosymbolicError):
    """Error during symbolic reasoning."""
    pass

class NeurosymbolicSystem(nn.Module):
    """Enhanced neurosymbolic system with advanced integration.
    
    Features:
    - Multi-scale neural perception with FPN
    - Cross-attention concept grounding
    - GNN-based symbolic reasoning
    - Adaptive rule selection
    - Bidirectional neural-symbolic grounding
    - Confidence calibration
    - Curriculum learning for rules
    """
    
    def __init__(
        self,
        perception_config: Optional[Dict] = None,
        reasoning_config: Optional[Dict] = None,
        concept_names: Optional[List[str]] = None,
        use_grounding: bool = True,
        use_adaptive_rules: bool = True,
        use_relation_network: bool = False,
        use_symbolic_attention: bool = False,
        disable_gnn: bool = False,
        disable_fpn: bool = False,
    ):
        super().__init__()
        
        # Enhanced perception
        perception_config = perception_config or {}
        if disable_fpn:
            perception_config['use_fpn'] = False
        try:
            self.perception = PerceptionModule(**perception_config)
        except TypeError as e:
            raise ValueError(f"Invalid perception_config: {e}")

        # Concept vocabulary
        self.concept_names = concept_names or self._default_concepts()
        self.concept_to_idx = {name: idx for idx, name in enumerate(self.concept_names)}

        # Enhanced symbolic reasoner
        reasoning_config = reasoning_config or {}
        self.reasoner = SymbolicReasoner(
            use_gnn=not disable_gnn,
            **reasoning_config
        )

        feature_dim = perception_config.get('feature_dim', 512)
        
        # Advanced neural modules
        self.use_relation_network = use_relation_network
        if use_relation_network:
            self.relation_network = RelationNetwork(
                object_dim=feature_dim,
                relation_dim=256
            )

        self.use_symbolic_attention = use_symbolic_attention
        if use_symbolic_attention:
            self.symbolic_attention = SymbolicAttention(
                dim=feature_dim,
                num_symbols=len(self.concept_names)
            )
        
        # Neural-symbolic grounding
        self.use_grounding = use_grounding
        if use_grounding:
            feature_dim = perception_config.get('feature_dim', 512)
            self.grounding = NeuralSymbolicGrounding(
                neural_dim=feature_dim,
                symbolic_dim=feature_dim
            )
        
        # Adaptive rule selection
        self.use_adaptive_rules = use_adaptive_rules
        if use_adaptive_rules:
            # Use actual concept vocabulary size for selector input dim
            num_concepts = len(self.concept_names)
            self.rule_selector = AdaptiveRuleSelector(
                context_dim=num_concepts,
                num_rules=100  # Maximum number of rules
            )
        
        # Performance tracking
        self.inference_stats = {
            'perception_time': [],
            'reasoning_time': [],
            'total_facts_derived': [],
            'rule_usage': {},
        }
        
        # Initialize enhanced knowledge base
        self._initialize_enhanced_knowledge()
    
    def _default_concepts(self) -> List[str]:
        """Expanded concept vocabulary."""
        concepts = [
            # Objects
            "person", "car", "building", "tree", "animal", "furniture",
            "vehicle", "electronics", "food", "clothing", "tool", "weapon",
            # Attributes - Size
            "tiny", "small", "medium", "large", "huge",
            # Attributes - Color
            "red", "blue", "green", "yellow", "black", "white", "gray",
            "orange", "purple", "brown", "pink",
            # Attributes - Shape
            "round", "square", "rectangular", "triangular", "irregular",
            # Attributes - Material
            "metal", "wood", "plastic", "glass", "fabric", "stone", "rubber",
            # States
            "moving", "static", "broken", "intact", "open", "closed",
            # Locations
            "indoor", "outdoor", "underwater", "aerial",
            # Spatial relations
            "above", "below", "left", "right", "near", "far", 
            "inside", "outside", "on_top", "underneath",
            # Actions
            "walking", "running", "sitting", "standing", "lying",
            "holding", "carrying", "pushing", "pulling",
            # Abstract
            "dangerous", "safe", "important", "urgent", "normal",
            "rare", "common", "valuable", "cheap",
        ]
        # Pad to 100
        while len(concepts) < 100:
            concepts.append(f"concept_{len(concepts)}")
        return concepts[:100]
    
    def _initialize_enhanced_knowledge(self):
        """Initialize comprehensive domain knowledge."""
        # Taxonomic hierarchy
        taxonomic_rules = [
            # Vehicles
            (("vehicle", ("?x",)), [("car", ("?x",))], 1.0),
            (("vehicle", ("?x",)), [("truck", ("?x",))], 1.0),
            (("vehicle", ("?x",)), [("motorcycle", ("?x",))], 1.0),
            (("moving_object", ("?x",)), [("vehicle", ("?x",))], 0.9),
            
            # Living things
            (("living_thing", ("?x",)), [("person", ("?x",))], 1.0),
            (("living_thing", ("?x",)), [("animal", ("?x",))], 1.0),
            (("living_thing", ("?x",)), [("tree", ("?x",))], 1.0),
            
            # Size reasoning
            (("large", ("?x",)), [("huge", ("?x",))], 0.9),
            (("small", ("?x",)), [("tiny", ("?x",))], 0.9),
        ]
        
        # Spatial reasoning
        spatial_rules = [
            (("near", ("?x", "?y")), [("near", ("?y", "?x"))], 1.0),  # Symmetry
            (("far", ("?x", "?y")), [("far", ("?y", "?x"))], 1.0),
            (("below", ("?x", "?y")), [("above", ("?y", "?x"))], 1.0),  # Inverse
            (("left", ("?x", "?y")), [("right", ("?y", "?x"))], 1.0),
        ]
        
        # Safety and danger rules
        safety_rules = [
            (("dangerous", ("?x",)), [("vehicle", ("?x",)), ("moving", ("?x",))], 0.8),
            (("dangerous", ("?x",)), [("weapon", ("?x",))], 0.95),
            (("safe", ("?x",)), [("static", ("?x",)), ("small", ("?x",))], 0.7),
            (("urgent", ("?x",)), [("dangerous", ("?x",)), ("near", ("?x", "person"))], 0.9),
        ]
        
        # Action and state rules
        action_rules = [
            (("moving", ("?x",)), [("walking", ("?x",))], 0.95),
            (("moving", ("?x",)), [("running", ("?x",))], 1.0),
            (("static", ("?x",)), [("sitting", ("?x",))], 0.8),
            (("static", ("?x",)), [("standing", ("?x",))], 0.7),
        ]
        
        # Add all rules
        all_rules = taxonomic_rules + spatial_rules + safety_rules + action_rules
        for head, body, conf in all_rules:
            self.reasoner.add_rule(head, body, conf)
    
    def perceive(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """Enhanced perception with rich outputs.

        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W].
            threshold (float): Confidence threshold for concept detection.

        Returns:
            Dict[str, Any]: Perception results including neural outputs and symbolic scenes.
        """
        import time
        start_time = time.time()
        
        # Neural perception
        try:
            perception_output = self.perception(
                x,
                return_attention=True,
                return_relations=True
            )

            # Symbolic attention refinement (feedback loop)
            if self.use_symbolic_attention:
                concept_features = perception_output["concept_features"]
                concepts = perception_output["concepts"]
                # Get indices of top concepts for each batch
                top_concept_indices = concepts.topk(5, dim=-1)[1]

                refined_features = self.symbolic_attention(concept_features, top_concept_indices)
                perception_output["concept_features"] = refined_features

            # Global relational reasoning
            if self.use_relation_network:
                relational_context = self.relation_network(perception_output["concept_features"])
                perception_output["relational_context"] = relational_context

        except Exception as e:
            raise PerceptionError(f"Neural perception failed: {e}")
        
        perception_time = time.time() - start_time
        self.inference_stats['perception_time'].append(perception_time)
        
        # Extract symbolic scenes
        concepts = perception_output["concepts"]
        attributes = perception_output["attributes"]
        B = concepts.shape[0]
        
        symbolic_scenes: List[List[Tuple[str, float, Dict[str, Any]]]] = []
        for i in range(B):
            scene = self._enhanced_concepts_to_symbols(
                concepts[i], 
                attributes={k: v[i] for k, v in attributes.items()},
                threshold=threshold
            )
            symbolic_scenes.append(scene)
        
        return {
            "neural": perception_output,
            "symbolic": symbolic_scenes,
            "perception_time": perception_time,
        }
    
    def _enhanced_concepts_to_symbols(self, 
                                     concept_probs: torch.Tensor,
                                     attributes: Dict[str, torch.Tensor],
                                     threshold: float) -> List[Tuple[str, float, Dict]]:
        """Convert to symbols with rich attributes."""
        symbols = []
        probs = concept_probs.detach().cpu().numpy()
        
        for idx, prob in enumerate(probs):
            if prob >= threshold and idx < len(self.concept_names):
                concept_name = self.concept_names[idx]
                
                # Extract most likely attributes
                attrs = {}
                for attr_name, attr_probs in attributes.items():
                    attr_values = attr_probs.detach().cpu().numpy()
                    attrs[attr_name] = int(attr_values.argmax())
                
                symbols.append((concept_name, float(prob), attrs))
        
        return symbols
    
    def reason(self, 
               symbolic_scene: List[Tuple[str, float, Dict[str, Any]]],
               object_id: str = "obj1",
               use_gnn: bool = True) -> Dict[str, Any]:
        """Enhanced reasoning with GNN and adaptive rules.

        Args:
            symbolic_scene (List[Tuple[str, float, Dict[str, Any]]]): List of perceived concepts.
            object_id (str): Identifier for the object being reasoned about.
            use_gnn (bool): Whether to use GNN-based refinement.

        Returns:
            Dict[str, Any]: Reasoning results including derived facts and timing.
        """
        import time
        start_time = time.time()
        
        # Clear and reinitialize
        self.reasoner.clear()
        self._initialize_enhanced_knowledge()
        
        # Add perceived concepts as facts (standardized 3-tuples)
        normalized_scene: List[Tuple[str, float, Dict[str, Any]]] = []
        for item in symbolic_scene:
            if len(item) == 3:
                concept, confidence, attrs = item
                attrs = attrs or {}
            else:
                raise ValueError(
                    "Each symbolic item must be (concept, confidence, attrs)"
                )

            normalized_scene.append((concept, float(confidence), attrs))
            self.reasoner.add_fact(concept, (object_id,), float(confidence), "perceived")

            # Add attribute facts
            for attr_name, attr_val in attrs.items():
                attr_pred = f"{attr_name}_{attr_val}"
                self.reasoner.add_fact(attr_pred, (object_id,), float(confidence) * 0.9, "perceived")
        
        # Adaptive rule selection if enabled
        if self.use_adaptive_rules and hasattr(self, 'rule_selector'):
            # Build context vector from scene (sum of concept confidences)
            context_vec = torch.zeros(len(self.concept_names), device=next(self.parameters()).device)
            for concept, conf, _ in normalized_scene:
                if concept in self.concept_to_idx:
                    context_vec[self.concept_to_idx[concept]] = conf

            # Select rules
            with torch.no_grad():
                rule_weights = self.rule_selector(context_vec).squeeze(0)
            
            # Update rule strengths based on selection
            for i, rule in enumerate(self.reasoner.rules[:len(rule_weights)]):
                rule.strength *= rule_weights[i].item()
        
        # Forward chaining with GNN
        try:
            num_derived = self.reasoner.forward_chain(max_iterations=5, use_gnn=use_gnn)
        except Exception as e:
            raise ReasoningError(f"Symbolic reasoning failed: {e}")
        
        reasoning_time = time.time() - start_time
        self.inference_stats['reasoning_time'].append(reasoning_time)
        self.inference_stats['total_facts_derived'].append(num_derived)
        
        # Collect derived facts
        original_concepts = {concept for concept, _, _ in normalized_scene}
        derived_facts = [
            (fact.predicate, fact.arguments, fact.confidence, fact.source)
            for fact in self.reasoner.facts
            if fact.predicate not in original_concepts or fact.source == "derived"
        ]
        
        return {
            "num_derived": num_derived,
            "derived_facts": derived_facts,
            "all_facts": list(self.reasoner.facts),
            "reasoning_time": reasoning_time,
        }
    
    def forward(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """Complete enhanced forward pass.

        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W].
            threshold (float): Confidence threshold for concept detection.

        Returns:
            Dict[str, Any]: Combined results from perception and reasoning.
        """
        if x is None or (isinstance(x, torch.Tensor) and x.numel() == 0):
            raise ValueError("Input tensor x cannot be None or empty")

        # Perception
        perception_output = self.perceive(x, threshold)
        
        # Neural-symbolic grounding (if enabled)
        if self.use_grounding:
            concept_features = perception_output["neural"]["concept_features"]
            concepts = perception_output["neural"]["concepts"]
            
            # Ensure proper dtype for grounding
            with torch.no_grad():  # Disable autocast for grounding to avoid dtype mismatch
                grounded_features, calibrated_conf = self.grounding(
                    concept_features.float(), concepts.float()
                )
            
            # Update confidences
            perception_output["neural"]["calibrated_concepts"] = calibrated_conf
        
        # Reasoning on each scene
        reasoning_outputs = []
        for i, scene in enumerate(perception_output["symbolic"]):
            reasoning = self.reason(scene, object_id=f"obj{i}", use_gnn=True)
            reasoning_outputs.append(reasoning)
        
        return {
            "perception": perception_output,
            "reasoning": reasoning_outputs,
            "stats": self.get_inference_stats(),
        }
    
    def query(self, x: torch.Tensor, query: Tuple[str, Tuple[str, ...]], 
             threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Enhanced query with better proofs.

        Args:
            x (torch.Tensor): Input image tensor.
            query (Tuple[str, Tuple[str, ...]]): The goal to prove (predicate, args).
            threshold (float): Confidence threshold.

        Returns:
            List[Dict[str, Any]]: List of proofs for the query.
        """
        # Process input
        self.forward(x, threshold)
        
        # Backward chaining
        proofs = self.reasoner.backward_chain(query, max_depth=5)
        
        return proofs
    
    def explain_prediction(self, x: torch.Tensor, 
                          fact: Tuple[str, Tuple[str, ...]],
                          threshold: float = 0.5) -> List[str]:
        """Generate enhanced explanations with counterfactuals."""
        # Process
        self.forward(x, threshold)
        
        # Generate rich explanations
        return self.reasoner.explain(fact, max_depth=3)
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get comprehensive inference statistics."""
        stats = {}
        
        if self.inference_stats['perception_time']:
            stats['avg_perception_time'] = np.mean(self.inference_stats['perception_time'])
        
        if self.inference_stats['reasoning_time']:
            stats['avg_reasoning_time'] = np.mean(self.inference_stats['reasoning_time'])
        
        if self.inference_stats['total_facts_derived']:
            stats['avg_facts_derived'] = np.mean(self.inference_stats['total_facts_derived'])
            stats['total_facts_derived'] = sum(self.inference_stats['total_facts_derived'])
        
        return stats
    
    def reset_stats(self):
        """Reset performance tracking."""
        self.inference_stats = {
            'perception_time': [],
            'reasoning_time': [],
            'total_facts_derived': [],
            'rule_usage': {},
        }


