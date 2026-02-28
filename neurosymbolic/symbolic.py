"""Enhanced symbolic reasoning with graph neural networks and rule learning.

Major improvements:
- Graph Neural Network for relational reasoning
- Hierarchical reasoning with abstraction levels
- Rule learning from examples
- Probabilistic logic with fuzzy operations
- Explanation generation with counterfactuals
- Temporal reasoning support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import re
import numpy as np


@dataclass
class Fact:
    """Atomic fact with metadata."""
    predicate: str
    arguments: Tuple[str, ...]
    confidence: float = 1.0
    timestamp: Optional[int] = None
    source: str = "perceived"  # perceived, derived, learned
    persistence: float = 1.0  # How long this fact persists (1.0 = standard)
    
    def __str__(self) -> str:
        args = ", ".join(self.arguments)
        return f"{self.predicate}({args}) [{self.confidence:.3f}]"
    
    def __hash__(self) -> int:
        """Hash based on predicate and arguments only.

        This allows Fact objects to be stored in sets while their confidence
        scores are refined in-place during reasoning without corrupting the
        set's internal structure.
        """
        return hash((self.predicate, self.arguments))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Fact):
            return NotImplemented
        return (
            self.predicate == other.predicate
            and self.arguments == other.arguments
        )


@dataclass
class Rule:
    """Logical rule with learning capability."""
    head: Tuple[str, Tuple[str, ...]]
    body: List[Tuple[str, Tuple[str, ...]]]
    confidence: float = 1.0
    strength: float = 1.0  # Learned rule strength
    usage_count: int = 0
    success_count: int = 0
    
    def __str__(self) -> str:
        head_str = f"{self.head[0]}({', '.join(self.head[1])})"
        body_str = " ∧ ".join(
            f"{pred}({', '.join(args)})" for pred, args in self.body
        )
        return f"{head_str} :- {body_str} [conf:{self.confidence:.2f}, str:{self.strength:.2f}]"
    
    def update_statistics(self, success: bool):
        """Update rule learning statistics."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.strength = self.success_count / max(self.usage_count, 1)


class GraphNeuralReasoner(nn.Module):
    """Graph Neural Network for relational reasoning over facts."""
    
    def __init__(self, feature_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Node feature encoder (for facts)
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.LayerNorm(feature_dim)
            ) for _ in range(num_layers)
        ])
        
        # Aggregation layers
        self.aggregation_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.LayerNorm(feature_dim)
            ) for _ in range(num_layers)
        ])
        
        # Output layer for confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN.
        
        Args:
            node_features: [N, feature_dim] node features
            edge_index: [2, E] edge indices
            
        Returns:
            updated_features: [N, feature_dim]
        """
        x = self.node_encoder(node_features)
        
        for message_layer, agg_layer in zip(self.message_layers, self.aggregation_layers):
            # Gather source and target node features
            src, dst = edge_index
            src_features = x[src]
            dst_features = x[dst]
            
            # Compute messages
            messages = message_layer(torch.cat([src_features, dst_features], dim=-1))
            
            # Aggregate messages (scatter sum)
            # Optimized with index_add_ for T4 performance
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, dst, messages)
            
            # Update node features
            x = agg_layer(torch.cat([x, aggregated], dim=-1))
        
        return x
    
    def predict_confidence(self, node_features: torch.Tensor) -> torch.Tensor:
        """Predict confidence scores for facts."""
        return self.confidence_head(node_features)


class HierarchicalReasoner:
    """Hierarchical reasoning with multiple abstraction levels."""
    
    def __init__(self, parent_reasoner: 'SymbolicReasoner' = None):
        self.levels: Dict[int, Set[Fact]] = defaultdict(set)
        self.abstraction_rules: Dict[int, List[Rule]] = defaultdict(list)
        self.parent_reasoner = parent_reasoner
        
    def add_fact_to_level(self, fact: Fact, level: int = 0):
        """Add fact to specific abstraction level."""
        self.levels[level].add(fact)
    
    def add_abstraction_rule(self, level: int, rule: Rule):
        """Add a rule that abstracts facts from 'level' to 'level + 1'."""
        self.abstraction_rules[level].append(rule)

    def abstract_to_higher_level(self, level: int) -> Set[Fact]:
        """Abstract facts from level to level+1."""
        if level not in self.levels or level not in self.abstraction_rules:
            return set()

        # Temporary reasoner to perform forward chaining with abstraction rules
        temp_reasoner = SymbolicReasoner(use_gnn=False)
        for fact in self.levels[level]:
            temp_reasoner.add_fact(fact.predicate, fact.arguments, fact.confidence, "hierarchical")
        
        for rule in self.abstraction_rules[level]:
            temp_reasoner.add_rule(rule.head, rule.body, rule.confidence)

        temp_reasoner.forward_chain(max_iterations=5)
        
        # Collect derived facts (the heads of the abstraction rules)
        abstract_predicates = {rule.head[0] for rule in self.abstraction_rules[level]}
        abstract_facts = {f for f in temp_reasoner.facts if f.predicate in abstract_predicates}

        for fact in abstract_facts:
            self.add_fact_to_level(fact, level + 1)

        return abstract_facts
    
    def reason_hierarchically(self, query: Tuple[str, Tuple[str, ...]], 
                             max_level: int = 3) -> List[Dict]:
        """Reason using hierarchical abstraction."""
        # Perform abstraction up to max_level
        for level in range(max_level):
            self.abstract_to_higher_level(level)

        # Try to answer query at any level, starting from the highest
        proofs = []
        for level in range(max_level, -1, -1):
            if level not in self.levels:
                continue

            temp_reasoner = SymbolicReasoner(use_gnn=False)
            for fact in self.levels[level]:
                temp_reasoner.add_fact(fact.predicate, fact.arguments, fact.confidence)

            # If we have parent reasoner, use its rules
            if self.parent_reasoner:
                temp_reasoner.rules = self.parent_reasoner.rules

            level_proofs = temp_reasoner.backward_chain(query)
            if level_proofs:
                for p in level_proofs:
                    p['hierarchical_level'] = level
                proofs.extend(level_proofs)
                break # Found answer at highest possible level
        
        return proofs


class SymbolicReasoner:
    """Enhanced symbolic reasoning with neural-symbolic integration.
    
    This reasoner combines classical symbolic logic with neural enhancements.
    It supports fuzzy logic operations, where the confidence of a conjunction
    is computed using the product t-norm:

    C(A ∧ B) = C(A) * C(B)

    And for a rule R: H :- B1, B2, ..., Bn with confidence C(R), the derived
    fact H has confidence:

    C(H) = C(R) * C(B1) * C(B2) * ... * C(Bn)

    Features:
    - Graph neural network for relational reasoning
    - Rule learning from examples
    - Hierarchical reasoning
    - Fuzzy logic operations
    - Temporal reasoning
    - Counterfactual explanations

    Attributes:
        facts (Set[Fact]): The set of known and derived facts.
        rules (List[Rule]): The set of logical rules.
        confidence_threshold (float): Threshold for pruning low-confidence facts.
        use_gnn (bool): Whether to use GNN-based refinement.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.3,
                 use_gnn: bool = True,
                 feature_dim: int = 128):
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []
        self.confidence_threshold = confidence_threshold
        self._fact_index: Dict[str, Set[Fact]] = defaultdict(set)
        self._arg_index: Dict[str, Set[Fact]] = defaultdict(set)
        
        # GNN for reasoning
        self.use_gnn = use_gnn
        if use_gnn:
            self.gnn = GraphNeuralReasoner(feature_dim)
            self.fact_embeddings: Dict[Fact, torch.Tensor] = {}
        
        # Hierarchical reasoning
        self.hierarchical = HierarchicalReasoner()
        
        # Rule learning
        self.candidate_rules: List[Rule] = []
        
        # Temporal state
        self.time_step = 0
        
    def add_fact(self, predicate: str, arguments: Tuple[str, ...], 
                 confidence: float = 1.0, source: str = "perceived",
                 persistence: float = 1.0):
        """Add fact with indexing."""
        fact = Fact(predicate, arguments, confidence, self.time_step, source, persistence)
        if fact not in self.facts:
            self.facts.add(fact)
            self._fact_index[predicate].add(fact)
            for arg in arguments:
                self._arg_index[arg].add(fact)
            
            # Initialize GNN embedding
            if self.use_gnn:
                self.fact_embeddings[fact] = self._compute_fact_embedding(fact)
    
    def _compute_fact_embedding(self, fact: Fact) -> torch.Tensor:
        """Compute neural embedding for a fact."""
        # Simple hash-based embedding (in practice, use learned embeddings)
        embedding = torch.randn(self.gnn.feature_dim if hasattr(self, 'gnn') else 128)
        return embedding
    
    def add_rule(self, head: Tuple[str, Tuple[str, ...]], 
                 body: List[Tuple[str, Tuple[str, ...]]], 
                 confidence: float = 1.0):
        """Add logical rule."""
        rule = Rule(head, body, confidence)
        self.rules.append(rule)
    
    def query(self, predicate: str, arguments: Tuple[str, ...]) -> Optional[float]:
        """Query with fuzzy matching."""
        best_confidence = None
        
        for fact in self._fact_index.get(predicate, set()):
            similarity = self._compute_similarity(arguments, fact.arguments)
            if similarity > 0.8:  # Fuzzy threshold
                conf = fact.confidence * similarity
                if best_confidence is None or conf > best_confidence:
                    best_confidence = conf
        
        return best_confidence
    
    def _compute_similarity(self, args1: Tuple[str, ...], args2: Tuple[str, ...]) -> float:
        """Compute similarity between argument tuples."""
        if len(args1) != len(args2):
            return 0.0
        
        matches = sum(1 for a1, a2 in zip(args1, args2) if a1 == a2 or a1.startswith('?') or a2.startswith('?'))
        return matches / len(args1)
    
    def forward_chain(self, max_iterations: int = 10, use_gnn: bool = None) -> int:
        """Enhanced forward chaining with GNN support.
        
        Args:
            max_iterations: Maximum inference iterations
            use_gnn: Whether to use GNN (overrides self.use_gnn)
            
        Returns:
            Number of new facts derived
        """
        if use_gnn is None:
            use_gnn = self.use_gnn
        
        new_facts = 0
        
        # Traditional symbolic forward chaining
        for iteration in range(max_iterations):
            derived_this_iteration = 0
            
            for rule in self.rules:
                # Track rule usage
                rule.usage_count += 1
                
                # Match rule body
                for binding in self._match_rule_body(rule.body):
                    # Substitute and derive
                    head_pred, head_args = rule.head
                    new_args = tuple(binding.get(arg, arg) for arg in head_args)
                    
                    # Compute confidence with fuzzy operations
                    confidence = rule.confidence * rule.strength
                    for pred, args in rule.body:
                        bound_args = tuple(binding.get(arg, arg) for arg in args)
                        fact_conf = self.query(pred, bound_args)
                        if fact_conf is not None:
                            # Fuzzy AND (product t-norm)
                            confidence *= fact_conf
                    
                    # Add derived fact
                    if confidence >= self.confidence_threshold:
                        fact = Fact(head_pred, new_args, confidence, self.time_step, "derived")
                        if fact not in self.facts:
                            self.add_fact(head_pred, new_args, confidence, "derived")
                            derived_this_iteration += 1
                            rule.success_count += 1
            
            new_facts += derived_this_iteration
            if derived_this_iteration == 0:
                break
        
        # GNN-based refinement
        if use_gnn and self.use_gnn and len(self.facts) > 1:
            refined_facts = self._gnn_reasoning()
            new_facts += refined_facts
        
        return new_facts
    
    def _gnn_reasoning(self) -> int:
        """Use GNN to refine fact confidences and derive new facts."""
        if not self.fact_embeddings:
            return 0
        
        # Build graph
        facts_list = list(self.facts)
        node_features = torch.stack([self.fact_embeddings[f] for f in facts_list])
        
        # Optimized edge index construction using argument indexing
        edges = []
        arg_to_indices = defaultdict(list)
        for idx, fact in enumerate(facts_list):
            for arg in fact.arguments:
                arg_to_indices[arg].append(idx)

        for indices in arg_to_indices.values():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edges.append([indices[i], indices[j]])
                    edges.append([indices[j], indices[i]])
        
        if not edges:
            return 0
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Run GNN
        with torch.no_grad():
            updated_features = self.gnn(node_features, edge_index)
            refined_confidences = self.gnn.predict_confidence(updated_features)
        
        # Update fact confidences
        new_facts = 0
        for i, fact in enumerate(facts_list):
            new_conf = refined_confidences[i].item()
            if new_conf > fact.confidence:
                # Update confidence
                fact.confidence = new_conf
                new_facts += 1
        
        return new_facts
    
    def _match_rule_body(self, body: List[Tuple[str, Tuple[str, ...]]]) -> List[Dict[str, str]]:
        """Find variable bindings satisfying rule body."""
        if not body:
            return [{}]
        
        pred, args = body[0]
        bindings_list = []
        
        for fact in self._fact_index.get(pred, set()):
            binding = self._try_bind(args, fact.arguments)
            if binding is not None:
                if len(body) == 1:
                    bindings_list.append(binding)
                else:
                    remaining = [
                        (p, tuple(binding.get(a, a) for a in args))
                        for p, args in body[1:]
                    ]
                    sub_bindings = self._match_rule_body(remaining)
                    for sub_binding in sub_bindings:
                        combined = {**binding, **sub_binding}
                        bindings_list.append(combined)
        
        return bindings_list
    
    def _try_bind(self, pattern: Tuple[str, ...], 
                 instance: Tuple[str, ...]) -> Optional[Dict[str, str]]:
        """Try variable binding."""
        if len(pattern) != len(instance):
            return None
        
        binding = {}
        for p, i in zip(pattern, instance):
            if p.startswith("?"):
                if p in binding and binding[p] != i:
                    return None
                binding[p] = i
            elif p != i:
                return None
        
        return binding
    
    def backward_chain(self, goal: Tuple[str, Tuple[str, ...]], 
                      depth: int = 0, max_depth: int = 5) -> List[Dict[str, Any]]:
        """Enhanced backward chaining with better pruning."""
        if depth > max_depth:
            return []
        
        pred, args = goal
        proofs = []
        
        # Check known facts
        for fact in self._fact_index.get(pred, set()):
            binding = self._try_bind(args, fact.arguments)
            if binding is not None and fact.confidence >= self.confidence_threshold:
                proofs.append({
                    "binding": binding,
                    "confidence": fact.confidence,
                    "proof": [str(fact)],
                    "depth": depth
                })
        
        # Try to prove via rules (only use strong rules)
        for rule in sorted(self.rules, key=lambda r: r.strength * r.confidence, reverse=True):
            if rule.head[0] == pred:
                head_binding = self._try_bind(rule.head[1], args)
                if head_binding is not None:
                    body_proofs = self._prove_conjunction(
                        rule.body, head_binding, depth + 1, max_depth
                    )
                    
                    for body_proof in body_proofs:
                        confidence = rule.confidence * rule.strength * body_proof["confidence"]
                        if confidence >= self.confidence_threshold:
                            proofs.append({
                                "binding": body_proof["binding"],
                                "confidence": confidence,
                                "proof": [str(rule)] + body_proof["proof"],
                                "depth": depth
                            })
        
        # Sort by confidence and return top proofs
        proofs.sort(key=lambda p: p["confidence"], reverse=True)
        return proofs[:10]  # Return top 10
    
    def _prove_conjunction(self, goals: List[Tuple[str, Tuple[str, ...]]], 
                          initial_binding: Dict[str, str],
                          depth: int, max_depth: int) -> List[Dict[str, Any]]:
        """Prove conjunction of goals."""
        if not goals:
            return [{"binding": initial_binding, "confidence": 1.0, "proof": []}]
        
        pred, args = goals[0]
        bound_args = tuple(initial_binding.get(arg, arg) for arg in args)
        
        first_proofs = self.backward_chain((pred, bound_args), depth, max_depth)
        
        all_proofs = []
        for first_proof in first_proofs:
            combined_binding = {**initial_binding, **first_proof["binding"]}
            remaining_proofs = self._prove_conjunction(
                goals[1:], combined_binding, depth, max_depth
            )
            
            for remaining_proof in remaining_proofs:
                confidence = first_proof["confidence"] * remaining_proof["confidence"]
                all_proofs.append({
                    "binding": {**combined_binding, **remaining_proof["binding"]},
                    "confidence": confidence,
                    "proof": first_proof["proof"] + remaining_proof["proof"]
                })
        
        return all_proofs
    
    def explain(self, fact: Tuple[str, Tuple[str, ...]], max_depth: int = 3) -> List[str]:
        """Generate rich explanations with counterfactuals."""
        proofs = self.backward_chain(fact, max_depth=max_depth)
        explanations = []
        
        for i, proof in enumerate(proofs[:5]):
            explanation = f"Explanation {i+1} (confidence: {proof['confidence']:.3f}, depth: {proof['depth']}):\n"
            explanation += "\n".join(f"  → {step}" for step in proof["proof"])
            
            # Add counterfactual
            explanation += "\n\nCounterfactual: "
            explanation += "This would not hold if any supporting fact was removed."
            
            explanations.append(explanation)
        
        return explanations
    
    def _generalize_to_rule(self, conclusion: Fact, premises: List[Fact]) -> Rule:
        """Generalize specific facts into a rule with variables.

        Replaces shared arguments with variables (?x, ?y, etc.) to create
        an abstract rule from concrete examples.
        """
        # Map specific arguments to variables
        arg_to_var = {}
        var_count = 0

        # All arguments in the example
        all_args = set(conclusion.arguments)
        for p in premises:
            all_args.update(p.arguments)

        # Create mapping for shared arguments
        for arg in sorted(list(all_args)):
            # If argument appears in more than one place, it's a candidate for a variable
            occurrences = 0
            if arg in conclusion.arguments:
                occurrences += 1
            for p in premises:
                if arg in p.arguments:
                    occurrences += 1

            if occurrences > 1:
                arg_to_var[arg] = f"?v{var_count}"
                var_count += 1

        # Build generalized head and body
        head_args = tuple(arg_to_var.get(a, a) for a in conclusion.arguments)
        head = (conclusion.predicate, head_args)

        body = []
        for p in premises:
            body_args = tuple(arg_to_var.get(a, a) for a in p.arguments)
            body.append((p.predicate, body_args))

        # Initial confidence based on example confidences
        avg_conf = (conclusion.confidence + sum(p.confidence for p in premises)) / (1 + len(premises))

        return Rule(head, body, confidence=avg_conf, strength=0.5)

    def learn_rule_from_examples(self, examples: List[Tuple[Fact, List[Fact]]]):
        """Learn and refine rules from positive examples.

        Uses inductive logic programming principles to generalize specific
        observations into reusable symbolic rules.
        
        Args:
            examples: List of (conclusion_fact, premise_facts) pairs
        """
        for conclusion, premises in examples:
            if not premises:
                continue
            
            # Generate candidate rule
            candidate = self._generalize_to_rule(conclusion, premises)

            # Check if we already have a similar rule
            found = False
            for rule in self.rules:
                if rule.head == candidate.head and set(rule.body) == set(candidate.body):
                    # Update existing rule statistics
                    rule.update_statistics(success=True)
                    rule.confidence = (rule.confidence + candidate.confidence) / 2
                    found = True
                    break
            
            if not found:
                # Add as a new rule if it has enough support
                # In a real system, we'd wait for multiple examples
                self.add_rule(candidate.head, candidate.body, candidate.confidence)
    
    def temporal_forward_chain(self, max_iterations: int = 10) -> int:
        """Forward chaining with enhanced temporal reasoning.

        Increments the system's internal clock and performs inference.
        Facts decay over time unless they have high persistence.
        """
        self.time_step += 1
        
        # Regular forward chaining
        new_facts = self.forward_chain(max_iterations)
        
        # Decay old facts with persistence awareness
        to_remove = set()
        for fact in self.facts:
            if fact.timestamp is not None:
                age = self.time_step - fact.timestamp

                # Dynamic decay based on persistence
                # High persistence facts (e.g., > 1.0) decay slowly
                # Standard persistence (1.0) gives 0.9 decay rate
                decay_rate = 1.0 - (0.1 / fact.persistence)
                decay_rate = max(0.0, min(0.999, decay_rate))

                if age > 5:
                    fact.confidence *= (decay_rate ** (age - 5))

            if fact.confidence < self.confidence_threshold:
                to_remove.add(fact)

        # Clean up expired facts
        for fact in to_remove:
            self.facts.remove(fact)
            self._fact_index[fact.predicate].discard(fact)
            for arg in fact.arguments:
                self._arg_index[arg].discard(fact)
        
        return new_facts
    
    def clear(self):
        """Clear all facts and reset."""
        self.facts.clear()
        self.rules.clear()
        self._fact_index.clear()
        self._arg_index.clear()
        if self.use_gnn:
            self.fact_embeddings.clear()
        self.time_step = 0


