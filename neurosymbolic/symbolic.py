"""Symbolic reasoning engine with logic programming and theorem proving."""

import torch
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class Fact:
    """Atomic fact in the knowledge base."""
    predicate: str
    arguments: Tuple[str, ...]
    confidence: float = 1.0
    
    def __str__(self) -> str:
        args = ", ".join(self.arguments)
        return f"{self.predicate}({args}) [{self.confidence:.3f}]"
    
    def __hash__(self) -> int:
        return hash((self.predicate, self.arguments))
    
    def __eq__(self, other) -> bool:
        return (self.predicate == other.predicate and 
                self.arguments == other.arguments)


@dataclass
class Rule:
    """Logical rule for inference."""
    head: Tuple[str, Tuple[str, ...]]  # (predicate, arguments)
    body: List[Tuple[str, Tuple[str, ...]]]  # List of (predicate, arguments)
    confidence: float = 1.0
    
    def __str__(self) -> str:
        head_str = f"{self.head[0]}({', '.join(self.head[1])})"
        body_str = " AND ".join(
            f"{pred}({', '.join(args)})" for pred, args in self.body
        )
        return f"{head_str} :- {body_str} [{self.confidence:.3f}]"


class SymbolicReasoner:
    """Symbolic reasoning engine combining logic programming and probabilistic inference.
    
    Implements:
    - Forward chaining inference
    - Backward chaining (goal-directed) reasoning
    - Abductive reasoning for explanation generation
    - Probabilistic logic with uncertainty propagation
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []
        self.confidence_threshold = confidence_threshold
        self._fact_index: Dict[str, Set[Fact]] = defaultdict(set)
        
    def add_fact(self, predicate: str, arguments: Tuple[str, ...], confidence: float = 1.0):
        """Add a fact to the knowledge base."""
        fact = Fact(predicate, arguments, confidence)
        if fact not in self.facts:
            self.facts.add(fact)
            self._fact_index[predicate].add(fact)
    
    def add_rule(self, head: Tuple[str, Tuple[str, ...]], 
                 body: List[Tuple[str, Tuple[str, ...]]], 
                 confidence: float = 1.0):
        """Add a logical rule."""
        rule = Rule(head, body, confidence)
        self.rules.append(rule)
    
    def query(self, predicate: str, arguments: Tuple[str, ...]) -> Optional[float]:
        """Query for a specific fact and return its confidence."""
        for fact in self._fact_index.get(predicate, set()):
            if self._unify(arguments, fact.arguments):
                return fact.confidence
        return None
    
    def _unify(self, pattern: Tuple[str, ...], instance: Tuple[str, ...]) -> bool:
        """Check if pattern unifies with instance (supports variables starting with ?)."""
        if len(pattern) != len(instance):
            return False
        
        for p, i in zip(pattern, instance):
            if p.startswith("?"):  # Variable
                continue
            if p != i:  # Constant must match
                return False
        return True
    
    def _substitute(self, template: Tuple[str, ...], 
                   pattern: Tuple[str, ...], 
                   instance: Tuple[str, ...]) -> Tuple[str, ...]:
        """Substitute variables in template based on pattern-instance binding."""
        bindings = {}
        for p, i in zip(pattern, instance):
            if p.startswith("?"):
                bindings[p] = i
        
        return tuple(bindings.get(arg, arg) for arg in template)
    
    def forward_chain(self, max_iterations: int = 10) -> int:
        """Apply forward chaining to derive new facts.
        
        Returns:
            Number of new facts derived
        """
        new_facts = 0
        
        for iteration in range(max_iterations):
            derived_this_iteration = 0
            
            for rule in self.rules:
                # Try to match rule body with existing facts
                for binding in self._match_rule_body(rule.body):
                    # Substitute variables in head
                    head_pred, head_args = rule.head
                    new_args = tuple(binding.get(arg, arg) for arg in head_args)
                    
                    # Compute confidence (product of body confidences and rule confidence)
                    confidence = rule.confidence
                    for pred, args in rule.body:
                        bound_args = tuple(binding.get(arg, arg) for arg in args)
                        fact_conf = self.query(pred, bound_args)
                        if fact_conf is not None:
                            confidence *= fact_conf
                    
                    # Add derived fact if above threshold
                    if confidence >= self.confidence_threshold:
                        fact = Fact(head_pred, new_args, confidence)
                        if fact not in self.facts:
                            self.add_fact(head_pred, new_args, confidence)
                            derived_this_iteration += 1
            
            new_facts += derived_this_iteration
            if derived_this_iteration == 0:
                break
        
        return new_facts
    
    def _match_rule_body(self, body: List[Tuple[str, Tuple[str, ...]]]) -> List[Dict[str, str]]:
        """Find all variable bindings that satisfy the rule body."""
        if not body:
            return [{}]
        
        # Start with first body predicate
        pred, args = body[0]
        bindings_list = []
        
        # Find all matching facts
        for fact in self._fact_index.get(pred, set()):
            binding = self._try_bind(args, fact.arguments)
            if binding is not None:
                # Recursively match remaining body predicates
                if len(body) == 1:
                    bindings_list.append(binding)
                else:
                    # Substitute variables in remaining body
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
        """Try to bind variables in pattern to instance."""
        if len(pattern) != len(instance):
            return None
        
        binding = {}
        for p, i in zip(pattern, instance):
            if p.startswith("?"):
                if p in binding and binding[p] != i:
                    return None  # Inconsistent binding
                binding[p] = i
            elif p != i:
                return None  # Constants must match
        
        return binding
    
    def backward_chain(self, goal: Tuple[str, Tuple[str, ...]], 
                      depth: int = 0, max_depth: int = 5) -> List[Dict[str, Any]]:
        """Backward chaining to prove a goal.
        
        Returns:
            List of proofs (variable bindings and confidence)
        """
        if depth > max_depth:
            return []
        
        pred, args = goal
        proofs = []
        
        # Check if goal is a known fact
        for fact in self._fact_index.get(pred, set()):
            binding = self._try_bind(args, fact.arguments)
            if binding is not None:
                proofs.append({
                    "binding": binding,
                    "confidence": fact.confidence,
                    "proof": [str(fact)]
                })
        
        # Try to prove via rules
        for rule in self.rules:
            if rule.head[0] == pred:
                head_binding = self._try_bind(args, rule.head[1])
                if head_binding is not None:
                    # Try to prove rule body
                    body_proofs = self._prove_conjunction(
                        rule.body, head_binding, depth + 1, max_depth
                    )
                    
                    for body_proof in body_proofs:
                        confidence = rule.confidence * body_proof["confidence"]
                        if confidence >= self.confidence_threshold:
                            proofs.append({
                                "binding": body_proof["binding"],
                                "confidence": confidence,
                                "proof": [str(rule)] + body_proof["proof"]
                            })
        
        return proofs
    
    def _prove_conjunction(self, goals: List[Tuple[str, Tuple[str, ...]]], 
                          initial_binding: Dict[str, str],
                          depth: int, max_depth: int) -> List[Dict[str, Any]]:
        """Prove a conjunction of goals."""
        if not goals:
            return [{"binding": initial_binding, "confidence": 1.0, "proof": []}]
        
        # Substitute initial binding in first goal
        pred, args = goals[0]
        bound_args = tuple(initial_binding.get(arg, arg) for arg in args)
        
        # Prove first goal
        first_proofs = self.backward_chain((pred, bound_args), depth, max_depth)
        
        all_proofs = []
        for first_proof in first_proofs:
            # Combine bindings
            combined_binding = {**initial_binding, **first_proof["binding"]}
            
            # Prove remaining goals
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
        """Generate explanations for why a fact holds."""
        proofs = self.backward_chain(fact, max_depth=max_depth)
        explanations = []
        
        for i, proof in enumerate(proofs[:5]):  # Top 5 explanations
            explanation = f"Explanation {i+1} (confidence: {proof['confidence']:.3f}):\n"
            explanation += "\n".join(f"  - {step}" for step in proof["proof"])
            explanations.append(explanation)
        
        return explanations
    
    def clear(self):
        """Clear all facts and rules."""
        self.facts.clear()
        self.rules.clear()
        self._fact_index.clear()