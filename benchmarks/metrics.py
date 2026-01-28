"""Evaluation metrics for neurosymbolic reasoning."""

import numpy as np
from typing import Dict, List, Any
from scipy import stats


class ReasoningMetrics:
    """Comprehensive metrics for neurosymbolic evaluation."""
    
    @staticmethod
    def accuracy(predictions: List[Any], targets: List[Any]) -> float:
        """Standard accuracy."""
        correct = sum(p == t for p, t in zip(predictions, targets))
        return correct / len(predictions) if predictions else 0.0
    
    @staticmethod
    def compositional_accuracy(predictions: List[Dict], targets: List[Dict]) -> float:
        """Accuracy on compositional reasoning tasks."""
        scores = []
        for pred, target in zip(predictions, targets):
            # Check if all components are correct
            if isinstance(pred, dict) and isinstance(target, dict):
                component_correct = [
                    pred.get(k) == target.get(k) for k in target.keys()
                ]
                scores.append(all(component_correct))
        
        return np.mean(scores) if scores else 0.0
    
    @staticmethod
    def explanation_quality(explanations: List[List[str]]) -> Dict[str, float]:
        """Evaluate explanation quality."""
        if not explanations:
            return {"avg_length": 0.0, "coverage": 0.0}
        
        lengths = [len(exp) for exp in explanations if exp]
        
        return {
            "avg_length": np.mean(lengths) if lengths else 0.0,
            "coverage": len([e for e in explanations if e]) / len(explanations),
            "min_length": min(lengths) if lengths else 0.0,
            "max_length": max(lengths) if lengths else 0.0,
        }
    
    @staticmethod
    def logical_consistency(facts: List[List[tuple]]) -> float:
        """Measure logical consistency of derived facts."""
        if not facts:
            return 1.0
        
        consistency_scores = []
        
        for fact_set in facts:
            # Check for contradictions
            predicates = {}
            contradictions = 0
            
            for pred, args, conf in fact_set:
                key = (pred, args)
                if key in predicates:
                    # Check if confidence is very different
                    if abs(predicates[key] - conf) > 0.5:
                        contradictions += 1
                predicates[key] = conf
            
            consistency = 1.0 - (contradictions / len(fact_set) if fact_set else 0)
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores)
    
    @staticmethod
    def reasoning_depth(reasoning_chains: List[List[str]]) -> Dict[str, float]:
        """Measure depth of reasoning chains."""
        if not reasoning_chains:
            return {"avg_depth": 0.0, "max_depth": 0.0}
        
        depths = [len(chain) for chain in reasoning_chains]
        
        return {
            "avg_depth": np.mean(depths),
            "max_depth": max(depths),
            "min_depth": min(depths),
            "std_depth": np.std(depths),
        }
    
    @staticmethod
    def generalization_score(train_acc: float, val_acc: float, test_acc: float) -> float:
        """Measure generalization capability."""
        # Penalize large train-test gap
        gap_penalty = abs(train_acc - test_acc)
        
        # Reward consistent performance across splits
        consistency = 1.0 - np.std([train_acc, val_acc, test_acc])
        
        return test_acc * (1.0 - gap_penalty) * consistency
    
    @staticmethod
    def symbolic_grounding_quality(predictions: List[List[tuple]], 
                                   ground_truth: List[List[str]]) -> float:
        """Evaluate quality of neural-to-symbolic grounding."""
        if not predictions or not ground_truth:
            return 0.0
        
        scores = []
        for pred, gt in zip(predictions, ground_truth):
            pred_concepts = {p[0] for p in pred}
            gt_concepts = set(gt)
            
            if gt_concepts:
                precision = len(pred_concepts & gt_concepts) / len(pred_concepts) if pred_concepts else 0
                recall = len(pred_concepts & gt_concepts) / len(gt_concepts)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                scores.append(f1)
        
        return np.mean(scores) if scores else 0.0
    
    @staticmethod
    def inference_efficiency(latencies: List[float], accuracies: List[float]) -> Dict[str, float]:
        """Compute efficiency metrics."""
        if not latencies or not accuracies:
            return {"efficiency_score": 0.0}
        
        avg_latency = np.mean(latencies)
        avg_accuracy = np.mean(accuracies)
        
        # Efficiency = Accuracy / Latency (higher is better)
        efficiency = avg_accuracy / avg_latency if avg_latency > 0 else 0
        
        return {
            "efficiency_score": efficiency,
            "avg_latency_ms": avg_latency * 1000,
            "throughput_qps": 1.0 / avg_latency if avg_latency > 0 else 0,
            "avg_accuracy": avg_accuracy,
        }
    
    @staticmethod
    def statistical_significance(baseline_scores: List[float], 
                                model_scores: List[float]) -> Dict[str, Any]:
        """Test statistical significance of improvements."""
        if len(baseline_scores) < 2 or len(model_scores) < 2:
            return {"significant": False, "p_value": 1.0}
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model_scores, baseline_scores)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(model_scores) - np.mean(baseline_scores)
        pooled_std = np.sqrt((np.std(baseline_scores)**2 + np.std(model_scores)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            "significant": p_value < 0.05,
            "p_value": p_value,
            "t_statistic": t_stat,
            "cohens_d": cohens_d,
            "improvement": mean_diff,
        }
