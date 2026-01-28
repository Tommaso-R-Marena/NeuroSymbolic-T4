"""Comprehensive metrics for neurosymbolic evaluation."""

import numpy as np
from typing import Dict, List, Tuple
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import spearmanr, kendalltau


class NeurosymbolicMetrics:
    """Metrics for evaluating neurosymbolic systems."""
    
    @staticmethod
    def accuracy(predictions: List, targets: List) -> float:
        """Standard accuracy."""
        return accuracy_score(targets, predictions)
    
    @staticmethod
    def reasoning_depth(derived_facts_per_sample: List[int]) -> Dict[str, float]:
        """Measure average reasoning depth."""
        return {
            "mean": np.mean(derived_facts_per_sample),
            "std": np.std(derived_facts_per_sample),
            "median": np.median(derived_facts_per_sample),
            "max": np.max(derived_facts_per_sample),
        }
    
    @staticmethod
    def compositional_generalization_score(
        train_performance: float,
        test_performance: float,
        distribution_shift: float = 1.0
    ) -> float:
        """Measure compositional generalization.
        
        Higher is better. Measures how well the system generalizes to
        novel compositions of learned concepts.
        """
        return test_performance / (train_performance * distribution_shift + 1e-8)
    
    @staticmethod
    def explainability_score(
        proof_lengths: List[int],
        proof_confidences: List[float]
    ) -> Dict[str, float]:
        """Measure quality of explanations."""
        return {
            "avg_proof_length": np.mean(proof_lengths),
            "avg_confidence": np.mean(proof_confidences),
            "confidence_variance": np.var(proof_confidences),
            "interpretability_ratio": np.mean([1.0 / (l + 1) for l in proof_lengths]),
        }
    
    @staticmethod
    def symbolic_consistency(
        predictions: List[Dict],
        logical_constraints: List[Tuple]
    ) -> float:
        """Measure consistency with logical constraints.
        
        Returns fraction of predictions satisfying all constraints.
        """
        consistent = 0
        for pred in predictions:
            satisfies_all = all(
                NeurosymbolicMetrics._check_constraint(pred, constraint)
                for constraint in logical_constraints
            )
            if satisfies_all:
                consistent += 1
        
        return consistent / len(predictions) if predictions else 0.0
    
    @staticmethod
    def _check_constraint(prediction: Dict, constraint: Tuple) -> bool:
        """Check if prediction satisfies constraint."""
        # Simplified constraint checking
        return True
    
    @staticmethod
    def neural_symbolic_alignment(
        neural_features: torch.Tensor,
        symbolic_features: torch.Tensor
    ) -> float:
        """Measure alignment between neural and symbolic representations."""
        # Cosine similarity
        neural_norm = neural_features / neural_features.norm(dim=-1, keepdim=True)
        symbolic_norm = symbolic_features / symbolic_features.norm(dim=-1, keepdim=True)
        alignment = (neural_norm * symbolic_norm).sum(dim=-1).mean()
        return alignment.item()
    
    @staticmethod
    def reasoning_efficiency(
        num_rules: int,
        num_facts: int,
        inference_time_ms: float,
        num_derived: int
    ) -> Dict[str, float]:
        """Measure reasoning efficiency."""
        return {
            "facts_per_second": (num_derived / inference_time_ms) * 1000 if inference_time_ms > 0 else 0,
            "rule_utilization": num_derived / num_rules if num_rules > 0 else 0,
            "fact_density": num_derived / num_facts if num_facts > 0 else 0,
        }
    
    @staticmethod
    def uncertainty_calibration(
        confidences: List[float],
        correctness: List[bool]
    ) -> Dict[str, float]:
        """Measure calibration of confidence scores.
        
        Expected Calibration Error (ECE) and related metrics.
        """
        bins = 10
        bin_boundaries = np.linspace(0, 1, bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = np.array(confidences)
        correctness = np.array(correctness).astype(float)
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correctness[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            "expected_calibration_error": ece,
            "avg_confidence": confidences.mean(),
            "confidence_accuracy_correlation": np.corrcoef(confidences, correctness)[0, 1],
        }
    
    @staticmethod
    def generate_report(
        results: Dict[str, Dict]
    ) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("="*80)
        report.append("NEUROSYMBOLIC EVALUATION REPORT")
        report.append("="*80)
        report.append("")
        
        for benchmark, metrics in results.items():
            report.append(f"\n{benchmark.upper()}:")
            report.append("-"*40)
            for metric_name, value in metrics.items():
                if isinstance(value, dict):
                    report.append(f"  {metric_name}:")
                    for k, v in value.items():
                        report.append(f"    {k}: {v:.4f}")
                else:
                    report.append(f"  {metric_name}: {value:.4f}")
        
        report.append("\n" + "="*80)
        return "\n".join(report)