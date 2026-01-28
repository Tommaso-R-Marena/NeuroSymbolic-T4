"""Statistical significance testing for ICML paper."""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalAnalysis:
    """Statistical significance testing and analysis."""
    
    @staticmethod
    def paired_t_test(model_scores: List[float], 
                     baseline_scores: List[float]) -> Dict[str, float]:
        """Paired t-test for significance."""
        if len(model_scores) != len(baseline_scores):
            raise ValueError("Score lists must have same length")
        
        t_stat, p_value = stats.ttest_rel(model_scores, baseline_scores)
        
        # Effect size (Cohen's d)
        diff = np.array(model_scores) - np.array(baseline_scores)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        
        # Confidence interval
        ci = stats.t.interval(0.95, len(diff)-1,
                            loc=np.mean(diff),
                            scale=stats.sem(diff))
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": p_value < 0.05,
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "mean_improvement": float(np.mean(diff)),
        }
    
    @staticmethod
    def mcnemar_test(model_correct: List[bool],
                    baseline_correct: List[bool]) -> Dict[str, float]:
        """McNemar's test for paired binary outcomes."""
        # Contingency table
        both_correct = sum([m and b for m, b in zip(model_correct, baseline_correct)])
        model_only = sum([m and not b for m, b in zip(model_correct, baseline_correct)])
        baseline_only = sum([not m and b for m, b in zip(model_correct, baseline_correct)])
        both_wrong = sum([not m and not b for m, b in zip(model_correct, baseline_correct)])
        
        # McNemar statistic
        if model_only + baseline_only == 0:
            p_value = 1.0
        else:
            statistic = (abs(model_only - baseline_only) - 1)**2 / (model_only + baseline_only)
            p_value = 1 - stats.chi2.cdf(statistic, 1)
        
        return {
            "both_correct": both_correct,
            "model_only_correct": model_only,
            "baseline_only_correct": baseline_only,
            "both_wrong": both_wrong,
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }
    
    @staticmethod
    def bootstrap_ci(scores: List[float], 
                    n_bootstrap: int = 10000,
                    confidence: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval."""
        scores = np.array(scores)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return float(lower), float(upper)
    
    @staticmethod
    def plot_comparison(model_scores: List[float],
                       baseline_scores: List[float],
                       title: str = "Model Comparison",
                       save_path: str = None):
        """Plot score comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot
        axes[0].boxplot([baseline_scores, model_scores],
                       labels=["Baseline", "NeuroSymbolic-T4"])
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Score Distribution")
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(baseline_scores, model_scores, alpha=0.5)
        axes[1].plot([0, 1], [0, 1], 'r--', label="Equal Performance")
        axes[1].set_xlabel("Baseline Accuracy")
        axes[1].set_ylabel("NeuroSymbolic-T4 Accuracy")
        axes[1].set_title("Per-Sample Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def anova_test(groups: Dict[str, List[float]]) -> Dict[str, float]:
        """One-way ANOVA for multiple groups."""
        group_data = list(groups.values())
        f_stat, p_value = stats.f_oneway(*group_data)
        
        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }
    
    @staticmethod
    def generate_significance_table(tests: Dict[str, Dict]) -> str:
        """Generate LaTeX table of significance tests."""
        latex = r"""
\begin{table}[t]
\centering
\caption{Statistical Significance Tests}
\label{tab:significance}
\begin{tabular}{lcccc}
\toprule
Comparison & $t$ & $p$ & Cohen's $d$ & 95\% CI \\
\midrule
"""
        
        for name, result in tests.items():
            if "t_statistic" in result:
                latex += f"{name} & "
                latex += f"{result['t_statistic']:.2f} & "
                latex += f"{result['p_value']:.4f} & "
                latex += f"{result['cohens_d']:.2f} & "
                latex += f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}] "
                if result['significant']:
                    latex += r"$^*$ "
                latex += r"\\" + "\n"
        
        latex += r"""
\bottomrule
\multicolumn{5}{l}{$^*$ Significant at $p < 0.05$}
\end{tabular}
\end{table}
"""
        
        return latex