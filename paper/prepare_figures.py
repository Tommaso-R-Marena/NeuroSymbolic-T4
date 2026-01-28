"""Generate figures for ICML paper."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

sns.set_style("whitegrid")
sns.set_context("paper")


def plot_reasoning_depth(results: dict, output_dir: Path):
    """Plot reasoning depth distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Example data
    depths = np.random.poisson(3, 1000)  # Replace with actual data
    
    ax.hist(depths, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Reasoning Depth (# Derived Facts)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Reasoning Depth")
    
    plt.tight_layout()
    plt.savefig(output_dir / "reasoning_depth.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_comparison(results: dict, output_dir: Path):
    """Plot performance comparison across benchmarks."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    benchmarks = list(results.keys())
    # Extract comparable metrics
    scores = [results[b].get("accuracy", 0) * 100 for b in benchmarks]
    
    bars = ax.bar(benchmarks, scores, color='steelblue', edgecolor='black')
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Performance Across Benchmarks")
    ax.set_ylim([0, 100])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_study(output_dir: Path):
    """Plot ablation study results."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = [
        "Full Model",
        "w/o Symbolic Reasoning",
        "w/o Forward Chaining",
        "w/o Backward Chaining",
        "Neural Only"
    ]
    
    scores = [75.3, 68.2, 71.5, 72.8, 64.1]  # Example values
    
    bars = ax.barh(models, scores, color='coral', edgecolor='black')
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Ablation Study")
    ax.set_xlim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_study.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    output_dir = Path("./paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_file = Path("./benchmark_results/benchmark_results.json")
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
    else:
        results = {}  # Use dummy data
    
    print("Generating figures for paper...")
    
    plot_reasoning_depth(results, output_dir)
    print("✓ reasoning_depth.pdf")
    
    plot_performance_comparison(results, output_dir)
    print("✓ performance_comparison.pdf")
    
    plot_ablation_study(output_dir)
    print("✓ ablation_study.pdf")
    
    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()