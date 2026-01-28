"""Evaluation script for neurosymbolic system."""

import torch
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from neurosymbolic import NeurosymbolicSystem


def evaluate_reasoning(model, device, num_samples=100):
    """Evaluate reasoning capabilities."""
    model.eval()
    
    print("\nEvaluating reasoning capabilities...")
    
    results = {
        "forward_chaining": [],
        "backward_chaining": [],
        "explanation_quality": [],
    }
    
    for i in tqdm(range(num_samples)):
        # Generate random input
        x = torch.randn(1, 3, 224, 224).to(device)
        
        # Full forward pass
        with torch.no_grad():
            output = model.forward(x, threshold=0.5)
        
        # Check reasoning outputs
        reasoning = output["reasoning"][0]
        results["forward_chaining"].append(reasoning["num_derived"])
        
        # Test backward chaining
        if reasoning["derived_facts"]:
            fact = reasoning["derived_facts"][0]
            query = (fact[0], fact[1])
            proofs = model.query(x, query, threshold=0.5)
            results["backward_chaining"].append(len(proofs))
            
            if proofs:
                # Explanation quality (length of proof)
                avg_proof_length = np.mean([len(p["proof"]) for p in proofs])
                results["explanation_quality"].append(avg_proof_length)
    
    # Compute statistics
    stats = {}
    for key, values in results.items():
        if values:
            stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
    
    return stats


def evaluate_perception(model, device, num_samples=100):
    """Evaluate perception module."""
    model.eval()
    
    print("\nEvaluating perception module...")
    
    concept_activations = []
    confidence_scores = []
    
    for _ in tqdm(range(num_samples)):
        x = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            perception = model.perceive(x, threshold=0.5)
        
        symbolic = perception["symbolic"][0]
        if symbolic:
            concepts, confidences = zip(*symbolic)
            concept_activations.append(len(concepts))
            confidence_scores.extend(confidences)
    
    stats = {
        "concepts_per_image": {
            "mean": np.mean(concept_activations),
            "std": np.std(concept_activations),
        },
        "confidence": {
            "mean": np.mean(confidence_scores),
            "std": np.std(confidence_scores),
        },
    }
    
    return stats


def benchmark_inference(model, device, num_iterations=100):
    """Benchmark inference speed."""
    model.eval()
    
    print("\nBenchmarking inference speed...")
    
    # Warmup
    for _ in range(10):
        x = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model.forward(x)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    import time
    times = []
    
    for _ in tqdm(range(num_iterations)):
        x = torch.randn(1, 3, 224, 224).to(device)
        
        start = time.time()
        with torch.no_grad():
            _ = model.forward(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        times.append(end - start)
    
    stats = {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "throughput_fps": 1.0 / np.mean(times),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate neurosymbolic system")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = NeurosymbolicSystem(
        perception_config={
            "backbone": "efficientnet_b0",
            "feature_dim": 512,
            "num_concepts": 100,
        }
    )
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Run evaluations
    results = {}
    
    results["reasoning"] = evaluate_reasoning(model, device, args.num_samples)
    results["perception"] = evaluate_perception(model, device, args.num_samples)
    results["inference_speed"] = benchmark_inference(model, device, 100)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2))
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()