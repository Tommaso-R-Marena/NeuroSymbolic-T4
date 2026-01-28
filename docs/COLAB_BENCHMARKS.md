# Running ICML Benchmarks in Google Colab

This guide shows how to run publication-ready benchmarks directly in Google Colab.

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/NeuroSymbolic-T4/blob/main/notebooks/NeuroSymbolic_T4_Demo.ipynb)

## Setup

```python
# 1. Verify T4 GPU
!nvidia-smi

# 2. Clone and install
!git clone https://github.com/Tommaso-R-Marena/NeuroSymbolic-T4.git
%cd NeuroSymbolic-T4
!pip install -q -r requirements.txt
```

## Synthetic Benchmarking

For quick testing without downloading large datasets:

```python
import torch
from neurosymbolic import NeurosymbolicSystem
from benchmarks.metrics import NeurosymbolicMetrics
import numpy as np

# Initialize
device = torch.device("cuda")
model = NeurosymbolicSystem().to(device)
model.eval()

# Generate synthetic test batch
test_images = torch.randn(100, 3, 224, 224).to(device)

# Evaluate
perception_concepts = []
reasoning_depths = []
proof_confidences = []

with torch.no_grad():
    for i in range(len(test_images)):
        img = test_images[i:i+1]
        output = model.forward(img, threshold=0.5)
        
        concepts = len(output["perception"]["symbolic"][0])
        derived = output["reasoning"][0]["num_derived"]
        
        perception_concepts.append(concepts)
        reasoning_depths.append(derived)

# Compute metrics
metrics = NeurosymbolicMetrics()
print(f"Average concepts detected: {np.mean(perception_concepts):.2f}")
print(f"Average reasoning depth: {np.mean(reasoning_depths):.2f}")
print(f"Reasoning depth std: {np.std(reasoning_depths):.2f}")
```

## Downloading Datasets (Optional)

### CLEVR Dataset (~18 GB)

```bash
# In Colab cell
!wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
!unzip -q CLEVR_v1.0.zip -d /content/data/
!ls /content/data/CLEVR_v1.0
```

**Note**: This takes ~10-15 minutes to download and extract.

### Running CLEVR Benchmark

```python
from benchmarks import CLEVRBenchmark
from benchmarks.clevr import CLEVRDataset

# Load dataset (use small subset for testing)
dataset = CLEVRDataset("/content/data/CLEVR_v1.0", split="val")

# Take small subset for Colab
small_dataset = torch.utils.data.Subset(dataset, range(100))

# Benchmark
benchmark = CLEVRBenchmark(model, device="cuda")
results = benchmark.evaluate(small_dataset, batch_size=16)

print("\nCLEVR Results (100 samples):")
for key, value in results.items():
    print(f"  {key}: {value}")
```

## Performance Benchmarking

### Latency and Throughput

```python
import time

# Warmup
for _ in range(10):
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        _ = model.forward(x)

# Benchmark
torch.cuda.synchronize()
times = []

for _ in range(100):
    x = torch.randn(1, 3, 224, 224).to(device)
    
    start = time.time()
    with torch.no_grad():
        _ = model.forward(x)
    torch.cuda.synchronize()
    end = time.time()
    
    times.append(end - start)

print(f"Mean latency: {np.mean(times)*1000:.2f} ms")
print(f"Std latency: {np.std(times)*1000:.2f} ms")
print(f"Throughput: {1/np.mean(times):.1f} FPS")
print(f"Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

### Batch Processing

```python
batch_sizes = [1, 4, 8, 16, 32]
results = {}

for bs in batch_sizes:
    times = []
    
    # Warmup
    for _ in range(5):
        x = torch.randn(bs, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model.forward(x)
    
    # Benchmark
    torch.cuda.synchronize()
    for _ in range(20):
        x = torch.randn(bs, 3, 224, 224).to(device)
        
        start = time.time()
        with torch.no_grad():
            _ = model.forward(x)
        torch.cuda.synchronize()
        end = time.time()
        
        times.append((end - start) / bs)
    
    results[bs] = {
        "latency_ms": np.mean(times) * 1000,
        "throughput_fps": bs / np.mean(times),
    }
    
    print(f"Batch size {bs:2d}: {results[bs]['latency_ms']:.2f} ms/sample, {results[bs]['throughput_fps']:.1f} FPS")
```

## Reasoning Analysis

### Distribution of Reasoning Depth

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Collect reasoning depths
depths = []

for _ in range(200):
    img = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model.forward(img, threshold=0.5)
    depths.append(output["reasoning"][0]["num_derived"])

# Plot
plt.figure(figsize=(8, 5))
plt.hist(depths, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Reasoning Depth (# Derived Facts)")
plt.ylabel("Frequency")
plt.title("Distribution of Reasoning Depth")
plt.tight_layout()
plt.show()

print(f"Mean: {np.mean(depths):.2f}")
print(f"Std: {np.std(depths):.2f}")
print(f"Median: {np.median(depths):.0f}")
print(f"Max: {np.max(depths)}")
```

### Proof Generation Analysis

```python
# Analyze proof quality
proof_lengths = []
proof_confidences = []

for _ in range(100):
    img = torch.randn(1, 3, 224, 224).to(device)
    query = ("dangerous", ("obj0",))
    
    with torch.no_grad():
        proofs = model.query(img, query, threshold=0.4)
    
    if proofs:
        proof_lengths.append(len(proofs[0]["proof"]))
        proof_confidences.append(proofs[0]["confidence"])

if proof_lengths:
    print(f"Average proof length: {np.mean(proof_lengths):.2f}")
    print(f"Average proof confidence: {np.mean(proof_confidences):.3f}")
    print(f"Proof success rate: {len(proof_lengths)}/100")
else:
    print("No proofs found (try different queries or lower threshold)")
```

## Saving Results

```python
import json

# Compile results
final_results = {
    "model": "NeuroSymbolic-T4",
    "device": "Tesla T4 (Google Colab)",
    "metrics": {
        "avg_concepts": float(np.mean(perception_concepts)),
        "avg_reasoning_depth": float(np.mean(reasoning_depths)),
        "reasoning_depth_std": float(np.std(reasoning_depths)),
        "mean_latency_ms": float(np.mean(times) * 1000),
        "throughput_fps": float(1 / np.mean(times)),
    }
}

# Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

with open('/content/drive/MyDrive/neurosymbolic_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("✅ Results saved to Google Drive")
```

## Training in Colab

```python
# Quick training run (10 epochs)
!python train_benchmarks.py \
    --batch-size 16 \
    --epochs 10 \
    --use-amp \
    --output-dir /content/checkpoints
```

**Note**: For full 30-epoch training, consider using Colab Pro for longer session times.

## Tips for Colab

1. **GPU Runtime**: Always select "T4 GPU" runtime
2. **Session Management**: Colab disconnects after ~12 hours
3. **Checkpointing**: Save checkpoints to Google Drive
4. **Memory**: Monitor with `torch.cuda.memory_summary()`
5. **Data**: Keep datasets on Google Drive for persistence

## Full Pipeline Example

```python
# Complete ICML benchmark pipeline

# 1. Setup
import torch
import numpy as np
from neurosymbolic import NeurosymbolicSystem
from benchmarks.metrics import NeurosymbolicMetrics

device = torch.device("cuda")
model = NeurosymbolicSystem().to(device)
model.eval()

# 2. Generate test data
test_images = torch.randn(500, 3, 224, 224).to(device)

# 3. Evaluate
metrics_data = {
    "perception": [],
    "reasoning": [],
    "proofs": [],
}

for i in range(len(test_images)):
    img = test_images[i:i+1]
    
    with torch.no_grad():
        output = model.forward(img, threshold=0.5)
        proofs = model.query(img, ("dangerous", ("obj0",)), threshold=0.4)
    
    metrics_data["perception"].append(len(output["perception"]["symbolic"][0]))
    metrics_data["reasoning"].append(output["reasoning"][0]["num_derived"])
    if proofs:
        metrics_data["proofs"].append(proofs[0]["confidence"])

# 4. Compute metrics
metrics = NeurosymbolicMetrics()

results = {
    "perception": {
        "mean": np.mean(metrics_data["perception"]),
        "std": np.std(metrics_data["perception"]),
    },
    "reasoning": metrics.reasoning_depth(metrics_data["reasoning"]),
    "proofs": {
        "mean_confidence": np.mean(metrics_data["proofs"]) if metrics_data["proofs"] else 0,
        "success_rate": len(metrics_data["proofs"]) / len(test_images),
    }
}

# 5. Display
print("\n" + "="*60)
print("NEUROSYMBOLIC-T4 BENCHMARK RESULTS")
print("="*60)
for category, values in results.items():
    print(f"\n{category.upper()}:")
    for metric, value in values.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
print("="*60)
```