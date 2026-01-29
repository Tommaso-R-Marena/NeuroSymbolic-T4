# Complete Training Guide

## 🚀 Quick Start: Train in Google Colab

**The fastest way to train NeuroSymbolic-T4 is using our pre-configured Colab notebook:**

[![Open Training Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/NeuroSymbolic-T4/blob/main/notebooks/NeuroSymbolic_T4_Training.ipynb)

**Features:**
- ✅ Automatic dataset download (CLEVR mini)
- ✅ Complete training loop with progress tracking
- ✅ Mixed precision training (AMP)
- ✅ Performance benchmarking
- ✅ Results exported to Google Drive
- ✅ Takes ~30-45 minutes on T4 GPU

---

## 📦 Local Training

### Step 1: Install Dependencies

```bash
git clone https://github.com/Tommaso-R-Marena/NeuroSymbolic-T4.git
cd NeuroSymbolic-T4
pip install -r requirements.txt
pip install wandb  # Optional: for experiment tracking
```

### Step 2: Download Datasets

#### Option A: Automatic Download (Recommended)

```bash
# Download CLEVR mini (~1.5GB, 10k train + 1k val)
python benchmarks/download_datasets.py --dataset clevr_mini --data-root ./data

# Or download full CLEVR (~18GB, 70k train + 15k val)
python benchmarks/download_datasets.py --dataset clevr --data-root ./data

# Or download all datasets
python benchmarks/download_datasets.py --dataset all --data-root ./data
```

**Available datasets:**
- `clevr_mini` - CLEVR subset (1.5GB) - **Recommended for quick experiments**
- `clevr` - Full CLEVR v1.0 (18GB)
- `vqa` - VQA v2.0 (25GB)
- `gqa` - GQA (20GB)
- `all` - All datasets (~63GB)

#### Option B: Manual Download

**CLEVR:**
```bash
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip -d ./data/
```

**VQA v2.0:** Visit [visualqa.org/download.html](https://visualqa.org/download.html)

**GQA:** Visit [cs.stanford.edu/people/dorarad/gqa/download.html](https://cs.stanford.edu/people/dorarad/gqa/download.html)

### Step 3: Start Training

#### Basic Training (CLEVR Mini)

```bash
python train_benchmarks.py \
    --dataset clevr \
    --clevr-root ./data/CLEVR_mini \
    --batch-size 32 \
    --epochs 20 \
    --lr 1e-3 \
    --use-amp
```

**Expected results:**
- Training time: ~30-45 minutes on T4 GPU
- Final val loss: ~0.4-0.6
- Inference speed: ~40-50 FPS

#### Advanced Training (Full CLEVR)

```bash
python train_benchmarks.py \
    --dataset clevr \
    --clevr-root ./data/CLEVR_v1.0 \
    --batch-size 32 \
    --epochs 30 \
    --lr 1e-3 \
    --warmup-epochs 3 \
    --use-amp \
    --curriculum \
    --use-wandb \
    --output-dir ./checkpoints_clevr
```

**Expected results:**
- Training time: ~4-6 hours on T4 GPU
- Final val loss: ~0.3-0.5
- Reasoning depth: ~3-4 facts per query

#### Multi-Dataset Training

```bash
python train_benchmarks.py \
    --dataset all \
    --clevr-root ./data/CLEVR_v1.0 \
    --vqa-root ./data/VQA \
    --gqa-root ./data/GQA \
    --batch-size 32 \
    --epochs 30 \
    --use-amp \
    --curriculum \
    --use-wandb
```

### Step 4: Monitor Training

#### Option A: WandB (Recommended)

```bash
# Login to WandB
wandb login

# Train with WandB logging
python train_benchmarks.py --use-wandb --wandb-project neurosymbolic-icml
```

#### Option B: Local Monitoring

```bash
# Training history saved to
cat checkpoints/training_history.json

# Visualize later
python scripts/plot_training.py --history checkpoints/training_history.json
```

---

## 🎛️ Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 32 | Training batch size |
| `--epochs` | 30 | Number of epochs |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 0.01 | AdamW weight decay |
| `--warmup-epochs` | 3 | LR warmup epochs |
| `--use-amp` | True | Mixed precision training |
| `--curriculum` | False | Curriculum learning |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backbone` | efficientnet_b0 | Vision backbone |
| `--feature-dim` | 512 | Feature dimension |
| `--num-concepts` | 100 | Number of concepts |

### System Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-workers` | 4 | DataLoader workers |
| `--device` | cuda | Training device |
| `--seed` | 42 | Random seed |
| `--output-dir` | checkpoints_benchmark | Output directory |

---

## 📊 Evaluation

### After Training

```bash
# Evaluate best checkpoint
python benchmarks/run_all.py \
    --checkpoint checkpoints/best_model.pt \
    --clevr-root ./data/CLEVR_v1.0 \
    --output-dir ./results
```

### Generate Paper Figures

```bash
# Create publication-ready visualizations
python paper/prepare_figures.py \
    --results-dir ./results \
    --output-dir ./paper/figures
```

Generates:
- `reasoning_depth.pdf` - Reasoning depth distribution
- `performance_comparison.pdf` - Model comparison
- `ablation_study.pdf` - Component analysis
- `training_curves.pdf` - Loss and metrics over time

---

## 🔧 Advanced Features

### 1. Curriculum Learning

```bash
python train_benchmarks.py --curriculum --epochs 30
```

Gradually increases task difficulty during training.

### 2. Multi-GPU Training

```bash
# Distributed training (coming soon)
python train_benchmarks.py --distributed --num-gpus 4
```

### 3. Resume from Checkpoint

```bash
python train_benchmarks.py --resume checkpoints/checkpoint_epoch_10.pt
```

### 4. Custom Learning Rate Schedule

```bash
# Cosine annealing (default)
python train_benchmarks.py --scheduler cosine

# Step decay
python train_benchmarks.py --scheduler step

# Reduce on plateau
python train_benchmarks.py --scheduler plateau
```

---

## 📈 Expected Performance

### Training Metrics (CLEVR)

| Epoch | Train Loss | Val Loss | Concepts | Facts Derived |
|-------|-----------|----------|----------|---------------|
| 5 | 0.82 | 0.78 | 2.3 | 1.2 |
| 10 | 0.61 | 0.58 | 3.1 | 2.1 |
| 15 | 0.52 | 0.50 | 3.8 | 2.8 |
| 20 | 0.46 | 0.45 | 4.2 | 3.2 |
| 30 | 0.39 | 0.41 | 4.6 | 3.5 |

### Final Performance (T4 GPU)

| Metric | NeuroSymbolic-T4 | Target |
|--------|------------------|--------|
| **Accuracy** | 73-75% | 75%+ |
| **Reasoning Depth** | 3.2±1.1 | 3.0+ |
| **Inference Speed** | 45 FPS | 40+ FPS |
| **Memory Usage** | 7.2 GB | <8 GB |

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python train_benchmarks.py --batch-size 16

# Or disable AMP
python train_benchmarks.py --batch-size 32 --no-use-amp
```

### Slow Training

```bash
# Increase workers
python train_benchmarks.py --num-workers 8

# Check GPU utilization
watch -n 1 nvidia-smi
```

### NaN Loss

```bash
# Reduce learning rate
python train_benchmarks.py --lr 5e-4

# Add gradient clipping (already enabled by default)
```

### Dataset Not Found

```bash
# Use synthetic data for testing
python train_benchmarks.py --dataset clevr --clevr-root ./nonexistent
# Falls back to synthetic dataset automatically
```

---

## 💡 Tips for Best Results

### 1. Start Small
- Use `clevr_mini` for initial experiments
- Verify everything works before full training
- Typical experiment: 20 epochs, ~45 minutes

### 2. Use Mixed Precision
- Always enable `--use-amp` on T4 GPUs
- 1.5-2x faster training
- Same accuracy as FP32

### 3. Monitor with WandB
- Real-time loss curves
- Compare multiple runs
- Track system metrics

### 4. Save Frequently
```bash
--save-interval 5  # Save every 5 epochs
```

### 5. Hyperparameter Search
```bash
# Try different learning rates
for lr in 1e-3 5e-4 1e-4; do
    python train_benchmarks.py --lr $lr --output-dir checkpoints_lr_$lr
done
```

---

## 📝 Checkpoint Management

### Checkpoint Contents

```python
checkpoint = {
    'epoch': 20,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'scheduler_state_dict': {...},
    'val_metrics': {
        'val_loss': 0.45,
        'avg_concepts': 4.2,
        'avg_facts_derived': 3.2,
        ...
    },
    'args': {...}
}
```

### Load Checkpoint

```python
import torch
from neurosymbolic import NeurosymbolicSystem

model = NeurosymbolicSystem().cuda()
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded epoch {checkpoint['epoch']}")
```

---

## 🎯 Next Steps

1. **Train on CLEVR mini**: Quick validation (~45 min)
2. **Full CLEVR training**: Production model (~6 hours)
3. **Multi-dataset**: CLEVR + VQA + GQA (~12 hours)
4. **Ablation studies**: Test component contributions
5. **Paper experiments**: Generate all figures

---

## 📚 Additional Resources

- **Demo Notebook**: [NeuroSymbolic_T4_Demo.ipynb](./notebooks/NeuroSymbolic_T4_Demo.ipynb)
- **Training Notebook**: [NeuroSymbolic_T4_Training.ipynb](./notebooks/NeuroSymbolic_T4_Training.ipynb)
- **API Documentation**: [docs/API.md](./docs/API.md)
- **Architecture Details**: [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)

---

## 🤝 Getting Help

- **Issues**: [GitHub Issues](https://github.com/Tommaso-R-Marena/NeuroSymbolic-T4/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Tommaso-R-Marena/NeuroSymbolic-T4/discussions)
- **Email**: marena@cua.edu

---

## 📖 Citation

If you use this training pipeline in your research:

```bibtex
@inproceedings{marena2026neurosymbolic,
  title={NeuroSymbolic-T4: Efficient Compositional Visual Reasoning with Explainable Inference},
  author={Marena, Tommaso R.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```