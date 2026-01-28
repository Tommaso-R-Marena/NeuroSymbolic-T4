"""Training script for neurosymbolic system."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from neurosymbolic import NeurosymbolicSystem


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for proof of concept."""
    
    def __init__(self, size: int = 1000, image_size: int = 224):
        self.size = size
        self.image_size = image_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random image
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Generate random concept labels (multi-label)
        num_concepts = 100
        concepts = torch.zeros(num_concepts)
        num_active = np.random.randint(1, 6)
        active_indices = np.random.choice(num_concepts, num_active, replace=False)
        concepts[active_indices] = 1.0
        
        return image, concepts


def train_epoch(model, dataloader, optimizer, scaler, device, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, concepts in pbar:
        images = images.to(device)
        concepts = concepts.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model.perception(images)
            pred_concepts = outputs["concepts"]
            
            # Multi-label classification loss
            loss = criterion(pred_concepts, concepts)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    for images, concepts in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        concepts = concepts.to(device)
        
        outputs = model.perception(images)
        pred_concepts = outputs["concepts"]
        
        loss = criterion(pred_concepts, concepts)
        
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        all_preds.append(pred_concepts.cpu())
        all_targets.append(concepts.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    threshold = 0.5
    pred_binary = (all_preds > threshold).float()
    accuracy = (pred_binary == all_targets).float().mean().item()
    
    return total_loss / total_samples, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train neurosymbolic system")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    print("Initializing model...")
    model = NeurosymbolicSystem(
        perception_config={
            "backbone": "efficientnet_b0",
            "feature_dim": 512,
            "num_concepts": 100,
        }
    )
    model = model.to(device)
    
    # Dataset
    print("Loading dataset...")
    train_dataset = SyntheticDataset(size=2000)
    val_dataset = SyntheticDataset(size=500)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.BCELoss()
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, criterion)
        
        # Validate
        val_loss, val_accuracy = evaluate(model, val_loader, device, criterion)
        
        # Log
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    
    # Save history
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()