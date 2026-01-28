"""GQA benchmark for real-world visual reasoning.

Reference: Hudson & Manning, "GQA: A New Dataset for Real-World Visual 
Reasoning and Compositional Question Answering", CVPR 2019.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


class GQADataset(Dataset):
    """GQA dataset."""
    
    def __init__(self, root: str, split: str = "val"):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / "images"
        
        # Load questions
        questions_file = self.root / f"{split}_balanced_questions.json"
        with open(questions_file) as f:
            self.questions = json.load(f)
        
        self.question_ids = list(self.questions.keys())
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.question_ids)
    
    def __getitem__(self, idx):
        qid = self.question_ids[idx]
        q_data = self.questions[qid]
        
        # Load image
        image_path = self.image_dir / f"{q_data['imageId']}.jpg"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        return {
            "image": image,
            "question": q_data["question"],
            "answer": q_data["answer"],
            "semantic": q_data.get("semantic", []),
            "types": q_data.get("types", {}),
        }


class GQABenchmark:
    """GQA benchmark evaluation."""
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate(self, dataset: GQADataset, batch_size: int = 32) -> Dict[str, float]:
        """Evaluate on GQA."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        results = {
            "total": 0,
            "compositional_steps": [],
            "spatial_relations": [],
        }
        
        print("Evaluating on GQA...")
        for batch in tqdm(dataloader):
            images = batch["image"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.forward(images, threshold=0.5)
            
            for i in range(len(images)):
                reasoning = outputs["reasoning"][i]
                results["compositional_steps"].append(reasoning["num_derived"])
                results["total"] += 1
        
        return {
            "avg_compositional_steps": np.mean(results["compositional_steps"]),
            "total_evaluated": results["total"],
        }