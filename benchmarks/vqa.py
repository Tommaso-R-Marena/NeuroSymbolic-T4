"""VQA v2.0 benchmark for visual question answering.

Reference: Goyal et al., "Making the V in VQA Matter: Elevating the Role of 
Image Understanding in Visual Question Answering", CVPR 2017.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


class VQADataset(Dataset):
    """VQA v2.0 dataset."""
    
    def __init__(self, root: str, split: str = "val", year: str = "2014"):
        self.root = Path(root)
        self.split = split
        self.year = year
        
        # Paths
        self.image_dir = self.root / "images" / f"{split}{year}"
        self.questions_file = self.root / f"v2_OpenEnded_mscoco_{split}{year}_questions.json"
        self.annotations_file = self.root / f"v2_mscoco_{split}{year}_annotations.json"
        
        # Load data
        with open(self.questions_file) as f:
            self.questions = json.load(f)["questions"]
        
        with open(self.annotations_file) as f:
            annotations = json.load(f)["annotations"]
            self.annotations = {ann["question_id"]: ann for ann in annotations}
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        ann = self.annotations[q["question_id"]]
        
        # Load image
        image_path = self.image_dir / f"COCO_{self.split}{self.year}_{q['image_id']:012d}.jpg"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        return {
            "image": image,
            "question": q["question"],
            "question_id": q["question_id"],
            "answers": [a["answer"] for a in ann["answers"]],
            "question_type": ann["question_type"],
            "answer_type": ann["answer_type"],
        }


class VQABenchmark:
    """VQA benchmark evaluation."""
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.eval()
        
    def evaluate(self, dataset: VQADataset, batch_size: int = 32) -> Dict[str, float]:
        """Evaluate on VQA."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        results = {
            "total": 0,
            "perception_concepts": [],
            "reasoning_facts": [],
            "by_question_type": {},
            "by_answer_type": {},
        }
        
        print("Evaluating on VQA v2.0...")
        for batch in tqdm(dataloader):
            images = batch["image"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.forward(images, threshold=0.5)
                
            for i in range(len(images)):
                perception = outputs["perception"]["symbolic"][i]
                reasoning = outputs["reasoning"][i]
                
                results["perception_concepts"].append(len(perception))
                results["reasoning_facts"].append(reasoning["num_derived"])
                results["total"] += 1
        
        metrics = {
            "avg_concepts_detected": np.mean(results["perception_concepts"]),
            "avg_facts_derived": np.mean(results["reasoning_facts"]),
            "total_evaluated": results["total"],
        }
        
        return metrics