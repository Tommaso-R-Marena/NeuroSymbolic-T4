"""CLEVR benchmark for compositional visual reasoning.

CLEVR (Compositional Language and Elementary Visual Reasoning) is the standard
benchmark for testing compositional reasoning in vision-language systems.

Reference: Johnson et al., "CLEVR: A Diagnostic Dataset for Compositional 
Language and Elementary Visual Reasoning", CVPR 2017.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


class CLEVRDataset(Dataset):
    """CLEVR dataset for visual reasoning."""
    
    def __init__(self, root: str, split: str = "val", download: bool = True):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / "images" / split
        self.questions_file = self.root / "questions" / f"CLEVR_{split}_questions.json"
        
        if download and not self.questions_file.exists():
            self._download()
        
        # Load questions
        with open(self.questions_file) as f:
            data = json.load(f)
            self.questions = data["questions"]
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Build answer vocabulary
        self.answer_vocab = self._build_answer_vocab()
        
    def _download(self):
        """Download CLEVR dataset."""
        print("CLEVR dataset not found. Please download from:")
        print("https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip")
        print(f"Extract to: {self.root}")
        raise FileNotFoundError("CLEVR dataset not found")
    
    def _build_answer_vocab(self) -> Dict[str, int]:
        """Build answer vocabulary."""
        answers = set()
        for q in self.questions:
            answers.add(q["answer"])
        return {ans: idx for idx, ans in enumerate(sorted(answers))}
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question_data = self.questions[idx]
        
        # Load image
        image_filename = question_data["image_filename"]
        image_path = self.image_dir / image_filename
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Parse question and answer
        question = question_data["question"]
        answer = question_data["answer"]
        answer_idx = self.answer_vocab.get(answer, 0)
        
        # Extract program (functional program for question)
        program = question_data.get("program", [])
        
        return {
            "image": image,
            "question": question,
            "answer": answer,
            "answer_idx": answer_idx,
            "program": program,
            "image_idx": question_data["image_index"],
        }


class CLEVRBenchmark:
    """CLEVR benchmark evaluation."""
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.eval()
        
    def evaluate(self, dataset: CLEVRDataset, batch_size: int = 32) -> Dict[str, float]:
        """Evaluate on CLEVR."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        results = {
            "total": 0,
            "correct": 0,
            "by_question_type": {},
            "reasoning_depth": [],
        }
        
        print("Evaluating on CLEVR...")
        for batch in tqdm(dataloader):
            images = batch["image"].to(self.device)
            
            with torch.no_grad():
                # Get neurosymbolic predictions
                outputs = self.model.forward(images, threshold=0.5)
                
                # Extract reasoning results
                for i in range(len(images)):
                    perception = outputs["perception"]["symbolic"][i]
                    reasoning = outputs["reasoning"][i]
                    
                    # Reasoning depth = number of derived facts
                    depth = reasoning["num_derived"]
                    results["reasoning_depth"].append(depth)
                    
                    # For CLEVR, we'd need to map to answers
                    # This is simplified - full implementation would use question parser
                    results["total"] += 1
        
        # Compute metrics
        metrics = {
            "accuracy": results["correct"] / results["total"] if results["total"] > 0 else 0,
            "avg_reasoning_depth": np.mean(results["reasoning_depth"]),
            "total_evaluated": results["total"],
        }
        
        return metrics
    
    def analyze_reasoning(self, dataset: CLEVRDataset, num_samples: int = 100) -> Dict:
        """Analyze reasoning capabilities."""
        analysis = {
            "compositional_generalization": [],
            "spatial_reasoning": [],
            "counting": [],
            "comparison": [],
        }
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            image = batch["image"].to(self.device)
            program = batch["program"][0]
            
            with torch.no_grad():
                output = self.model.forward(image, threshold=0.5)
                
            # Analyze by question type
            if any("count" in str(op) for op in program):
                analysis["counting"].append(output["reasoning"][0]["num_derived"])
            elif any("relate" in str(op) or "spatial" in str(op) for op in program):
                analysis["spatial_reasoning"].append(output["reasoning"][0]["num_derived"])
        
        return {
            k: {"mean": np.mean(v) if v else 0, "std": np.std(v) if v else 0}
            for k, v in analysis.items()
        }