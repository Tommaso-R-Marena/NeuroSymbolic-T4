"""Baseline models for comparison."""

import torch
import torch.nn as nn
from typing import Dict, Any


class NeuralOnlyBaseline(nn.Module):
    """Pure neural baseline (no symbolic reasoning)."""
    
    def __init__(self, backbone="efficientnet_b0", num_classes=100):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class SymbolicOnlyBaseline:
    """Pure symbolic baseline (no neural perception)."""
    
    def __init__(self):
        from neurosymbolic.symbolic import SymbolicReasoner
        self.reasoner = SymbolicReasoner()
        self._init_rules()
    
    def _init_rules(self):
        # Add common rules
        self.reasoner.add_rule(
            head=("vehicle", ("?x",)),
            body=[("car", ("?x",))],
            confidence=1.0
        )
    
    def forward(self, concepts: list):
        # Manual concept input
        for concept, conf in concepts:
            self.reasoner.add_fact(concept, ("obj1",), conf)
        
        self.reasoner.forward_chain()
        return {"facts": list(self.reasoner.facts)}


class VisionTransformerBaseline(nn.Module):
    """Vision Transformer baseline."""
    
    def __init__(self, model_name="vit_base_patch16_224", num_classes=100):
        super().__init__()
        import timm
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class CLIPBaseline(nn.Module):
    """CLIP baseline for vision-language tasks."""
    
    def __init__(self):
        super().__init__()
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device="cpu")
        except:
            print("Warning: CLIP not available. Using dummy model.")
            self.model = None
    
    def forward(self, x, text_queries=None):
        if self.model is None:
            return torch.randn(x.size(0), 512)
        
        with torch.no_grad():
            features = self.model.encode_image(x)
        return features


class NeuralModuleNetwork(nn.Module):
    """Neural Module Networks baseline.
    
    Reference: Andreas et al., CVPR 2016
    """
    
    def __init__(self, num_modules=10):
        super().__init__()
        import timm
        self.feature_extractor = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        
        # Simple module bank
        self.modules = nn.ModuleDict({
            "find": nn.Linear(1280, 256),
            "filter": nn.Linear(256, 256),
            "relate": nn.Linear(256, 256),
            "query": nn.Linear(256, 100),
        })
    
    def forward(self, x, program=None):
        features = self.feature_extractor(x)
        
        # Execute simple program
        out = features
        if program:
            for module_name in program:
                if module_name in self.modules:
                    out = self.modules[module_name](out)
        
        return out