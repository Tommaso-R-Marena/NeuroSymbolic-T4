"""Neural perception module with T4-optimized architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import timm


class AttentionPool(nn.Module):
    """Attention-based pooling for feature aggregation."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)
        out, _ = self.attention(query, x, x)
        return out.squeeze(1)


class PerceptionModule(nn.Module):
    """Neural perception module for feature extraction.
    
    Optimized for T4 GPU with mixed precision and efficient architectures.
    Extracts high-level features and symbolic groundings from raw inputs.
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        feature_dim: int = 512,
        num_concepts: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Efficient backbone pre-trained on ImageNet
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool=""
        )
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[1]
        
        # Attention pooling for spatial features
        self.attention_pool = AttentionPool(backbone_dim)
        
        # Project to feature dimension
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Concept grounding head
        self.concept_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_concepts),
        )
        
        # Object detection head (for spatial reasoning)
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # x, y, w, h
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_spatial: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass extracting features and symbolic groundings.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_spatial: Whether to return spatial feature maps
            
        Returns:
            Dictionary containing:
                - features: Global feature vectors [B, feature_dim]
                - concepts: Concept probabilities [B, num_concepts]
                - bboxes: Bounding boxes [B, 4]
                - confidence: Detection confidence [B, 1]
        """
        # Extract spatial features
        spatial_features = self.backbone(x)  # [B, C, H, W]
        B, C, H, W = spatial_features.shape
        
        # Reshape for attention: [B, H*W, C]
        spatial_flat = spatial_features.view(B, C, H * W).transpose(1, 2)
        
        # Attention pooling
        pooled = self.attention_pool(spatial_flat)  # [B, C]
        
        # Project to feature space
        features = self.feature_proj(pooled)  # [B, feature_dim]
        
        # Concept grounding
        concepts = torch.sigmoid(self.concept_head(features))
        
        # Object detection (simplified)
        bboxes = torch.sigmoid(self.bbox_head(features))
        confidence = torch.sigmoid(self.confidence_head(features))
        
        output = {
            "features": features,
            "concepts": concepts,
            "bboxes": bboxes,
            "confidence": confidence,
        }
        
        if return_spatial:
            output["spatial_features"] = spatial_features
            
        return output
    
    def extract_symbolic_scene(self, x: torch.Tensor, threshold: float = 0.5) -> List[Dict]:
        """Extract symbolic scene representation from image.
        
        Args:
            x: Input image tensor
            threshold: Confidence threshold for concept detection
            
        Returns:
            List of detected objects with symbolic attributes
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
        concepts = outputs["concepts"].cpu().numpy()
        bboxes = outputs["bboxes"].cpu().numpy()
        confidence = outputs["confidence"].cpu().numpy()
        
        scene = []
        for i in range(len(x)):
            detected_concepts = (concepts[i] > threshold).nonzero()[0]
            if len(detected_concepts) > 0 and confidence[i, 0] > threshold:
                scene.append({
                    "concepts": detected_concepts.tolist(),
                    "bbox": bboxes[i].tolist(),
                    "confidence": float(confidence[i, 0]),
                })
        
        return scene