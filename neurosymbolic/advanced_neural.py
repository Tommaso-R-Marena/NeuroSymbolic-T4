"""Advanced neural architectures for ICML submission."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class RelationNetwork(nn.Module):
    """Relation Network for relational reasoning.
    
    Reference: Santoro et al., NeurIPS 2017
    """
    
    def __init__(self, object_dim: int = 512, relation_dim: int = 256):
        super().__init__()
        
        self.g_theta = nn.Sequential(
            nn.Linear(object_dim * 2, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
        )
        
        self.f_phi = nn.Sequential(
            nn.Linear(relation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
    
    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        """Compute relations between all object pairs.
        
        Args:
            objects: [B, N, object_dim]
            
        Returns:
            relations: [B, 128]
        """
        B, N, D = objects.shape
        
        # Create all pairs
        obj_i = objects.unsqueeze(2).expand(B, N, N, D)
        obj_j = objects.unsqueeze(1).expand(B, N, N, D)
        pairs = torch.cat([obj_i, obj_j], dim=-1)  # [B, N, N, 2*D]
        
        # Compute relations for each pair
        relations = self.g_theta(pairs)  # [B, N, N, relation_dim]
        
        # Aggregate relations
        aggregated = relations.sum(dim=[1, 2])  # [B, relation_dim]
        
        # Final reasoning
        output = self.f_phi(aggregated)
        
        return output


class SymbolicAttention(nn.Module):
    """Attention mechanism guided by symbolic knowledge."""
    
    def __init__(self, dim: int, num_symbols: int = 100):
        super().__init__()
        self.dim = dim
        self.num_symbols = num_symbols
        
        self.symbol_embeddings = nn.Embedding(num_symbols, dim)
        self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, symbol_indices: torch.Tensor) -> torch.Tensor:
        """Apply symbol-guided attention.
        
        Args:
            features: [B, N, dim]
            symbol_indices: [B, K] indices of active symbols
            
        Returns:
            attended: [B, N, dim]
        """
        B = features.shape[0]
        
        # Get symbol embeddings
        symbol_embeds = self.symbol_embeddings(symbol_indices)  # [B, K, dim]
        
        # Attend to features using symbols as queries
        attended, _ = self.attention(symbol_embeds, features, features)
        
        # Broadcast to feature size
        attended_expanded = attended.mean(dim=1, keepdim=True).expand_as(features)
        
        # Gating mechanism
        combined = torch.cat([features, attended_expanded], dim=-1)
        gate_vals = self.gate(combined)
        
        output = gate_vals * features + (1 - gate_vals) * attended_expanded
        
        return output


class ProgramGenerator(nn.Module):
    """Generate executable programs for compositional reasoning.
    
    Inspired by Neural Module Networks.
    """
    
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 512, num_modules: int = 10):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.module_classifier = nn.Linear(hidden_dim, num_modules)
        
        self.modules = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_modules)
        ])
    
    def forward(self, question_tokens: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        """Generate and execute program.
        
        Args:
            question_tokens: [B, seq_len]
            visual_features: [B, feature_dim]
            
        Returns:
            output: [B, feature_dim]
        """
        # Encode question
        embedded = self.embedding(question_tokens)
        _, (hidden, _) = self.lstm(embedded)
        question_repr = hidden[-1]  # [B, hidden_dim]
        
        # Predict module sequence
        module_logits = self.module_classifier(question_repr)  # [B, num_modules]
        module_probs = F.softmax(module_logits, dim=-1)
        
        # Execute modules (soft execution)
        output = visual_features
        for i, module in enumerate(self.modules):
            module_output = module(output)
            # Weight by probability
            output = output + module_probs[:, i:i+1] * module_output
        
        return output


class NeuroSymbolicTransformer(nn.Module):
    """Transformer with symbolic reasoning integration."""
    
    def __init__(self, dim: int = 512, num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.symbolic_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, symbolic_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional symbolic context.
        
        Args:
            x: [B, N, dim]
            symbolic_context: [B, dim] optional symbolic information
            
        Returns:
            output: [B, N, dim]
        """
        for layer, gate in zip(self.layers, self.symbolic_gate):
            # Standard transformer layer
            x_transformed = layer(x)
            
            # Incorporate symbolic context if available
            if symbolic_context is not None:
                # Broadcast context
                context_expanded = symbolic_context.unsqueeze(1).expand_as(x)
                
                # Gate mechanism
                combined = torch.cat([x_transformed, context_expanded], dim=-1)
                gate_val = gate(combined)
                
                x = gate_val * x_transformed + (1 - gate_val) * context_expanded
            else:
                x = x_transformed
        
        return self.norm(x)


class HierarchicalReasoning(nn.Module):
    """Hierarchical reasoning module for multi-level abstraction."""
    
    def __init__(self, input_dim: int = 512, levels: int = 3):
        super().__init__()
        self.levels = levels
        
        self.level_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            )
            for _ in range(levels)
        ])
        
        self.level_pooling = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1)
            for _ in range(levels)
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * levels, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Hierarchical reasoning.
        
        Args:
            x: [B, N, dim]
            
        Returns:
            output: [B, dim]
            level_representations: List of representations at each level
        """
        level_reps = []
        
        current = x
        for encoder, pool in zip(self.level_encoders, self.level_pooling):
            # Process at this level
            processed = encoder(current)  # [B, N, dim]
            
            # Pool to next level
            pooled = pool(processed.transpose(1, 2)).squeeze(-1)  # [B, dim]
            level_reps.append(pooled)
            
            # Prepare for next level (reduce sequence length)
            if current.shape[1] > 1:
                current = processed[:, ::2, :]  # Subsample
        
        # Fuse all levels
        fused = torch.cat(level_reps, dim=-1)
        output = self.fusion(fused)
        
        return output, level_reps