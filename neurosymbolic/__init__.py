"""NeuroSymbolic-T4: State-of-the-art neurosymbolic AI system for T4 GPUs."""

__version__ = "0.1.0"

import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from .neural import PerceptionModule
from .symbolic import SymbolicReasoner
from .integration import NeurosymbolicSystem

__all__ = [
    "PerceptionModule",
    "SymbolicReasoner",
    "NeurosymbolicSystem",
    "set_seed",
]
