"""Benchmark suite for ICML submission."""

from .clevr import CLEVRBenchmark
from .vqa import VQABenchmark
from .gqa import GQABenchmark
from .kandinsky import KandinskyBenchmark
from .arc import ARCBenchmark
from .metrics import NeurosymbolicMetrics

# Aliases for ICML Benchmarks notebook
VQAv2Benchmark = VQABenchmark
ReasoningMetrics = NeurosymbolicMetrics

__all__ = [
    "CLEVRBenchmark",
    "VQABenchmark",
    "VQAv2Benchmark",
    "GQABenchmark",
    "KandinskyBenchmark",
    "ARCBenchmark",
    "NeurosymbolicMetrics",
    "ReasoningMetrics",
]