"""Benchmark suite for ICML submission."""

from .clevr import CLEVRBenchmark
from .gqa import GQABenchmark
from .kandinsky import KandinskyBenchmark
from .arc import ARCBenchmark
from .vqa import VQAv2Benchmark

__all__ = [
    "CLEVRBenchmark",
    "GQABenchmark",
    "KandinskyBenchmark",
    "ARCBenchmark",
    "VQAv2Benchmark",
]