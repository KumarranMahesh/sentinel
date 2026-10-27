"""Streaming detectors: D1 residual, D2 change, D3 shape, D4 local outlier factor (optional)."""
from .change import ChangeDetector
from .residual import ResidualDetector
from .lof import LofDetector
from .shape import ShapeDetector

ORDER = ["residual", "change", "shape", "lof"]
