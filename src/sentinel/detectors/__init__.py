"""The three cheap streaming detectors: D1 residual, D2 change, D3 shape."""
from .change import ChangeDetector
from .residual import ResidualDetector
from .shape import ShapeDetector

ORDER = ["residual", "change", "shape"]
