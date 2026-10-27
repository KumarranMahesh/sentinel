"""Streaming detectors: D1 residual, D2 change, D3 shape, D4 local outlier factor,
D5 distribution shift, D6 joint (Hotelling T^2), D7 subspace (SSA), D8 forecast
(online AR), D9 neural forecaster. D3-D9 optional."""
from .change import ChangeDetector
from .forecast import ForecastDetector
from .joint import JointDetector
from .lof import LofDetector
from .neural import NeuralDetector
from .residual import ResidualDetector
from .shape import ShapeDetector
from .shift import ShiftDetector
from .subspace import SubspaceDetector

ORDER = ["residual", "change", "shape", "lof", "shift", "joint", "subspace", "forecast", "neural"]
