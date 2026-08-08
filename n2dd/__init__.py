"""Diffusion-based inverse design of bimetallic N2-activation catalysts.

Repaired and productionised from the supporting information of ja5c14652.
See README.md for the list of defects found in the published code.
"""
from .config import Config
from .data import load, Dataset
from .forward import load_rf, evaluate_rf, train_surrogate
from .diffusion import DiffusionModel, DiffusionSchedule
from .decode import AlloyDecoder
from . import screen

__all__ = ["Config", "load", "Dataset", "load_rf", "evaluate_rf", "train_surrogate",
           "DiffusionModel", "DiffusionSchedule", "AlloyDecoder", "screen"]
__version__ = "1.0.0"
