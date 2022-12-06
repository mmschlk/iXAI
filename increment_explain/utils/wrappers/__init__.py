"""
This modul gathers basic wrapper objects to transform common ML model architectures into callable functions.

Note: To decrease the dependency count the required wrappers should be imported directly.
"""

from .generic import GenericWrapper
from .river import RiverPredictionFunctionWrapper
from .torch import TorchSupervisedLearningWrapper

__all__ = [
    "TorchSupervisedLearningWrapper",
    "RiverPredictionFunctionWrapper",
    "GenericWrapper"
]
