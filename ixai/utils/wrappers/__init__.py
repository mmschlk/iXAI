"""
This modul gathers basic wrapper objects to transform common ML model architectures into callable functions.

Note: To decrease the dependency count the required wrappers should be imported directly.
"""

from .sklearn import SklearnWrapper
from .river import RiverWrapper, RiverMetricToLossFunction
from .torch import TorchSupervisedLearningWrapper, TorchWrapper

__all__ = [
    "TorchWrapper",
    "TorchSupervisedLearningWrapper",
    "RiverWrapper",
    "SklearnWrapper",
    "RiverMetricToLossFunction"
]
