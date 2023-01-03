"""
This modul gathers basic wrapper objects to transform common ML model architectures into callable
functions.

Note: To decrease the dependency count the required wrappers should be imported directly.
"""

from ixai.utils.wrappers.sklearn import SklearnWrapper
from ixai.utils.wrappers.river import RiverWrapper, RiverMetricToLossFunction
from ixai.utils.wrappers.torch import TorchSupervisedLearningWrapper, TorchWrapper

__all__ = [
    "TorchWrapper",
    "TorchSupervisedLearningWrapper",
    "RiverWrapper",
    "SklearnWrapper",
    "RiverMetricToLossFunction"
]
