"""
This module contains Imputer Mechanisms for Efficient Feature Removal.
"""
from .default_imputer import DefaultImputer
from .marginal_imputer import MarginalImputer
from .base import BaseImputer
from .tree_imputer import TreeImputer

__all__ = [
    "DefaultImputer",
    "MarginalImputer",
    "BaseImputer",
    "TreeImputer"
]
