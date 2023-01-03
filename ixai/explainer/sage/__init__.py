"""
This module gathers SAGE Explanation Methods
"""

from .batch import BatchSage
from .interval import IntervalSage
from .incremental import IncrementalSage

__all__ = [
    "IncrementalSage",
    "BatchSage",
    "IntervalSage"
]
