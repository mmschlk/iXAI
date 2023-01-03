"""
This module contains the base ReservoirStorage.
"""
from abc import ABC

from .base import BaseStorage


class ReservoirStorage(BaseStorage, ABC):
    """Reservoir Storage - base class

    Attributes:
        size int: Size of the reservoir.
        store_targets bool: Flag if the target values are also stored.
    """

    def __init__(
            self,
            size: int,
            store_targets: bool = False
    ):
        super().__init__()
        self.size = size
        self.store_targets = store_targets
