"""
This module contains the OrderedReservoirStorage.
"""
import random
from collections import deque
from typing import Optional, Any, List

from .reservoir_storage import ReservoirStorage


class OrderedReservoirStorage(ReservoirStorage):
    """ Ordered Reservoir Storage
    """

    def __init__(
            self,
            size: int = 100,
            constant_probability: float = None,
            store_targets: bool = False
    ):
        super().__init__(
            size=size,
            store_targets=store_targets
        )
        self._storage_x: deque[dict] = deque(maxlen=size)
        self._storage_y: deque = deque(maxlen=size)
        if constant_probability is not None:
            self.constant_probability = constant_probability
        else:
            self.constant_probability = 1.

    def update(self, x: dict, y: Optional[Any] = None):
        if len(self._storage_x) < self.size:
            self._storage_x.append(x)
            if self.store_targets:
                self._storage_y.append(y)
        else:
            random_float = random.random()
            if random_float <= self.constant_probability:
                self._storage_x.append(x)
                if self.store_targets:
                    self._storage_y.append(y)
