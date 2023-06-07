"""
This module contains the OrderedReservoirStorage.
"""
import random
from collections import deque
from typing import Optional, Any, List

from ixai import GeometricReservoirStorage
from .reservoir_storage import ReservoirStorage


class OrderedReservoirStorage(GeometricReservoirStorage):
    """ Ordered Reservoir Storage

    The OrderedReservoirStorage is a subclass of the GeometricReservoirStorage. A observation is
    removed from the reservoir when a new observation is added by the order it was received.
    """

    def __init__(
            self,
            size: int = 200,
            constant_probability: float = None,
            store_targets: bool = False
    ):
        super().__init__(
            size=size,
            store_targets=store_targets,
            constant_probability=constant_probability
        )
        self._storage_x: deque[dict] = deque(maxlen=size)
        self._storage_y: deque = deque(maxlen=size)

    def _add_observation_to_full_storage(self, x: dict, y: Optional[Any] = None):
        self._storage_x.append(x)
        if self.store_targets:
            self._storage_y.append(y)

    def update(self, x: dict, y: Optional[Any] = None):
        if len(self._storage_x) < self.size:
            self._add_observation_to_not_full_storage(x, y)
        else:
            random_float = random.random()
            if random_float <= self.constant_probability:
                self._add_observation_to_full_storage(x, y)
