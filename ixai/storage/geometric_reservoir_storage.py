"""
This module contains the GeometricReservoirStorage.
"""
import random
from typing import Optional, Any

from .reservoir_storage import ReservoirStorage


class GeometricReservoirStorage(ReservoirStorage):
    """ Geometric Reservoir Storage

    Summarizes a data stream by keeping track of a fixed length reservoir of observations.
    Unlike the `UniformReservoirStorage`, the `GeometricReservoirStorage` maintains a geometric
    distribution over the observations in the reservoir. The probability of an observation being
    in the reservoir is proportional to the age of the observation. More recent observations are
    more likely to be in the reservoir.
    """

    def __init__(
            self,
            size: int = 200,
            constant_probability: float = 1.,
            store_targets: bool = False
    ):
        super().__init__(
            size=size,
            store_targets=store_targets
        )
        assert 0 < constant_probability <= 1, ValueError("constant_probability must be in (0, 1]")
        self.constant_probability = constant_probability

    def _add_observation_to_not_full_storage(self, x: dict, y: Optional[Any] = None):
        self._storage_x.append(x)
        if self.store_targets:
            self._storage_y.append(y)

    def _add_observation_to_full_storage(self, x: dict, y: Optional[Any] = None):
        rand_idx = random.randint(0, self.size - 1)
        self._storage_x[rand_idx] = x
        if self.store_targets:
            self._storage_y[rand_idx] = y

    def update(self, x: dict, y: Optional[Any] = None):
        if len(self._storage_x) < self.size:
            self._add_observation_to_not_full_storage(x, y)
        else:
            random_float = random.random()
            if random_float <= self.constant_probability:
                self._add_observation_to_full_storage(x, y)
