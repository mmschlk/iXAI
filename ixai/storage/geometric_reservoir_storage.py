from .reservoir_storage import ReservoirStorage
from typing import Dict, Optional, Any
import random


class GeometricReservoirStorage(ReservoirStorage):
    """ Geometric Reservoir Storage
    """

    def __init__(
            self,
            size: int,
            constant_probability: float = None,
            store_targets: bool = False
    ):
        super().__init__(
            size=size,
            store_targets=store_targets
        )
        super(ReservoirStorage, self).__init__()
        if constant_probability is not None:
            self.constant_probability = constant_probability
        else:
            self.constant_probability = 1 / self.size

    def update(self, x: Dict, y: Optional[Any] = None):
        if len(self._storage_x) < self.size:
            self._storage_x.append(x)
            if self.store_targets:
                self._storage_y.append(y)
        else:
            random_float = random.random()
            if random_float <= self.constant_probability:
                rand_idx = random.randrange(self.size)
                self._storage_x[rand_idx] = x
                if self.store_targets:
                    self._storage_y[rand_idx] = y
