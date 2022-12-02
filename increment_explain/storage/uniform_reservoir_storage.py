from .reservoir_storage import ReservoirStorage
from typing import Dict, Optional, Any
import random


class UniformReservoirStorage(ReservoirStorage):
    """ Uniform Reservoir Storage

    Attributes:
        stored_samples int: Number of samples seen in the stream.
    """

    def __init__(
            self,
            store_targets: bool,
            size: int
    ):
        super().__init__(
            size=size,
            store_targets=store_targets
        )
        super(ReservoirStorage, self).__init__()
        self.stored_samples: int = 0

    def update(self, x: Dict, y: Optional[Any] = None):
        self.stored_samples += 1
        if len(self._storage_x) < self.size:
            self._storage_x.append(x)
            if self.store_targets:
                self._storage_y.append(y)
        else:
            random_integer = random.randint(1, self.stored_samples)
            if random_integer <= self.size:
                rand_idx = random.randrange(self.size)
                self._storage_x[rand_idx] = x
                if self.store_targets:
                    self._storage_y[rand_idx] = y
