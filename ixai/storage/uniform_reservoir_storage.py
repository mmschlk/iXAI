"""
This module contains the UniformReservoirStorage.
"""
import random
from typing import Optional, Any

import numpy as np

from .reservoir_storage import ReservoirStorage


class UniformReservoirStorage(ReservoirStorage):
    """ Uniform Reservoir Storage

    Summarizes a data stream by keeping track of a fixed length reservoir of observations.
    Each past observation of the stream has an equal probability of being in the reservoir at
    the current time.
    For more information we refer to https://en.wikipedia.org/wiki/Reservoir_sampling.

    Attributes:
        stored_samples int: Number of samples observed in the stream.
    """

    def __init__(
            self,
            size: int = 1000,
            store_targets: bool = False
    ):
        """
        Args:
            size (int): Length of the reservoir to store samples. Defaults to 1000.
            store_targets (bool): Flag if target labels should also be stored. Defaults to False.
        """
        super().__init__(
            size=size,
            store_targets=store_targets
        )
        self.stored_samples: int = 0
        self._algo_wt = np.exp(np.log(random.random()) / self.size)
        self._algo_l_counter: int = (
                self.size + (np.floor(np.log(random.random()) / np.log(1 - self._algo_wt)) + 1)
        )

    def update(self, x: dict, y: Optional[Any] = None):
        """Updates the reservoir with the current sample if necessary.

        The update mechanism follows the optimal algorithm as stated here:
        https://en.wikipedia.org/wiki/Reservoir_sampling#Optimal:_Algorithm_L.

        Args:
            x (dict): Current observation's features.
            y (Any, optional): Current observation's label. Defaults to None.
        """
        self.stored_samples += 1
        if self.stored_samples <= self.size:
            self._storage_x.append(x)
            if self.store_targets:
                self._storage_y.append(y)
        else:
            if self._algo_l_counter == self.stored_samples:
                self._algo_l_counter += (np.floor(
                    np.log(random.random()) / np.log(1 - self._algo_wt)) + 1)
                rand_idx = random.randrange(self.size)
                self._storage_x[rand_idx] = x
                if self.store_targets:
                    self._storage_y[rand_idx] = y
                self._algo_wt *= np.exp(np.log(random.random()) / self.size)
