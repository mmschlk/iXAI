from .base import BaseStorage
from collections import deque
from typing import Optional, Any


class IntervalStorage(BaseStorage):
    """ An Interval Storage storing last k samples.
    """

    def __init__(
            self,
            size: int,
            store_targets: bool = True
    ):
        """
        Args:
            size (int): The length of the interval for which data points should be stored.
            store_targets (bool): Flag if the target values should be stored (`True`) or not (`False`). Defaults to
                `True`.
        """
        self.size = size
        self.store_targets = store_targets
        self._storage_x = deque()
        self._storage_y = deque()

    def update(self, x: dict, y: Optional[Any] = None):
        if len(self._storage_x) < self.size:
            self._storage_x.append(x)
            if self.store_targets:
                self._storage_y.append(y)
        else:
            self._storage_x.popleft()
            self._storage_x.append(x)
            if self.store_targets:
                self._storage_y.popleft()
                self._storage_y.append(y)

    def get_data(self):
        return list(self._storage_x), list(self._storage_y)
