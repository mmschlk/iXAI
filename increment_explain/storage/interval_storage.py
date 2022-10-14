from .base_storage import BaseStorage
from collections import deque
from typing import Dict, Optional, Any


class IntervalStorage(BaseStorage):
    """ An Interval Storage storing last k samples.
    """

    def __init__(
            self,
            size: int,
            store_targets: bool = True
    ):
        self.size = size
        self.store_targets = store_targets
        self._storage_x = deque()
        self._storage_y = deque()

    def update(self, x: Dict, y: Optional[Any] = None):
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
        return (list(self._storage_x), list(self._storage_y))
