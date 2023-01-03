from .base import BaseStorage
from typing import Any, Optional


class BatchStorage(BaseStorage):
    """ A Batch Storage storing all seen samples.
    """

    def __init__(self, store_targets: bool = True):
        self.store_targets = store_targets
        super().__init__()

    def update(self, x: dict, y: Optional[Any] = None):
        """Given a data point, it updates the storage.
        Args:
            x: Features as List of Dicts
            y: Target as float or integer
        Returns:
            None
        """
        self._storage_x.append(x)
        if self.store_targets:
            self._storage_y.append(y)
