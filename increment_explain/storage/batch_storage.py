from .base_storage import BaseStorage
from typing import Dict, Any, Optional


class BatchStorage(BaseStorage):
    """ A Batch Storage storing all seen samples.
    """

    def __init__(self, store_targets=True):
        self.store_targets = store_targets
        super().__init__()

    def update(self, x: Dict, y: Optional[Any] = None):
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
