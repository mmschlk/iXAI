from .base_storage import BaseStorage
from typing import List, Dict, Any, Optional


class BatchStorage(BaseStorage):
    """ A Batch Storage storing all seen samples.
    """

    def __init__(self, store_targets=True):
        self.store_targets = store_targets
        self._storage_x: List[Dict[str, Any]] = []
        self._storage_y: List = []

    def update(self, x: Dict, y: Optional[Any] = None):
        self._storage_x.append(x)
        if self.store_targets:
            self._storage_y.append(y)
        self.stored_samples = len(self._storage_x)
