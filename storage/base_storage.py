from typing import List, Dict, Optional, Any
from abc import abstractmethod, ABC


class BaseStorage(ABC):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    @abstractmethod
    def __init__(self):
        self._storage_x: List[Dict[str, Any]] = []
        self._storage_y: List = []

    @abstractmethod
    def update(self, x: Dict, y: Optional[Any]):
        raise NotImplementedError

    def get_data(self):
        return (self._storage_x, self._storage_y)
