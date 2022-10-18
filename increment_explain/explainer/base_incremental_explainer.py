from abc import ABCMeta
from abc import abstractmethod


class BaseIncrementalExplainer(metaclass=ABCMeta):
    """Base class for incremental explainer algorithms.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
            self,
            model
    ):
        self.model = model

    @abstractmethod
    def explain_one(self, x, y):
        pass
