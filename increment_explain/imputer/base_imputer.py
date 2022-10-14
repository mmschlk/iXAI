from abc import abstractmethod, ABC


class BaseImputer(ABC):
    """Base class for sampling algorithms.

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
    def impute(self, feature_subset, x_i, n_samples=None):
        raise NotImplementedError
