from abc import abstractmethod, ABC


class BaseImputer(ABC):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def impute(self, model, replacement_size, subset_of_features, x_i):
        raise NotImplementedError
