from abc import abstractmethod, ABC
import typing

from ixai.utils.validators import validate_model_function


class BaseImputer(ABC):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
            self,
            model_function: typing.Callable
    ):
        self.model_function = validate_model_function(model_function)

    @abstractmethod
    def impute(self, feature_subset, x_i, n_samples=None):
        raise NotImplementedError
