from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, List

from numpy import ndarray
import river
import numpy as np


__all__ = [
    "SyntheticDataset",
]


def _default_target_function(x):
    weights = [1 for _ in range(len(x))]
    return np.dot(x, weights)


# =============================================================================
# Base Datasets
# =============================================================================


class BaseSyntheticDataset(metaclass=ABCMeta):
    """Base class for creating synthetic data sets.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
            self,
            *,
            x_data: ndarray,
            y_data: ndarray,
            feature_names: List[str]
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.feature_names = feature_names

    def to_stream(self):
        """ Generates a data stream from the dataset

        :return: data stream as an iterator of dictionary data samples
        :rtype: river.base.typing.Stream
        """
        return river.stream.iter_array(self.x_data, self.y_data, feature_names=self.feature_names)


# =============================================================================
# Public Datasets
# =============================================================================


class SyntheticDataset(BaseSyntheticDataset):
    """ A synthetic dataset with numeric and categorical values.

    Parameters
    ----------
    target_function : Callable, default=None
        The function used to create the target labels (i.e. the mapping function / relationship from X to y).
        If no target function is provided the default function is a simple dot-product of the input data x with a
        weight-vector containing only ones (i.e. sum of the feature values) is used.

    n_samples : int, default=1000
        Number of samples (i.e. rows) of the dataset.

    n_features : int, default=15
        Total Number of features (i.e. columns) of the dataset.

    n_numeric : int default=10
        Number of numerical features of the dataset.

    n_categorical_values : int default=5
        Number of categorical features of the dataset. If "n_numeric" + "n_categorical_values" is larger than
        "n_features", then "n_categorical" is the remaining difference between "n_features" - "n_numeric".

    noise_mean : float default=0.
        Mean of the artificial noise added to the label function. This mean can create a bias in the data.
        If set to 0. then no bias is added.

    noise_std : float default=0.
        Standard deviation of the artificial noise added to the label function. This adds some random deviation around
        the "target_function" to induce more randomness into the dataset.
        If set to 0. then no deviation is added.
    """

    def __init__(
            self,
            classification: bool = False,
            target_function: Callable = None,
            n_samples: int = 10000,
            n_features: int = 15,
            n_numeric: int = 10,
            n_categorical_values: int = 5,
            noise_mean: float = 0.,
            noise_std: float = 0.,
            random_seed: Optional[int] = None
    ):
        self.classification = classification
        np.random.seed(random_seed)
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_numeric = n_numeric if n_numeric <= self.n_features else self.n_features
        self.n_categorical = self.n_features - self.n_numeric
        self.n_categorical_values = n_categorical_values
        self.target_function = target_function
        if self.target_function is None:
            self.target_function = _default_target_function

        self._noise_mean = noise_mean
        self._noise_std = noise_std

        x_data, feature_names = self._create_x_data()
        y_data = self._create_y_data(x_data)

        super(SyntheticDataset, self).__init__(
            x_data=x_data,
            y_data=y_data,
            feature_names=feature_names
        )

    def __len__(self):
        return self.n_samples

    def _create_x_data(self):
        x_data = np.random.rand(self.n_samples, self.n_numeric)
        feature_names = ["N_"+str(i + 1) for i in range(self.n_numeric)]
        if self.n_categorical > 0:
            x_categorical = np.random.randint(low=0, high=self.n_categorical_values,
                                              size=(self.n_samples, self.n_categorical))
            x_data = np.concatenate((x_data, x_categorical), axis=1)
            feature_names.extend(["C_" + str(i + 1) for i in range(self.n_categorical)])
        return x_data, feature_names

    def _create_y_data(self, x_data):
        y_data = np.apply_along_axis(self.target_function, arr=x_data, axis=1)
        if self.classification:
            y_data = y_data.astype(int)
        else:
            y_data = y_data + np.random.normal(self._noise_mean, self._noise_std, len(y_data))
        return y_data


if __name__ == "__main__":
    dataset = SyntheticDataset(n_features=20, n_numeric=10, n_samples=1000)
    pass
