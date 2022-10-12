"""
This module gathers Sampling Methods that are meant for Incremental Learning
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>
import copy
from abc import ABCMeta
from abc import abstractmethod
import random
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

__all__ = [
    "HistogramSampler",
    "ReservoirSampler",
    "BatchSampler",
]

# =============================================================================
# Types and constants
# =============================================================================

EPS = 1e-10

# =============================================================================
# Base Sampler Class
# =============================================================================


class BaseSampler(metaclass=ABCMeta):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
            self,
            *,
            feature_names,
            store_targets,
            random_state
    ):
        self.feature_names = feature_names
        self.store_targets = store_targets
        self.random_state = random_state
        self.stored_samples = 0
        self.number_of_features = len(feature_names)

    @abstractmethod
    def update(self, x):
        pass

    @abstractmethod
    def sample(self, k):
        pass

# =============================================================================
# Public Samplers
# =============================================================================


class BatchSampler(BaseSampler):
    """ A Batch Sampler storing all seen samples.

    Parameters
    ----------
    feature_names : list(str)
        The names of the features.

    store_targets : bool, default=True
        Controls if target values (labels) should be stored and sampled.
        ``True`` denotes that target values are stored and sampled.
        ``False`` denotes that target values are neither stored nor sampled.

    sampling_strategy : {"joint", "product"}, default="joint"
    The strategy used to sample data points from the storage.
    ``"joint"`` joint marginal distribution is sampled. Features are sampled together.
    ``"product"`` sampling from the product of the marginal distribution is performed. Features are sampled
    independently of each other.

    random_state : int, or None, default=None
        Sets the random seed of the histogram

    Attributes
    ----------
    seen_samples : int
        number of samples stored in the batch sampler

    Notes
    -----
    This should be used for working with small incrementalized 'datasets' instead of real 'data streams'.
    Sampling with replacement is performed.
    """

    def __init__(
            self,
            *,
            feature_names,
            store_targets=True,
            sampling_strategy='joint',
            random_state=None
    ):
        super().__init__(
            feature_names=feature_names,
            store_targets=store_targets,
            random_state=random_state
        )
        assert sampling_strategy in ['joint', 'product']
        if sampling_strategy == 'product':
            # TODO implement product and conditional sampling
            raise NotImplementedError("'product' sampling is not implemented yet.")
        self.sampling_strategy = sampling_strategy
        self._storage_x = []
        self._storage_y = []

    def update(self, x, y=None):
        self._storage_x.append(x)
        self._storage_y.append(y)
        self.stored_samples += 1

    def sample(self, k=None, last_k=True, sampling_strategy=None):
        if sampling_strategy is None:
            sampling_strategy = self.sampling_strategy
        if k is None:
            k = self.stored_samples
        if last_k:
            return self._sample_last_k_jointly(k)

        if sampling_strategy == 'joint':
            return self._sample_k_jointly(k)
        return self._sample_k_product(k)

    def _sample_last_k_jointly(self, k):
        return self._storage_x[-k:], self._storage_y[-k:]

    def _sample_k_jointly(self, k):
        sample_indices = random.sample(list(range(0, self.stored_samples)), k)
        sampled_x = [self._storage_x[index] for index in sample_indices]
        sampled_y = [self._storage_y[index] for index in sample_indices]
        return sampled_x, sampled_y

    def _sample_k_product(self, k):
        sampled_x_dict = {}
        for feature in self.feature_names:
            sample_indices = random.sample(list(range(0, self.stored_samples)), k)
            sampled_features = [self._storage_x[index][feature] for index in sample_indices]
            sampled_x_dict[feature] = copy.deepcopy(sampled_features)
        sampled_x = []
        for sample_i in range(k):
            sampled_x.append({feature: sampled_x_dict[feature][sample_i] for feature in self.feature_names})
        return sampled_x, None


class HistogramSampler(BaseSampler):
    """ A Histogram Sampling Method

    Parameters
    ----------
    feature_names : list(str)
        The names of the feature names.

    sample_with_replacement : bool
        Controls what sampling strategy to use.
        ``True`` denotes sampling with replacement.
        ``False`` denotes sampling without replacement.

    random_state : int, or None
        Sets the random seed of the histogram

    Attributes
    ----------
    histograms_ : dict
        The histogram storing the frequency of the observed feature values

    seen_samples : int
        number of samples stored in the histogram

    Notes
    -----
    Sampling with histograms essentially is sampling with the product of the marginal distribution, as samples for the
    individual features are drawn independently of each other and from the features.
    """

    def __init__(
            self,
            *,
            feature_names,
            sample_with_replacement=True,
            random_state=None,
    ):
        super().__init__(
            feature_names=feature_names,
            store_targets=False,
            random_state=random_state  # doesn't do anything atm
        )
        self.histograms_ = {}
        for feature in self.feature_names:
            self.histograms_[feature] = Counter()
        self.sample_with_replacement = sample_with_replacement

    def update(self, x):
        """Adds the feature values of the sample to the histogram sampler.

        Parameters
        ----------
        x : dict of feature-value pairs
            individual sample of features to store in the histogram
        """
        # TODO Add binning support for numeric features
        self.stored_samples += 1
        for feature in self.feature_names:
            self.histograms_[feature].update([x[feature]])

    def sample(self, k):
        """Sample k features values independently of each other.

        Parameters
        ----------
        k : int greater or equal to 1
            Number of samples to draw.

        Returns
        -------
        x_samples : ndarray of dicts of feature value pairs
            sampled features value pairs
        """
        x_samples = []
        for _ in range(k):
            sampled_features = {}
            for feature in self.feature_names:
                sampled_feature_value = self._sample_from_feature(feature_name=feature)
                sampled_features[feature] = sampled_feature_value
            x_samples.append(sampled_features)
            self.stored_samples -= 1
        x_samples = np.asarray(x_samples)
        return x_samples

    def _sample_from_feature(self, feature_name):
        """Samples one value for a given feature.

        Parameters
        ----------
        feature_name : str
            name of the feature to draw a sample from

        Returns
        -------
        sampled_feature_value : object
            marginalized feature value
        """
        feature_values, sample_weights = zip(*self.histograms_[feature_name].items())
        sampled_feature_value = random.sample(feature_values, k=1, counts=sample_weights)[0]
        if self.sample_with_replacement:
            self.histograms_[feature_name].subtract([sampled_feature_value])
        return sampled_feature_value


class ReservoirSampler(BaseSampler):
    """An incremental Reservoir Sampler.

    """

    def __init__(
            self,
            *,
            feature_names,
            reservoir_length=1000,
            reservoir_mode='normal',
            sample_features_independently=False,
            constant_probability=None,
            sample_with_replacement=True,
            store_targets=True,
            random_state=None
    ):
        super().__init__(
            feature_names=feature_names,
            store_targets=store_targets,
            random_state=random_state
        )
        self.reservoir_length = reservoir_length
        assert reservoir_mode in ['normal', 'constant', 'skip']
        self.reservoir_mode = reservoir_mode
        self.sample_features_independently = sample_features_independently
        self.sample_with_replacement = sample_with_replacement
        if constant_probability is not None:
            self.constant_probability = constant_probability
        else:
            self.constant_probability = 1 / reservoir_length
        if self.sample_with_replacement and self.constant_probability < 1:
            raise ValueError(
                "The probability of adding a new sample must be set to 1, "
                "if the sampling strategy is sampling with replacement.")
        self.reservoir_x = {feature_name: [] for feature_name in self.feature_names}
        self.reservoir_y = []

    @property
    def n_reservoir_samples(self):
        return min(self.stored_samples, self.reservoir_length)

    def update(self, x, y=None):
        self.stored_samples += 1
        x = dict((feature, x[feature]) for feature in self.feature_names)
        if self.stored_samples <= self.reservoir_length:
            self._add_x_to_reservoir(x)
            if self.store_targets:
                self.reservoir_y.append(y)
        else:
            if self.reservoir_mode == 'constant':
                self._update_constant(x, y)
            else:
                self._update_normally(x, y)

    def sample(self, k=1):
        k = min(self.n_reservoir_samples, k)
        if self.sample_features_independently:
            x_samples = self._sample_independently(k)
            y_samples = None
        else:
            x_samples, y_samples = self._sample_jointly(k)
        return x_samples, y_samples

    def _add_x_to_reservoir(self, x, insertion_position=None):
        if not insertion_position:
            for feature in x.keys():
                self.reservoir_x[feature].append(x[feature])
        else:
            for feature in x.keys():
                self.reservoir_x[feature][insertion_position] = x[feature]

    def _add_sample_to_full_reservoir(self, x, y=None):
        if self.sample_with_replacement:
            self._add_x_to_reservoir(x)
            if self.store_targets:
                self.reservoir_y.append(y)
        else:
            random_insertion_position = random.randint(0, self.reservoir_length - 1)
            self._add_x_to_reservoir(x, insertion_position=random_insertion_position)
            if self.store_targets:
                self.reservoir_y[random_insertion_position] = y

    def _update_constant(self, x, y=None):
        random_float = random.random()
        if random_float <= self.constant_probability:
            self._add_sample_to_full_reservoir(x, y)

    def _update_normally(self, x, y=None):
        random_integer = random.randint(1, self.stored_samples)
        if random_integer <= self.reservoir_length:
            self._add_sample_to_full_reservoir(x, y)

    def _sample_jointly(self, k):
        sample_indices = random.sample(list(range(0, self.n_reservoir_samples)), k)
        x_samples = []
        y_samples = []
        if self.sample_with_replacement:
            for sample_index in sample_indices:
                x_samples.append({feature: self.reservoir_x[feature].pop(sample_index) for feature in self.feature_names})
        else:
            for sample_index in sample_indices:
                x_samples.append({feature: self.reservoir_x[feature][sample_index] for feature in self.feature_names})
        if self.store_targets:
            if self.sample_with_replacement:
                y_samples = [self.reservoir_y.pop(sub_sample_index) for sub_sample_index in sample_indices]
            else:
                y_samples = [self.reservoir_y[sub_sample_index] for sub_sample_index in sample_indices]
        return x_samples, y_samples

    def _sample_independently(self, k):
        x_samples = []
        if self.sample_with_replacement:
            for feature in self.feature_names:
                sample_indices = random.sample(list(range(0, self.stored_samples)), k)
                x_samples.append({feature: self.reservoir_x[feature].pop(sample_index) for sample_index in sample_indices})
        else:
            for feature in self.feature_names:
                sample_indices = random.sample(list(range(0, self.stored_samples)), k)
                x_samples.append({feature: self.reservoir_x[feature][sample_index] for sample_index in sample_indices})
        return x_samples
