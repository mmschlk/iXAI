"""
This module gathers SAGE Explanation Methods
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>

import copy
from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from storage.sampler import BatchSampler, ReservoirSampler, HistogramSampler
from utils.trackers import WelfordTracker

__all__ = [
    "IncrementalSAGE",
]

# =============================================================================
# Types and constants
# =============================================================================

EPS = 1e-10

# TODO change MAE and MSE loss to be used with arrays
def mae_loss(y_true, y_prediction):
    return abs(y_true - y_prediction)


def mse_loss(y_true, y_prediction):
    return (y_true - y_prediction) ** 2


# =============================================================================
# Base Sampler Class
# =============================================================================


class BaseIncrementalExplainer(metaclass=ABCMeta):
    """Base class for incremental explainer algorithms.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
            self,
            *,
            model_fn,
            feature_names,
            random_state
    ):
        self.model_fn = model_fn
        self.feature_names = feature_names
        self.random_state = random_state
        self.number_of_features = len(feature_names)
        self.seen_samples = 0

    @abstractmethod
    def explain_one(self, x, y):
        pass


# =============================================================================
# Public SAGE Explainers
# =============================================================================


class IncrementalSAGE(BaseIncrementalExplainer):

    def __init__(
            self,
            model_fn,
            *,
            feature_names,
            loss_function='mse',
            random_state=None,
            sub_sample_size=1,
            empty_prediction=None
    ):
        super(IncrementalSAGE, self).__init__(
            model_fn=model_fn,
            feature_names=feature_names,
            random_state=random_state
        )
        # TODO enable custom loss functions as parameters
        assert loss_function in ['mae', 'mse'], "Loss function must be either 'mae' or 'mse'."
        if loss_function == 'mae':
            self.loss_function = mae_loss
        else:
            self.loss_function = mse_loss
        #self.sampler = BatchSampler(feature_names=feature_names, store_targets=False)  # TODO add incremental sampler
        self.sampler = ReservoirSampler(feature_names=feature_names, store_targets=False,
                                        reservoir_length=100, sample_with_replacement=False)
        self.marginal_prediction = WelfordTracker()
        self.sub_sample_size = sub_sample_size
        self.SAGE_values = {feature_name: WelfordTracker() for feature_name in feature_names}
        self.empty_prediction = 0 if empty_prediction is None else empty_prediction
        self.update_empty_prediction = True if empty_prediction is None else False  # flag if empty prediction must be updated

    def _update_sampler(self, x, y):
        self.sampler.update(x, y)

    def explain_one(self, x_i, y_i):
        if self.seen_samples <= 10:
            self.seen_samples += 1
            self._update_sampler(x_i, y_i)
            return self.SAGE_values
        permutation_chain = np.random.permutation(self.feature_names)
        # x_marginal, _ = self.sampler.sample(k=1, last_k=False, sampling_strategy='product')
        x_marginal, _ = self.sampler.sample(k=1)
        empty_prediction = self.model_fn(x_marginal[0]) if self.update_empty_prediction else self.empty_prediction
        self.marginal_prediction.update(empty_prediction)
        sample_loss = self.loss_function(y_true=y_i, y_prediction=self.marginal_prediction.mean)
        x_S = {}
        for feature in permutation_chain:
            x_S[feature] = x_i[feature]
            y = 0
            # x_marginals, _ = self.sampler.sample(k=self.sub_sample_size, last_k=False)
            x_marginals, _ = self.sampler.sample(k=self.sub_sample_size)
            k = 0
            while k < self.sub_sample_size:
                x_marginal = {**x_marginals[k], **x_S}
                y += self.model_fn(x_marginal)
                k += 1
            y /= k
            feature_loss = self.loss_function(y_true=y_i, y_prediction=y)
            marginal_contribution = sample_loss - feature_loss
            self.SAGE_values[feature].update(marginal_contribution)
            sample_loss = feature_loss
        self.seen_samples += 1
        self._update_sampler(x_i, y_i)
        return self.SAGE_values
