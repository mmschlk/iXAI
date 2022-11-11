"""
This module gathers SAGE Explanation Methods
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>
import math
import random
from typing import Optional, Callable, Any

import numpy as np
from tqdm import tqdm

from increment_explain.explainer import BaseIncrementalFeatureImportance
from increment_explain.imputer import MarginalImputer, BaseImputer
from increment_explain.storage import GeometricReservoirStorage, UniformReservoirStorage, IntervalStorage, BatchStorage
from increment_explain.storage.base import BaseStorage
from increment_explain.utils.trackers import ExponentialSmoothingTracker

__all__ = [
    "IncrementalSageExplainer",
    "BatchSageExplainer",
    "IntervalSageExplainer"
]


# =============================================================================
# Public SAGE Explainers
# =============================================================================


class IncrementalSageExplainer(BaseIncrementalFeatureImportance):

    def __init__(
            self,
            model_function: Callable,
            *,
            storage: Optional[BaseStorage] = None,
            imputer: Optional[BaseImputer] = None,
            feature_names: list[str],
            loss_function: Callable,
            n_inner_samples: int = 1,
            smoothing_alpha: Optional[float] = None,
            dynamic_setting: bool = True,
            pfi_mode: bool = False
    ):
        super(IncrementalSageExplainer, self).__init__(
            model_function=model_function,
            feature_names=feature_names,
            dynamic_setting=dynamic_setting,
            smoothing_alpha=smoothing_alpha,
        )
        self._loss_function = loss_function
        self.storage = storage
        self.imputer = imputer
        if imputer is None:
            if storage is None:
                if dynamic_setting:
                    self.storage = GeometricReservoirStorage(store_targets=False, size=100)
                else:
                    self.storage = UniformReservoirStorage(store_targets=False, size=100)
            self.imputer = MarginalImputer(self.model_function, 'product', storage)
        self.n_inner_samples = n_inner_samples
        self._pfi_mode = pfi_mode

    def get_confidence_bound(self, delta: float):
        assert 0 < delta <= 1., f"Delta must be float in the interval of ]0,1] and not {delta}."
        return {
            feature_name:
                (1 - self._smoothing_alpha) ** self.seen_samples +
                (1 / math.sqrt(delta)) * math.sqrt(self.variances[feature_name].get()) *
                math.sqrt(self._smoothing_alpha / (2 - self._smoothing_alpha))
            for feature_name in self.feature_names}

    def explain_one(self, x_i, y_i, update_storage=True):
        if self.seen_samples >= 1:
            permutation_chain = np.random.permutation(self.feature_names)
            y_i_pred = self.model_function(x_i)[0]
            self._marginal_prediction.update(y_i_pred)
            sample_loss = self._loss_function(y_true=y_i, y_prediction=self._marginal_prediction())
            features_not_in_S = set(self.feature_names)
            for feature in permutation_chain:
                features_not_in_S.remove(feature)
                predictions = self.imputer.impute(
                    feature_subset=features_not_in_S,
                    x_i=x_i,
                    n_samples=self.n_inner_samples
                )
                y = np.mean(predictions)
                feature_loss = self._loss_function(y_true=y_i, y_prediction=y)
                marginal_contribution = sample_loss - feature_loss
                sample_loss = feature_loss
                self.importance_trackers[feature].update(marginal_contribution)
                self.variances[feature].update((marginal_contribution - self.importance_values[feature])**2)
        self.seen_samples += 1
        if update_storage:
            self.storage.update(x_i, y_i)
        return self.importance_values


class BatchSageExplainer:

    def __init__(
            self,
            model_function: Callable,
            feature_names: list[str],
            loss_function: Callable,
            n_inner_samples: int = 1,
            storage: BaseStorage = None,
            imputer: BaseImputer = None,
    ):
        self.feature_names = feature_names
        self.model_function = model_function
        self._loss_function = loss_function
        self._n_inner_samples = n_inner_samples
        self._loss_function = loss_function
        self.storage = storage
        if storage is None:
            self.storage = BatchStorage(store_targets=True)
        self.imputer = imputer
        if imputer is None:
            self.imputer = MarginalImputer(
                sampling_strategy='joint', model_function=self.model_function, storage_object=self.storage
            )
        self.importance_values = {feature_name: 0. for feature_name in self.feature_names}

    def update_storage(self, x_i, y_i):
        self.storage.update(x=x_i, y=y_i)

    def explain_one(self, x_i, y_i):
        self.storage.update(x_i, y_i)
        x_data, y_data = self.storage.get_data()
        self.explain_many(x_data=x_data, y_data=y_data)
        return self.importance_values

    def explain_many(
            self,
            x_data: list[dict[str, Any]],
            y_data: list[Any],
    ) -> dict[str, float]:
        sage_values = {feature: 0. for feature in self.feature_names}
        n_data = len(x_data)
        marginal_prediction = sum(self.model_function(x_data)) / n_data
        for n, (x_i, y_i) in tqdm(enumerate(zip(x_data, y_data), start=1), total=n_data):
            permutation_chain = np.random.permutation(self.feature_names)
            loss_previous = self._loss_function(y_true=y_i, y_prediction=marginal_prediction)
            features_not_in_S = set(self.feature_names)
            for feature in permutation_chain:
                features_not_in_S.remove(feature)
                predictions = self.imputer.impute(
                    feature_subset=features_not_in_S,
                    x_i=x_i,
                    n_samples=self._n_inner_samples
                )
                y = np.mean(predictions)
                feature_loss = self._loss_function(y_true=y_i, y_prediction=y)
                marginal_contribution = loss_previous - feature_loss
                sage_values[feature] += marginal_contribution
                loss_previous = feature_loss
            n_data = n
        self.importance_values = {feature: sage_value / n_data for feature, sage_value in sage_values.items()}
        return self.importance_values

    def explain_many_original(
            self,
            x_data: list[dict[str, Any]],
            y_data: list[Any],
    ) -> dict[str, float]:
        sage_values = {feature: 0. for feature in self.feature_names}
        n_data = len(x_data)
        marginal_prediction = sum(self.model_function(x_data)) / n_data
        for n, (x_i, y_i) in tqdm(enumerate(zip(x_data, y_data), start=1), total=n_data):
            permutation_chain = np.random.permutation(self.feature_names)
            x_S = {}
            loss_previous = self._loss_function(y_true=y_i, y_prediction=marginal_prediction)
            for feature in permutation_chain:
                x_S[feature] = x_i[feature]
                y = 0
                for k in range(1, self._n_inner_samples + 1):
                    x_marginal = x_data[random.randint(0, n_data - 1)]
                    x_marginal = {**x_marginal, **x_S}
                    y += self.model_function(x_marginal)[0]
                y /= k
                feature_loss = self._loss_function(y_true=y_i, y_prediction=y)
                marginal_contribution = loss_previous - feature_loss
                sage_values[feature] += marginal_contribution
                loss_previous = feature_loss
            n_data = n
        self.importance_values = {feature: sage_value / n_data for feature, sage_value in sage_values.items()}
        return self.importance_values


class IntervalSageExplainer(BatchSageExplainer):

    def __init__(
            self,
            model_function: Callable,
            feature_names: list[str],
            loss_function: Callable,
            n_inner_samples: int = 1,
            interval_length: int = 100,
            storage: BaseStorage = None,
            imputer: BaseImputer = None,
    ):

        if storage is None:
            storage = IntervalStorage(store_targets=True, size=interval_length)

        if imputer is None:
            imputer = MarginalImputer(model_function=model_function, sampling_strategy='joint', storage_object=storage)

        super().__init__(
            model_function=model_function,
            feature_names=feature_names,
            loss_function=loss_function,
            n_inner_samples=n_inner_samples,
            storage=storage,
            imputer=imputer
        )
        self._interval_length = interval_length
        self.seen_samples = 0

    def explain_one(
            self,
            x_i: dict[str, Any],
            y_i: Any,
            sub_sample_size: Optional[int] = None,
            force_explain: bool = False
    ) -> dict[str, float]:
        self.storage.update(x=x_i, y=y_i)
        self.seen_samples += 1
        if not force_explain and self.seen_samples % self._interval_length != 0:
            return self.importance_values
        x_data, y_data = self.storage.get_data()
        super().explain_many(x_data=x_data, y_data=y_data)
        return self.importance_values
