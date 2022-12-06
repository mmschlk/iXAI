"""
This module gathers SAGE Explanation Methods
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>
#          Rohit Jagtani

import random
from typing import Optional, Callable, Any, Union

from river.metrics.base import Metric
import numpy as np
from tqdm import tqdm

from increment_explain.explainer.base import BaseIncrementalFeatureImportance
from increment_explain.imputer import MarginalImputer, BaseImputer
from increment_explain.storage import IntervalStorage, BatchStorage
from increment_explain.storage.base import BaseStorage
from increment_explain.utils.validators.loss import validate_loss_function

__all__ = [
    "IncrementalSage",
    "BatchSage",
    "IntervalSage"
]

# =============================================================================
# Public SAGE Explainers
# =============================================================================


class IncrementalSage(BaseIncrementalFeatureImportance):
    """Incremental SAGE Explainer

    Computes SAGE importance values incrementally by applying exponential smoothing.
    For each input instance tuple x_i, y_i one update of the explanation procedure is performed.

    Attributes:
        marginal_prediction (np.ndarray): The current marginal prediction of the model_function (smoothed over time).
    """

    def __init__(
            self,
            model_function: Callable,
            loss_function: Union[Callable, Metric],
            feature_names: list,
            *,
            smoothing_alpha: Optional[float] = None,
            storage: Optional[BaseStorage] = None,
            imputer: Optional[BaseImputer] = None,
            n_inner_samples: int = 1,
            dynamic_setting: bool = True,
            loss_bigger_is_better: bool = False
    ):
        """
        Args:
            model_function (Callable): The Model function to be explained.
            loss_function (Union[Callable, Metric]): The loss function for which the importance values are calculated.
                This can either be a callable function or a predefined river.metric.base.Metric.
                - callable function: The loss_function needs to follow the signature of loss_function(y_true, y_pred)
                    and handle the output dimensions of the model function. Smaller values are interpreted as being
                    better. y_pred needs to be an array / list with the shape corresponding to the output dimension
                    (e.g. single-value outputs (e.g. regression, classification): y_pred_1 = [0], y_pred_2 = [1],
                    etc.; for multi-label outputs (e.g. probability scores) y_pred_1 = [0.72, 0.28],
                    y_pred_2 = [0.01, 0.99]).
                - river.metric.base.Metric: Any Metric implemented in river (e.g. river.metrics.CrossEntropy() for
                    classification or river.metrics.MSE() for regression).
            feature_names (list): List of feature names to be explained for the model.
            smoothing_alpha (float): The smoothing parameter for the exponential smoothing of the importance values.
                Should be in the interval between ]0,1]. Defaults to 0.001.
            storage (BaseStorage): Optional incremental data storage Mechanism. Defaults to
                `GeometricReservoirStorage(size=100)` for dynamic modelling settings and to
                `UniformReservoirStorage(size=100)` in static modelling settings.
            imputer (BaseImputer): Incremental imputing strategy to be used. Defaults to
                `MarginalImputer(sampling_strategy='joint')`.
            n_inner_samples (int): Number of model evaluation per feature and explanation step (observation).
                Defaults to 1.
            dynamic_setting (bool): Flag to indicate if the modelling setting is dynamic `True` (changing model, and
                adaptive explanation) or a static modelling setting `False` (all observations contribute equally to the
                final importance) is assumed. Defaults to `True`.
            loss_bigger_is_better (bool): Flag that indicates if a smaller loss value indicates a better fit ('True') or
                not ('False'). This is only used to represent the marginal- and model-loss more sensibly.
        """
        super(IncrementalSage, self).__init__(
            model_function=model_function,
            loss_function=loss_function,
            feature_names=feature_names,
            dynamic_setting=dynamic_setting,
            smoothing_alpha=smoothing_alpha,
            storage=storage,
            imputer=imputer
        )
        self._loss_direction = 1. if loss_bigger_is_better else 0.
        self.n_inner_samples = n_inner_samples
        self.marginal_prediction = np.zeros(shape=1)

    @property
    def marginal_loss(self):
        """Marginal loss (loss of the model without any features, default prediction loss) property,
        which is smoothed over time."""
        return self._marginal_loss_tracker.get() + self._loss_direction

    @property
    def model_loss(self):
        """Model loss (loss of model with features) property, which is smoothed over time."""
        return self._model_loss_tracker.get() + self._loss_direction

    @property
    def explained_loss(self):
        """Explained loss (difference between the current marginal and model loss.) property."""
        return self.marginal_loss - self.model_loss

    def explain_one(
            self,
            x_i: dict,
            y_i: Any,
            update_storage: bool = True
    ) -> dict[str, float]:
        """Explain one observation (x_i, y_i).

        Args:
            x_i (dict[str, Any]): The input features of the current observation as a dict of feature names to feature
                values.
            y_i (Any): Target label of the current observation.
            update_storage (bool): Flag if the underlying incremental data storage mechanism is to be updated with the
                new observation (`True`) or not (`False`). Defaults to `True`.

        Returns:
            (dict[str, float]): The current SAGE feature importance scores.
        """
        if self.seen_samples >= 1:
            permutation_chain = np.random.permutation(self.feature_names)
            y_i_pred = self._model_function(x_i)[0]
            model_loss = self._loss_function(y_i, y_i_pred)
            self._model_loss_tracker.update(model_loss)
            self.marginal_prediction = self._marginal_prediction_tracker.update(y_i_pred).get_normalized()
            sample_loss = self._loss_function(y_i, self.marginal_prediction)
            self._marginal_loss_tracker.update(sample_loss)
            features_not_in_S = set(self.feature_names)
            for feature in permutation_chain:
                features_not_in_S.remove(feature)
                predictions = self._imputer.impute(
                    feature_subset=features_not_in_S,
                    x_i=x_i,
                    n_samples=self.n_inner_samples
                )
                y = np.mean(np.asarray(predictions), axis=0)[0]
                feature_loss = self._loss_function(y_i, y)
                marginal_contribution = sample_loss - feature_loss
                sample_loss = feature_loss
                self._importance_trackers[feature].update(marginal_contribution)
                self._variance_trackers[feature].update((marginal_contribution - self.importance_values[feature])**2)
        self.seen_samples += 1
        if update_storage:
            self._storage.update(x_i, y_i)
        return self.importance_values


class BatchSage:

    def __init__(
            self,
            model_function: Callable,
            feature_names: list,
            loss_function: Union[Callable, Metric],
            n_inner_samples: int = 1,
            storage: Optional[BaseStorage] = None,
            imputer: Optional[BaseImputer] = None,
    ):
        """
        Args:
            model_function (Callable): The Model function to be explained.
            loss_function (Union[Callable, Metric]): The loss function for which the importance values are calculated.
                This can either be a callable function or a predefined river.metric.base.Metric.
                - callable function: The loss_function needs to follow the signature of loss_function(y_true, y_pred)
                    and handle the output dimensions of the model function. Smaller values are interpreted as being
                    better. y_pred needs to be an array / list with the shape corresponding to the output dimension
                    (e.g. single-value outputs (e.g. regression, classification): y_pred_1 = [0], y_pred_2 = [1],
                    etc.; for multi-label outputs (e.g. probability scores) y_pred_1 = [0.72, 0.28],
                    y_pred_2 = [0.01, 0.99]).
                - river.metric.base.Metric: Any Metric implemented in river (e.g. river.metrics.CrossEntropy() for
                    classification or river.metrics.MSE() for regression).
            feature_names (list): List of feature names to be explained for the model.
            storage (BaseStorage): Optional incremental data storage mechanism. Defaults to 
                `BatchStorage()`.
            imputer (BaseImputer): Incremental imputing strategy to be used. Defaults to 
                `MarginalImputer(sampling_strategy='joint')`.
            n_inner_samples (int): Number of model evaluation per feature and explanation step (observation).
                Defaults to 1.
        """
        self.feature_names = feature_names
        self.n_inner_samples = n_inner_samples
        self._model_function = model_function
        self._loss_function = validate_loss_function(loss_function, model_function)
        self._storage: BaseStorage = storage
        if self._storage is None:
            self._storage = BatchStorage(store_targets=True)
        self._imputer: BaseImputer = imputer
        if self._imputer is None:
            self._imputer = MarginalImputer(
                sampling_strategy='joint', model_function=self._model_function, storage_object=self._storage)
        self.importance_values = {feature_name: 0. for feature_name in self.feature_names}

    def update_storage(self, x_i, y_i):
        self._storage.update(x=x_i, y=y_i)

    def explain_one(self, x_i, y_i, original_sage: bool = False):
        self._storage.update(x_i, y_i)
        x_data, y_data = self._storage.get_data()
        if original_sage:
            self.explain_many_original(x_data=x_data, y_data=y_data)
        else:
            self.explain_many(x_data=x_data, y_data=y_data)
        return self.importance_values

    def explain_many(
            self,
            x_data: list[dict],
            y_data: list[Any],
    ) -> dict[str, float]:
        sage_values = {feature: 0. for feature in self.feature_names}
        n_data = len(x_data)
        marginal_prediction = sum(self._model_function(x_data)) / n_data
        for n, (x_i, y_i) in tqdm(enumerate(zip(x_data, y_data), start=1), total=n_data):
            permutation_chain = np.random.permutation(self.feature_names)
            loss_previous = self._loss_function(y_true=y_i, y_prediction=marginal_prediction)
            features_not_in_S = set(self.feature_names)
            for feature in permutation_chain:
                features_not_in_S.remove(feature)
                predictions = self._imputer.impute(
                    feature_subset=features_not_in_S,
                    x_i=x_i,
                    n_samples=self.n_inner_samples
                )
                y = np.mean(np.asarray(predictions), axis=0)[0]
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
        marginal_prediction = sum(self._model_function(x_data)) / n_data
        for n, (x_i, y_i) in tqdm(enumerate(zip(x_data, y_data), start=1), total=n_data):
            permutation_chain = np.random.permutation(self.feature_names)
            x_S = {}
            loss_previous = self._loss_function(y_true=y_i, y_prediction=marginal_prediction)
            for feature in permutation_chain:
                x_S[feature] = x_i[feature]
                y = 0
                for k in range(1, self.n_inner_samples + 1):
                    x_marginal = x_data[random.randint(0, n_data - 1)]
                    x_marginal = {**x_marginal, **x_S}
                    y += self._model_function(x_marginal)[0]
                y /= k
                feature_loss = self._loss_function(y_true=y_i, y_prediction=y)
                marginal_contribution = loss_previous - feature_loss
                sage_values[feature] += marginal_contribution
                loss_previous = feature_loss
            n_data = n
        self.importance_values = {feature: sage_value / n_data for feature, sage_value in sage_values.items()}
        return self.importance_values


class IntervalSage(BatchSage):

    def __init__(
            self,
            model_function: Callable,
            feature_names: list,
            loss_function: Union[Callable, Metric],
            n_inner_samples: int = 1,
            interval_length: int = 100,
            storage: BaseStorage = None,
            imputer: BaseImputer = None,
    ):

        if storage is None:
            storage = IntervalStorage(store_targets=True, size=interval_length)
        assert isinstance(storage, IntervalStorage), f"Only 'IntervalStorage' allowed not {type(storage)}."

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
        self._storage.update(x=x_i, y=y_i)
        self.seen_samples += 1
        if not force_explain and self.seen_samples % self._interval_length != 0:
            return self.importance_values
        x_data, y_data = self._storage.get_data()
        super().explain_many(x_data=x_data, y_data=y_data)
        return self.importance_values
