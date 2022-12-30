"""
This module gathers PFI Explanation Methods
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>
#          Rohit Jagtani

from typing import Optional, Union, Callable
from .base import BaseIncrementalFeatureImportance
import numpy as np
from river.metrics.base import Metric
from ..imputer import BaseImputer
from ..storage.base import BaseStorage


__all__ = [
    "IncrementalPFI",
]


class IncrementalPFI(BaseIncrementalFeatureImportance):
    """Incremental PFI Explainer

    Computes PFI importance values incrementally by applying exponential smoothing.
    For each input instance tuple x_i, y_i one update of the explanation procedure is performed.

    Attributes:
        n_inner_samples (int): The number of inner_samples used for removing features.
    """
    def __init__(
            self,
            model_function: Callable,
            loss_function: Union[Callable, Metric],
            feature_names: list[str],
            storage: Optional[BaseStorage] = None,
            imputer: Optional[BaseImputer] = None,
            n_inner_samples: int = 5,
            smoothing_alpha: float = 0.001,
            dynamic_setting: bool = True
    ):
        """
        Args:
            model_function (Callable): The Model function to be explained.
            loss_function (Union[Callable, Metric]): The loss function for which the importance values are calculated.
                This can either be a callable function or a predefined river.metric.base.Metric.<br>
                - callable function: The loss_function needs to follow the signature of loss_function(y_true, y_pred)
                    and handle the output dimensions of the model function. Smaller values are interpreted as being
                    better. y_pred needs to be an array / list with the shape corresponding to the output dimension
                    (e.g. single-value outputs (e.g. regression, classification): y_pred_1 = [0], y_pred_2 = [1],
                    etc.; for multi-label outputs (e.g. probability scores) y_pred_1 = [0.72, 0.28],
                    y_pred_2 = [0.01, 0.99]).<br>
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
        """
        super(IncrementalPFI, self).__init__(
            model_function=model_function,
            loss_function=loss_function,
            feature_names=feature_names,
            dynamic_setting=dynamic_setting,
            smoothing_alpha=smoothing_alpha,
            storage=storage,
            imputer=imputer
        )
        self.n_inner_samples = n_inner_samples

    def explain_one(
            self,
            x_i,
            y_i,
            n_inner_samples: Optional[int] = None,
            update_storage: bool = True
    ):
        """Explain one observation (x_i, y_i).

        Args:
            x_i (dict[str, Any]): The input features of the current observation as a dict of feature names to feature
                values.
            y_i (Any): Target label of the current observation.
            update_storage (bool): Flag if the underlying incremental data storage mechanism is to be updated with the
                new observation (`True`) or not (`False`). Defaults to `True`.
            n_inner_samples (int, optional): Number of model evaluation per feature for the current explanation step
                (observation). Overrides the attribute `n_inner_samples` only for the current explanation step.
                Defaults to `None`.
            update_storage: (bool): Flag that indicates if the current sample should also be added to the underling
                data storage mechanism. Defaults to `True`.

        Returns:
            (dict[Any, float]): The current PFI feature importance scores.
        """
        if self.seen_samples >= 1:
            if n_inner_samples is None:
                n_inner_samples = self.n_inner_samples
            original_prediction = self._model_function(x_i)
            original_loss = self._loss_function(y_i, original_prediction)
            pfi = {}
            for feature in self.feature_names:
                feature_subset = [feature]
                predictions = self._imputer.impute(
                    feature_subset=feature_subset,
                    x_i=x_i,
                    n_samples=n_inner_samples
                )
                losses = [self._loss_function(y_i, prediction) for prediction in predictions]
                avg_loss = np.mean(losses)
                pfi[feature] = avg_loss - original_loss
            self._importance_trackers.update(pfi)
            variances = {feature: (pfi[feature] - self.importance_values[feature]) ** 2
                         for feature in self.feature_names}
            self._variance_trackers.update(variances)
        self.seen_samples += 1
        if update_storage:
            self._storage.update(x_i, y_i)
        return self.importance_values
