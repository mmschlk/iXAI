"""
This module contains the incremental SAGE explainer.
"""
from typing import Callable, Any, Union, Dict, Sequence, Optional

import numpy as np
from river.metrics.base import Metric

from ixai.explainer.base import BaseIncrementalFeatureImportance, _get_mean_model_output
from ixai.imputer import BaseImputer
from ixai.storage.base import BaseStorage


class IncrementalSage(BaseIncrementalFeatureImportance):
    """Incremental SAGE Explainer

    Computes SAGE importance values incrementally by applying exponential smoothing.
    For each input instance tuple x_i, y_i one update of the explanation procedure is performed.

    Args:
        model_function (Callable): The Model function to be explained (e.g. ``model.predict_one``
        (river), ``model.predict_proba`` (sklearn)).
        loss_function (Union[Metric, Callable[[Any, Dict], float]]): The loss function for which
            the importance values are calculated. This can either be a callable function or a
            predefined ``river.metric.base.Metric``.
                river.metric.base.Metric: Any Metric implemented in river (e.g.
                ``river.metrics.CrossEntropy()`` for classification or ``river.metrics.MSE()`` for
                regression).
                callable function: The loss_function needs to follow the signature of
                loss_function(y_true, y_pred) and handle the output dimensions of the model
                function. Smaller values are interpreted as being better if not overriden with
                ``loss_bigger_is_better=True``. ``y_pred`` is passed as a dict.
        feature_names (Sequence[Union[str, int, float]]): List of feature names to be explained
            for the model.
        smoothing_alpha (float, optional): The smoothing parameter for the exponential smoothing
            of the importance values. Should be in the interval between ]0,1].
            Defaults to 0.001.
        storage (BaseStorage, optional): Optional incremental data storage Mechanism.
            Defaults to ``GeometricReservoirStorage(size=100)`` for dynamic modelling settings
            (``dynamic_setting=True``) and ``UniformReservoirStorage(size=100)`` in static modelling
            settings (``dynamic_setting=False``).
        imputer (BaseImputer, optional): Incremental imputing strategy to be used. Defaults to
            ``MarginalImputer(sampling_strategy='joint')``.
        n_inner_samples (int): Number of model evaluation per feature and explanation step
            (observation). Defaults to 1.
        dynamic_setting (bool): Flag to indicate if the modelling setting is dynamic ``True``
            (changing model, and adaptive explanation) or a static modelling setting ``False``
            (all observations contribute equally to the final importance). Defaults to ``True``.
        loss_bigger_is_better (bool): Flag that indicates if a smaller loss value indicates a
            better fit ('True') or not ('False').  This is only used to represent the marginal-
            and model-loss more sensibly.

    Attributes:
        marginal_prediction (dict): The current marginal prediction of the model_function
            (smoothed over time).
        n_inner_samples (int): Number of model evaluation per feature and explanation step
            (observation).
    """

    def __init__(
            self,
            model_function: Callable,
            loss_function: Union[Metric, Callable[[Any, Dict], float]],
            feature_names: Sequence[Union[str, int, float]],
            *,
            smoothing_alpha: Optional[float] = None,
            storage: Optional[BaseStorage] = None,
            imputer: Optional[BaseImputer] = None,
            n_inner_samples: int = 1,
            dynamic_setting: bool = True,
            loss_bigger_is_better: bool = False
    ):
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
        self.marginal_prediction: dict = {}

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
            n_inner_samples: Optional[int] = None,
            update_storage: bool = True
    ) -> dict:
        """Explain one observation (x_i, y_i).

        Args:
            x_i (dict): The input features of the current observation as a dict of feature names to
                feature values.
            y_i (Any): Target label of the current observation.
            n_inner_samples (int, optional): Number of model evaluation per feature for the current
                explanation step (observation). Defaults to ``None``.
            update_storage (bool): Flag if the underlying incremental data storage mechanism is to
                be updated with the new observation (``True``) or not (``False``). Defaults to
                ``True``.

        Returns:
            (dict): The current SAGE feature importance scores.
        """
        if self.seen_samples >= 1:
            if n_inner_samples is None:
                n_inner_samples = self.n_inner_samples
            permutation_chain = np.random.permutation(self.feature_names)
            y_i_pred = self._model_function(x_i)
            model_loss = self._loss_function(y_i, y_i_pred)
            self._model_loss_tracker.update(model_loss)
            self._marginal_prediction_tracker.update(y_i_pred)
            self.marginal_prediction = self._marginal_prediction_tracker.get_normalized()
            sample_loss = self._loss_function(y_i, self.marginal_prediction)
            self._marginal_loss_tracker.update(sample_loss)
            features_not_in_s = set(self.feature_names)
            marginal_contributions = {}
            for feature in permutation_chain:
                features_not_in_s.remove(feature)
                predictions = self._imputer.impute(
                    feature_subset=features_not_in_s,
                    x_i=x_i,
                    n_samples=n_inner_samples
                )
                y = _get_mean_model_output(predictions)
                feature_loss = self._loss_function(y_i, y)
                marginal_contribution = sample_loss - feature_loss
                sample_loss = feature_loss
                marginal_contributions[feature] = marginal_contribution
            self._importance_trackers.update(marginal_contributions)
            variances = {
                feature: (marginal_contributions[feature] - self.importance_values[feature])**2
                for feature in self.feature_names
            }
            self._variance_trackers.update(variances)
        self.seen_samples += 1
        if update_storage:
            self._storage.update(x_i, y_i)
        return self.importance_values
