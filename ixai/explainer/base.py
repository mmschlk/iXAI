"""
This module gathers base Explanation Methods
"""
import copy
import math
import abc
from typing import Union, Sequence, Dict, List, Callable, Any, Optional

from river.metrics.base import Metric

from ixai.imputer import BaseImputer, MarginalImputer
from ixai.storage import GeometricReservoirStorage, UniformReservoirStorage
from ixai.storage.base import BaseStorage
from ixai.utils.tracker.base import Tracker
from ixai.utils.tracker import MultiValueTracker, WelfordTracker, ExponentialSmoothingTracker
from ixai.utils.validators.loss import validate_loss_function
from ixai.utils.validators.model import validate_model_function


class BaseIncrementalExplainer(metaclass=abc.ABCMeta):
    """Base class for incremental explainer algorithms.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Args:
        model_function (Callable): The Model function to be explained.
        feature_names (list): List of feature names to be explained for the model.

    Attributes:
        feature_names (list[typing.Any]): List of feature names that are explained.
        number_of_features (int): Number of features that are explained.
        seen_samples (int): Number of instances observed.
    """

    @abc.abstractmethod
    def __init__(
            self,
            model_function: Callable[[Any], Any],
            feature_names: Sequence[Union[str, int, float]]
    ):
        self._model_function = validate_model_function(model_function)
        self.feature_names = feature_names
        self.number_of_features: int = len(feature_names)
        self.seen_samples: int = 0

    def __repr__(self):
        return f"Explainer for {self.number_of_features} features after {self.seen_samples} samples."


class BaseIncrementalFeatureImportance(BaseIncrementalExplainer):
    """Base class for incremental feature importance explainer algorithms.

   Warning: This class should not be used directly.
   Use derived classes instead.
   """

    @abc.abstractmethod
    def __init__(
            self,
            model_function: Callable[[Any], Any],
            loss_function: Union[Metric, Callable[[Any, Dict], float]],
            feature_names: Sequence[Union[str, int, float]],
            storage: Optional[BaseStorage] = None,
            imputer: Optional[BaseImputer] = None,
            dynamic_setting: bool = False,
            smoothing_alpha: Optional[float] = None
    ):
        super().__init__(model_function, feature_names)
        self._loss_function = validate_loss_function(loss_function)

        self._smoothing_alpha = 0.001 if smoothing_alpha is None else smoothing_alpha
        if dynamic_setting:
            assert 0. < smoothing_alpha <= 1., f"The smoothing parameter needs to be in the range" \
                                               f" of ']0,1]' and not " \
                                               f"'{self._smoothing_alpha}'."
            base_tracker = ExponentialSmoothingTracker(alpha=self._smoothing_alpha)
        else:
            base_tracker = WelfordTracker()
        self._marginal_loss_tracker: Tracker = copy.deepcopy(base_tracker)
        self._model_loss_tracker: Tracker = copy.deepcopy(base_tracker)
        self._marginal_prediction_tracker: MultiValueTracker = MultiValueTracker(copy.deepcopy(base_tracker))
        self._importance_trackers: MultiValueTracker = MultiValueTracker(copy.deepcopy(base_tracker))
        self._variance_trackers: MultiValueTracker = MultiValueTracker(copy.deepcopy(base_tracker))
        self._storage: BaseStorage = storage
        if self._storage is None:
            if dynamic_setting:
                self._storage = GeometricReservoirStorage(store_targets=False, size=100)
            else:
                self._storage = UniformReservoirStorage(store_targets=False, size=100)
        self._imputer: BaseImputer = imputer
        if self._imputer is None:
            self._imputer = MarginalImputer(self._model_function, 'joint', self._storage)

    @abc.abstractmethod
    def explain_one(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def importance_values(self):
        """Incremental Importance Values property."""
        return self._importance_trackers.get()

    @property
    def variances(self):
        """Incremental Variances values property."""
        return self._variance_trackers.get()

    def get_normalized_importance_values(self, mode: str = 'sum') -> dict:
        """Normalizes the importance scores.

        Args:
            mode (str): The normalization mode to be applied. Possible values are 'sum' and 'delta'.
            Defaults to 'sum'.
                - sum: Normalizes the importance scores by division through the sum of importance scores.
                - delta: Normalizes the importance scores by division through the difference between the max of the
                importance scores and the min of the importance scores.

        Returns:
            (dict): The normalized importance values.
        """
        return self._normalize_importance_values(self.importance_values, mode=mode)

    def get_confidence_bound(self, delta: float):
        """Calculates Delta-Confidence Bounds.

        Args:
            delta (float): The confidence parameter. Must be a value in the interval of ]0,1].

        Returns:
            (dict): The upper confidence bound around the point estimate of the importance values.
                This value needs to be added to the top and bottom of the point estimate.
        """
        assert 0 < delta <= 1., f"Delta must be float in the interval of ]0,1] and not {delta}."
        return {
            feature_name:
                (1 - self._smoothing_alpha) ** self.seen_samples +
                (1 / math.sqrt(delta)) * math.sqrt(self.variances[feature_name]) *
                math.sqrt(self._smoothing_alpha / (2 - self._smoothing_alpha))
            for feature_name in self.feature_names}

    @staticmethod
    def _normalize_importance_values(importance_values: dict, mode: str = 'sum') -> dict:
        importance_values_list = list(importance_values.values())
        if mode == 'delta':
            factor = max(importance_values_list) - min(importance_values_list)
        elif mode == 'sum':
            factor = sum(importance_values_list)
        else:
            raise NotImplementedError(f"The mode must be either 'sum', or 'delta' not '{mode}'.")
        try:
            return {feature: importance_value / factor for feature, importance_value in importance_values.items()}
        except ZeroDivisionError:
            return {feature: 0.0 for feature, importance_value in importance_values.items()}

    def update_storage(self, x_i: dict, y_i: Optional[Any] = None):
        """Manually updates the data storage with the given observation.
        Args:
            x_i (dict): The input features of the current observation.
            y_i (Any, optional): Target label of the current observation. Defaults to `None`
        """
        self._storage.update(x=x_i, y=y_i)


def _get_mean_model_output(model_outputs: List[dict]) -> dict:
    """Calculates the mean values of a list of dict model outputs.

    Args:
        model_outputs (list[dict]): List of model outputs.

    Returns:
        (dict) The mean model output, where every label value is the average of all individual label values.
    """
    all_labels = {label for model_output in model_outputs for label in model_output}
    mean_output = {label: sum([output.get(label, 0) for output in model_outputs]) / len(model_outputs)
                   for label in all_labels}
    return mean_output
