"""
This module gathers base Explanation Methods
"""
import copy
import math
import abc
from typing import Union, Dict, List, Callable, Any, Optional

from river.metrics.base import Metric

from ixai.imputer import BaseImputer, MarginalImputer
from ixai.storage import GeometricReservoirStorage, UniformReservoirStorage, BatchStorage
from ixai.storage.base import BaseStorage
from ixai.utils.tracker.base import Tracker
from ixai.utils.tracker import MultiValueTracker, WelfordTracker, ExponentialSmoothingTracker
from ixai.utils.validators.loss import validate_loss_function
from ixai.utils.validators.model import validate_model_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from collections import deque


def no_transform(values):
    return values


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
            feature_names: List[Any]
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
            feature_names: List[Any],
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


class BasePDP(metaclass=abc.ABCMeta):
    """Base class for PDP algorithms.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """
    @abc.abstractmethod
    def __init__(
            self,
            model_function: Callable[[Any], Any],
            pdp_feature,
            gridsize,
            dynamic_setting=None,
            smoothing_alpha=None,
            storage=None,
            output_key=1,
            ylim=None,
            is_batch_pdp=False,
            storage_size=None
    ):
        self.model_function = validate_model_function(model_function)
        self.pdp_feature = pdp_feature
        self.gridsize = gridsize
        self.output_key = output_key
        self.storage_size = storage_size
        if ylim is None:
            self.ylim = (0, 1)
        else:
            self.ylim = ylim
        self.is_batch_pdp = is_batch_pdp
        if self.is_batch_pdp:
            self.ice_curves_y = []
            self.ice_curves_x = []
        else:
            self.ice_curves_y = deque()
            self.ice_curves_x = deque()
        if storage is None:
            if self.is_batch_pdp:
                self._storage = BatchStorage(store_targets=False)
            else:
                self._storage = GeometricReservoirStorage(
                                                        size=100,
                                                        store_targets=False,
                                                        constant_probability=0.8)
        else:
            self._storage = storage

        if not self.is_batch_pdp:
            self.pdp_storage_x = deque()
            self.pdp_storage_y = deque()
            self._smoothing_alpha = smoothing_alpha
            if dynamic_setting:
                assert 0. < smoothing_alpha <= 1., f"The smoothing parameter needs to be in the range" \
                                                   f" of ']0,1]' and not " \
                                                   f"'{self._smoothing_alpha}'."
                base_tracker = ExponentialSmoothingTracker(alpha=self._smoothing_alpha, dynamic_alpha=True)
            else:
                base_tracker = WelfordTracker()
            self.pdp_y_tracker = MultiValueTracker(copy.deepcopy(base_tracker))
            self.pdp_x_tracker = MultiValueTracker(copy.deepcopy(base_tracker))
            if self.storage_size is None:
                self.storage_size = self._storage.size

    @abc.abstractmethod
    def explain_one(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def _add_ice_curve_to_storage(self, *args, **kwargs):
        raise NotImplementedError

    def plot_pdp(self, title: str = None, x_min=None, x_max=None, y_max=None, y_min=None,
                 return_plot: bool = False, show_pdp_transition: bool = True,
                 x_transform=None, y_transform=None, batch_pdp=None,
                 show_ice_curves: bool = True, y_label="Model Output", figsize=None,
                 mean_centered_pd: bool = False, n_decimals=None, xticks=None,
                 xticklabels=None, show_legend=False, n_ice_curves_prop: float = 1.):
        if self.is_batch_pdp:
            if title is None:
                title = f"Batch PDP Curve for feature {self.pdp_feature}"
            ice_curves_x = np.asarray(self.ice_curves_x)
            n_ice_curves: int = int(len(ice_curves_x) * n_ice_curves_prop)
            idx = np.round(np.linspace(0, n_ice_curves - 1, n_ice_curves)).astype(int)

            mean_x = np.mean(ice_curves_x, axis=0)
            ice_curves_y = np.asarray(self.ice_curves_y)
            mean_y = np.mean(ice_curves_y, axis=0)

            ice_curves_x = ice_curves_x[idx]
            ice_curves_y = ice_curves_y[idx]

            alphas = [1.0] * len(ice_curves_x)

            fig, (axis, dist_axis) = plt.subplots(2, 1, height_ratios=(15, 1), figsize=figsize)

            if show_ice_curves:
                for ice_curve_x, ice_curve_y, alpha in zip(ice_curves_x, ice_curves_y, alphas):
                    axis.plot(ice_curve_x, ice_curve_y, ls='-', c='black', alpha=alpha, linewidth=1)
                axis.plot([], [], ls='-', c='black', label="ICE")

            axis.plot([], [], ls='-', c='red', label="PDP")

            axis.plot(mean_x, mean_y, ls='-', c='red', alpha=1., linewidth=5)

            # draw data distribution axis
            x_data, _ = self._storage.get_data()
            feature_values = list(pd.DataFrame(x_data)[self.pdp_feature].values[idx])
            if n_ice_curves_prop < 1:
                feature_values.append(np.min(mean_x))
                feature_values.append(np.max(mean_x))
            for x_value in feature_values:
                dist_axis.axvline(x_value, c="black", alpha=0.2)
            if x_min is None:
                x_min = np.min(mean_x) * 0.95
            if x_max is None:
                x_max = np.max(mean_x) * 1.05

        else:
            if title is None:
                title = f"Incremental PDP Curve for feature {self.pdp_feature}"

            alphas = np.linspace(start=0.1, stop=0.95, num=self.storage_size)

            x_transform = no_transform if x_transform is None else x_transform
            y_transform = no_transform if y_transform is None else y_transform

            # transform ice curves
            ice_curves_x, ice_curves_y = [], []
            for ice_curve_x, ice_curve_y in zip(self.ice_curves_x, self.ice_curves_y):
                ice_curves_x.append(x_transform(np.asarray(list(ice_curve_x.values()))))
                ice_curves_y.append(y_transform(np.asarray(list(ice_curve_y.values()))))

            # transform pdp
            pdp_x = x_transform(np.asarray(list(self.pdp_x_tracker.get().values())))
            pdp_y = y_transform(np.asarray(list(self.pdp_y_tracker.get().values())))

            # get plot
            fig, (axis, dist_axis) = plt.subplots(2, 1, height_ratios=(15, 1), figsize=figsize)

            # plot ice curves
            if show_ice_curves:
                for ice_curve_x, ice_curve_y, alpha in zip(ice_curves_x, ice_curves_y, alphas):
                    axis.plot(ice_curve_x, ice_curve_y, ls='-', c='black', alpha=alpha, linewidth=1)
                axis.plot([], [], ls='-', c='black', label="ICE curves")

            # plot the pdp history
            if show_pdp_transition:
                pdp_history_x, pdp_history_y = [], []
                for pdp_x_old, pdp_y_old in zip(self.pdp_storage_x, self.pdp_storage_y):
                    pdp_history_x.append(x_transform(np.asarray(list(pdp_x_old.values()))))
                    pdp_history_y.append(y_transform(np.asarray(list(pdp_y_old.values()))))
                n_pdps_in_storage = len(self.pdp_storage_x)
                pdp_alphas = np.linspace(start=0.04, stop=0.95, num=n_pdps_in_storage)
                for i, (pdp_x_old, pdp_y_old, alpha) in enumerate(
                        zip(pdp_history_x, pdp_history_y, pdp_alphas)):
                    if i >= n_pdps_in_storage - 1:
                        break
                    axis.plot(pdp_x_old, pdp_y_old, ls='-', c='#FFA800', alpha=alpha, linewidth=1)
                axis.plot([], [], ls='-', c='#FFA800', linewidth=1, label="iPDP (historic)")

            if batch_pdp is not None:
                batch_pdp_x, batch_pdp_y = batch_pdp
                axis.plot(
                    x_transform(batch_pdp_x), y_transform(batch_pdp_y),
                    ls='--', c='blue', alpha=1., linewidth=3, label="PDP (batch)")

            # plot the current PDP
            if mean_centered_pd:
                pdp_y = pdp_y - np.mean(pdp_y)

            axis.plot(pdp_x, pdp_y, ls='-', c='red', alpha=1., linewidth=4, label="iPDP (current)")

            # draw data distribution axis
            x_data, _ = self._storage.get_data()
            feature_values = pd.DataFrame(x_data)[self.pdp_feature].values
            feature_values = x_transform(feature_values)
            for x_value in feature_values:
                dist_axis.axvline(x_value, c="black", alpha=0.2)

            if x_min is None:
                x_min = np.min(pdp_x) * 0.95
            if x_max is None:
                x_max = np.max(pdp_x) * 1.05

        axis.set_title(title)
        axis.set_ylim((y_min, y_max))

        axis.set_ylabel(y_label)
        axis.set_xlim((x_min, x_max))
        dist_axis.set_xlim((x_min, x_max))

        dist_axis.set_xlabel(f"feature: {self.pdp_feature}")
        axis.set_xticklabels([])
        dist_axis.set_yticks([], [])

        if xticklabels is not None:
            dist_axis.set_xticks(xticks, xticklabels)

        if show_legend:
            axis.legend()

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        if n_decimals is not None:
            axis_labels_x = dist_axis.get_xticklabels
            try:
                axis_labels_x = [f"{x_value:.2f}" for x_value in axis_labels_x]
                dist_axis.set_xticklabels(axis_labels_x)
            except TypeError:
                pass

        if return_plot:
            return fig, (axis, dist_axis)

        plt.show()
        return None, None
