"""
This module gathers SAGE Explanation Methods
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>
#          Rohit Jagtani

from typing import Optional, Union, Callable
from .base import BaseIncrementalFeatureImportance
import numpy as np
from river.metrics.base import Metric

__all__ = [
    "IncrementalPFI",
]


class IncrementalPFI(BaseIncrementalFeatureImportance):

    def __init__(
            self,
            model_function: Callable,
            feature_names: list[str],
            storage,
            imputer,
            loss_function: Union[Callable, Metric],
            n_inner_samples: int = 5,
            smoothing_alpha: float = 0.001,
            dynamic_setting: bool = True
    ):
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
        if self.seen_samples >= 1:
            if n_inner_samples is None:
                n_inner_samples = self.n_inner_samples
            original_prediction = self._model_function(x_i)[0]
            original_loss = self._loss_function(y_i, original_prediction)
            for feature in self.feature_names:
                feature_subset = [feature]
                predictions = self._imputer.impute(
                    feature_subset=feature_subset,
                    x_i=x_i,
                    n_samples=n_inner_samples
                )
                losses = []
                for prediction in predictions:
                    loss = self._loss_function(y_i, prediction[0])
                    losses.append(loss)
                avg_loss = np.mean(losses)
                pfi = avg_loss - original_loss
                self._update_pfi(pfi, feature)
                self._variance_trackers[feature].update((pfi - self.importance_values[feature]) ** 2)
        self.seen_samples += 1
        if update_storage:
            self._storage.update(x_i, y_i)
        return self.importance_values

    def _update_pfi(self, value, feature):
        self._importance_trackers[feature].update(value)
