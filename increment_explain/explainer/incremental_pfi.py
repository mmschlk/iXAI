from typing import Callable, Optional
from .base_incremental_explainer import BaseIncrementalExplainer
import numpy as np
from utils.trackers import ExponentialSmoothingTracker
from utils.loss_functions import mse_loss, mae_loss
# from imputer.default_imputer import DefaultImputer

__all__ = [
    "IncrementalPFI",
]


class IncrementalPFI(BaseIncrementalExplainer):

    def __init__(
            self,
            model_function,
            feature_names,
            storage,
            imputer,
            loss_function: Callable,
            n_samples: int = 5,
            smoothing_alpha: float = 0.005,
            dynamic_setting: bool = True
    ):
        super(IncrementalPFI, self).__init__(
            model_function=model_function,
            feature_names=feature_names,
            dynamic_setting=dynamic_setting,
            smoothing_alpha=smoothing_alpha
        )
        self._loss_function = loss_function
        self.storage = storage
        self.imputer = imputer
        self.n_samples = n_samples

    def explain_one(
            self,
            x_i,
            y_i,
            n_samples: Optional[int] = None
    ):
        if self.seen_samples >= 1:
            if n_samples is None:
                n_samples = self.n_samples
            original_prediction = self.model_function(x_i)
            original_loss = self._loss_function(y_true=y_i, y_prediction=original_prediction)
            for feature in self.feature_names:
                feature_subset = [feature]
                predictions = self.imputer.impute(feature_subset, x_i, n_samples)
                losses = []
                for prediction in predictions:
                    loss = self._loss_function(y_true=y_i, y_prediction=prediction)
                    losses.append(loss)
                avg_loss = np.mean(losses)
                # TODO - keep argument for ratio/constant in init - separate issue
                pfi = avg_loss - original_loss
                self._update_pfi(pfi, feature)
        self.storage.update(x_i, y_i)
        self.seen_samples += 1
        return self.pfi_values

    def _update_pfi(self, value, feature):
        self.importance_trackers[feature].update(value)

    @property
    def pfi_values(self):
        return {feature_name:
                float(self.importance_trackers[feature_name].tracked_value)
                for feature_name in self.feature_names}
