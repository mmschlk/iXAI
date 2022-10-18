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
            model,
            feature_names,
            storage,
            imputer,
            loss_function='mse',
            nsamples: int = 5,
            alpha: float = 0.005
    ):
        super(IncrementalPFI, self).__init__(
            model=model
        )
        self.feature_names = feature_names
        self.loss_function = loss_function
        self.storage = storage
        self.imputer = imputer
        self.alpha = alpha
        self.pfi_trackers = {feature_name:
                             ExponentialSmoothingTracker(alpha=self.alpha)
                             for feature_name in feature_names}
        self.nsamples = nsamples

    def explain_one(self, x_i, y_i, nsamples=None):
        if nsamples is None:
            nsamples = self.nsamples
        original_prediction = self.model.predict_one(x_i)
        original_loss = self._calculate_loss(y_i, original_prediction)
        for feature in self.feature_names:
            feature_subset = [feature]
            predictions = self.imputer.impute(feature_subset,
                                              x_i, nsamples)
            losses = []
            for prediction in predictions:
                loss = self._calculate_loss(y_i, prediction)
                losses.append(loss)
            avg_loss = np.mean(losses)
            # TODO - keep argument for ratio/constant in init - separate issue
            pfi = avg_loss - original_loss
            self._update_pfi(pfi, feature)
        self.storage.update(x_i, y_i)
        return self.pfi_values

    def _update_pfi(self, value, feature):
        self.pfi_trackers[feature].update(value)

    def _calculate_loss(self, y_i, prediction):
        if self.loss_function == 'mae':
            loss = mae_loss(y_i, prediction)
        else:
            loss = mse_loss(y_i, prediction)
        return loss

    @property
    def pfi_values(self):
        return {feature_name:
                float(self.pfi_trackers[feature_name].tracked_value)
                for feature_name in self.feature_names}
