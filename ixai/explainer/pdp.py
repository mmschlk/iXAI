from ixai.explainer.base import BasePDP
from collections import OrderedDict
from ixai.storage import BatchStorage
from ixai.utils.tracker import ExtremeValueTracker
import numpy as np
import pandas as pd


def no_transform(values):
    return values


class IncrementalPDP(BasePDP):

    def __init__(self, model_function,
                 pdp_feature, gridsize, storage,
                 smoothing_alpha, dynamic_setting,
                 storage_size, output_key=1, pdp_history_size=10, pdp_history_interval=1000,
                 min_max_grid: bool = False, ylim=None):
        super(IncrementalPDP, self).__init__(model_function=model_function, pdp_feature=pdp_feature,
                                             gridsize=gridsize, output_key=output_key, ylim=ylim,
                                             smoothing_alpha=smoothing_alpha,
                                             dynamic_setting=dynamic_setting, is_batch_pdp=False)
        self._smoothing_alpha = 0.001 if smoothing_alpha is None else smoothing_alpha
        self.seen_samples = 0
        self._storage = storage
        self.storage_size = storage_size
        self.waiting_period = 20
        self.pdp_storage_interval = pdp_history_interval
        self.pdp_storage_size = pdp_history_size
        self.min_max_grid = min_max_grid
        if self.min_max_grid:
            self._max_tracker = ExtremeValueTracker(window_length=1000, size=50)
            self._min_tracker = ExtremeValueTracker(window_length=1000, size=50, higher_is_better=False)

    def _add_ice_curve_to_pdp(self, ice_curve_y, ice_curve_x):
        self.pdp_y_tracker.update(ice_curve_y)
        self.pdp_x_tracker.update(ice_curve_x)

    def _add_ice_curve_to_storage(self, ice_curve_y, ice_curve_x):
        if self.seen_samples < (self.storage_size + self.waiting_period):
            self.ice_curves_y.append(ice_curve_y)
            self.ice_curves_x.append(ice_curve_x)
        else:
            self.ice_curves_y.popleft()
            self.ice_curves_x.popleft()
            self.ice_curves_y.append(ice_curve_y)
            self.ice_curves_x.append(ice_curve_x)

    def explain_one(
            self,
            x_i
    ):
        # Warm up for explanations
        if self.seen_samples <= self.waiting_period:
            if self.min_max_grid:
                self._min_tracker.update(x_i[self.pdp_feature])
                self._max_tracker.update(x_i[self.pdp_feature])
            else:
                self._storage.update(x=x_i)
            self.seen_samples += 1
        else:

            if self.min_max_grid:
                min_value = self._min_tracker.get()
                max_value = self._max_tracker.get()
            else:
                x_data, _ = self._storage.get_data()
                x_data = pd.DataFrame(x_data)
                min_value = np.quantile(x_data[self.pdp_feature], q=0.05)
                max_value = np.quantile(x_data[self.pdp_feature], q=0.95)

            feature_grid_values = np.linspace(start=min_value, stop=max_value, num=self.gridsize)
            feature_grid_dict = OrderedDict({i: value for i, value in enumerate(feature_grid_values)})
            predictions_dict = OrderedDict()
            for i, sampled_feature in enumerate(feature_grid_values):
                prediction = self.model_function({**x_i, self.pdp_feature: sampled_feature})[self.output_key]
                predictions_dict[i] = prediction
            self._add_ice_curve_to_pdp(ice_curve_y=predictions_dict, ice_curve_x=feature_grid_dict)
            self._add_ice_curve_to_storage(ice_curve_y=predictions_dict, ice_curve_x=feature_grid_dict)
            if self.min_max_grid:
                self._min_tracker.update(x_i[self.pdp_feature])
                self._max_tracker.update(x_i[self.pdp_feature])
            self._storage.update(x=x_i)
            self.seen_samples += 1
            if self.seen_samples % self.pdp_storage_interval == 0:
                self._add_pdp_to_storage()

    def _add_pdp_to_storage(self):
        if len(self.pdp_storage_x) < self.pdp_storage_size:
            self.pdp_storage_x.append(OrderedDict(self.pdp_x_tracker.get()))
            self.pdp_storage_y.append(OrderedDict(self.pdp_y_tracker.get()))
        else:
            self.pdp_storage_x.popleft()
            self.pdp_storage_y.popleft()
            self.pdp_storage_x.append(OrderedDict(self.pdp_x_tracker.get()))
            self.pdp_storage_y.append(OrderedDict(self.pdp_y_tracker.get()))


class BatchPDP(BasePDP):

    def __init__(self, pdp_feature, gridsize, model_function, ylim=None, storage=None, output_key=1):
        super(BatchPDP, self).__init__(model_function=model_function, pdp_feature=pdp_feature,
                                       gridsize=gridsize, output_key=output_key, ylim=ylim, is_batch_pdp=True)
        if storage is None:
            self._storage = BatchStorage(store_targets=False)
        else:
            self._storage = storage
        self.ice_curves_y = []
        self.ice_curves_x = []

    @property
    def pdp(self):
        ice_curves_x = np.asarray(self.ice_curves_x)
        pdp_x = np.mean(ice_curves_x, axis=0)
        ice_curves_y = np.asarray(self.ice_curves_y)
        pdp_y = np.mean(ice_curves_y, axis=0)
        return pdp_x, pdp_y

    def _add_ice_curve_to_storage(self, ice_curve_y, ice_curve_x):
        self.ice_curves_y.append(ice_curve_y)
        self.ice_curves_x.append(ice_curve_x)

    def update_storage(self, x_i: dict):
        self._storage.update(x=x_i)

    def explain_one(self, x_i):
        self._storage.update(x_i)
        x_data, _ = self._storage.get_data()
        self.explain_many(x_data=x_data)

    def explain_many(self, x_data):
        x_data = pd.DataFrame(x_data)
        min_value = np.min(x_data[self.pdp_feature])
        max_value = np.max(x_data[self.pdp_feature])
        feature_grid_values = np.linspace(start=min_value, stop=max_value, num=self.gridsize)
        for x_i in x_data.to_dict('records'):
            predictions = np.empty(shape=self.gridsize)
            for i, sampled_feature in enumerate(feature_grid_values):
                try:
                    prediction = self.model_function({**x_i, self.pdp_feature: sampled_feature})[self.output_key]
                except TypeError:
                    prediction = self.model_function({**x_i, self.pdp_feature: sampled_feature})
                predictions[i] = prediction
            self._add_ice_curve_to_storage(predictions, feature_grid_values)
