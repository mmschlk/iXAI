from ixai.explainer.base import BaseIncrementalExplainer
from ixai.utils.tracker import MultiValueTracker, WelfordTracker, ExponentialSmoothingTracker
from collections import deque, OrderedDict
from ixai.storage import BatchStorage
from ixai.utils.tracker import ExtremeValueTracker

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import copy


def no_transform(values):
    return values


class IncrementalPDP(BaseIncrementalExplainer):

    def __init__(self, model_function, feature_names,
                 pdp_feature, gridsize, storage,
                 smoothing_alpha, dynamic_setting,
                 storage_size, output_key=1, pdp_history_size=10, pdp_history_interval=1000,
                 min_max_grid: bool = False, **kwargs):
        super(IncrementalPDP, self).__init__(model_function=model_function, feature_names=feature_names)
        self.pdp_feature = pdp_feature
        self.model_function = model_function
        self.gridsize = gridsize
        self.ylim = (0., 1)
        self._smoothing_alpha = 0.001 if smoothing_alpha is None else smoothing_alpha
        if dynamic_setting:
            assert 0. < smoothing_alpha <= 1., f"The smoothing parameter needs to be in the range" \
                                               f" of ']0,1]' and not " \
                                               f"'{self._smoothing_alpha}'."
            base_tracker = ExponentialSmoothingTracker(alpha=self._smoothing_alpha, dynamic_alpha=True)
        else:
            base_tracker = WelfordTracker()
        self.pdp_y_tracker = MultiValueTracker(copy.deepcopy(base_tracker))
        self.pdp_x_tracker = MultiValueTracker(copy.deepcopy(base_tracker))
        self.ice_curves_y = deque()
        self.ice_curves_x = deque()
        self.seen_samples = 0
        self.storage = storage
        # TODO - Remove this
        self.storage_size = storage_size
        self.waiting_period = 20
        self.pdp_storage_x = deque()
        self.pdp_storage_y = deque()
        self.pdp_storage_interval = pdp_history_interval
        self.pdp_storage_size = pdp_history_size
        self.output_key = output_key
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
                self.storage.update(x=x_i)
            self.seen_samples += 1
        else:

            if self.min_max_grid:
                min_value = self._min_tracker.get()
                max_value = self._max_tracker.get()
            else:
                x_data, _ = self.storage.get_data()
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
            self.storage.update(x=x_i)
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

    def plot_pdp(self, title: str = None, x_min=None, x_max=None, y_max=None, y_min=None,
                 return_plot: bool = False, show_pdp_transition: bool = True,
                 x_transform=None, y_transform=None, batch_pdp=None, y_scale=None, show_ice_curves: bool = True,
                 y_label="Model Output", figsize=None, mean_centered_pd: bool = False, n_decimals=None,
                 xticks=None, xticklabels=None, show_legend=False):
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
        x_data, _ = self.storage.get_data()
        feature_values = pd.DataFrame(x_data)[self.pdp_feature].values
        feature_values = x_transform(feature_values)
        for x_value in feature_values:
            dist_axis.axvline(x_value, c="black", alpha=0.2)

        axis.set_title(title)
        axis.set_ylim((y_min, y_max))

        axis.set_ylabel(y_label)
        xlim_lower = x_min
        if x_min is None:
            xlim_lower = np.min(pdp_x) * 0.95
        xlim_upper = x_max
        if x_max is None:
            xlim_upper = np.max(pdp_x) * 1.05

        axis.set_xlim((xlim_lower, xlim_upper))
        dist_axis.set_xlim((xlim_lower, xlim_upper))

        dist_axis.set_xlabel(f"feature: {self.pdp_feature}")
        # dist_axis.set_yticks([])
        # axis.set_xticks([])

        axis.set_xticklabels([])
        dist_axis.set_yticks([], [])
        # dist_axis.set_yticklabels([])

        if xticklabels is not None:
            dist_axis.set_xticks(xticks, xticklabels)

        if show_legend:
            axis.legend()

        if y_scale is not None:
            axis.set_yscale(y_scale)

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


class BatchPDP:

    def __init__(self, pdp_feature, gridsize, model_function, ylim=None, storage=None, output_key=1):
        if storage is None:
            self._storage = BatchStorage(store_targets=False)
        else:
            self._storage = storage
        self.pdp_feature = pdp_feature
        self.gridsize = gridsize
        if ylim is None:
            self.ylim = (0, 1)
        else:
            self.ylim = ylim
        self.model_function = model_function
        self.ice_curves_y = []
        self.ice_curves_x = []
        self.output_key = output_key

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

    def plot_pdp(self, title: str = None,  x_min=None, x_max=None, y_max=None, y_min=None,
                 return_plot=False, n_ice_curves_prop: float = 1., xticks=None, xticklabels=None,
                 y_label="Model Output", show_legend: bool = False, figsize=None, show_ice_curves=True):
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

        axis.set_title(title)
        axis.set_ylim((y_min, y_max))

        axis.set_title(title)
        axis.set_ylim((y_min, y_max))

        axis.set_ylabel(y_label)
        xlim_lower = x_min
        if x_min is None:
            xlim_lower = np.min(mean_x) * 0.95
        xlim_upper = x_max
        if x_max is None:
            xlim_upper = np.max(mean_x) * 1.05

        axis.set_xlim((xlim_lower, xlim_upper))
        dist_axis.set_xlim((xlim_lower, xlim_upper))

        dist_axis.set_xlabel(f"feature: {self.pdp_feature}")
        axis.set_xticklabels([])
        dist_axis.set_yticks([], [])

        if xticklabels is not None:
            dist_axis.set_xticks(xticks, xticklabels)

        if show_legend:
            axis.legend()

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        if return_plot:
            return fig, (axis, dist_axis)

        plt.show()
        return None, None