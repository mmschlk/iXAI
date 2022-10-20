from collections import deque
from typing import Callable, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from increment_explain.explainer import BaseIncrementalExplainer
from increment_explain.storage import UniformReservoirStorage
from increment_explain.storage.base import BaseStorage


class PartialDependenceExplainer(BaseIncrementalExplainer):

    def __init__(
            self,
            model_function: Callable,
            feature_names: list[str],
            feature_name: str,
            xlim: tuple[Any, Any],
            storage_size: int = 100
    ):
        super(PartialDependenceExplainer, self).__init__(model_function=model_function, feature_names=feature_names)
        self.ylim = (0., 1)
        self.xlim = xlim
        self.ice_curves_y = deque()
        self.ice_curves_x = deque()
        self.seen_samples = 0
        self.storage_size = storage_size
        self.feature_name = feature_name

    def _add_ice_curve_to_storage(self, ice_curve_y, ice_curve_x):
        if self.seen_samples < self.storage_size:
            self.ice_curves_y.append(ice_curve_y)
            self.ice_curves_x.append(ice_curve_x)
        else:
            self.ice_curves_y.popleft()
            self.ice_curves_x.popleft()
            self.ice_curves_y.append(ice_curve_y)
            self.ice_curves_x.append(ice_curve_x)
        self.seen_samples += 1

    def explain_one(
            self,
            x_i: dict[str, Any],
            storage: BaseStorage,
            sample_frequency: int = 50,
    ):
        x_data, _ = storage.get_data()
        x_data_df = pd.DataFrame(x_data)

        min_value = x_data_df[self.feature_name].quantile(q=0.05)
        max_value = x_data_df[self.feature_name].quantile(q=0.95)

        features_samples = np.linspace(start=min_value, stop=max_value, num=sample_frequency)
        predictions = np.empty(shape=sample_frequency)
        for i, sampled_feature in enumerate(features_samples):
            predictions[i] = self.model_function({**x_i, self.feature_name: sampled_feature})

        self._add_ice_curve_to_storage(ice_curve_y=predictions, ice_curve_x=features_samples)

    def plot_pdp(self, title: str = None):
        if title is not None:
            title = f"PDP Curve for feature {self.feature_name}"

        ice_curves_x = np.asarray(self.ice_curves_x)
        mean_x = np.mean(ice_curves_x, axis=0)
        ice_curves_y = np.asarray(self.ice_curves_y)
        mean_y = np.mean(ice_curves_y, axis=0)

        alphas = np.linspace(start=0.1, stop=1., num=self.storage_size)

        fig, axis = plt.subplots(1, 1)
        for ice_curve_x, ice_curve_y, alpha in zip(self.ice_curves_x, self.ice_curves_y, alphas):
            axis.plot(ice_curve_x, ice_curve_y, ls='-', c='black', alpha=alpha, linewidth=1)

        axis.plot(mean_x, mean_y, ls='-', c='red', alpha=1., linewidth=2)

        plt.title(title)
        plt.ylim(self.ylim)
        plt.xlabel(f"feature: {self.feature_name}")
        plt.ylabel("Model Output")
        plt.xlim(self.xlim)
        plt.show()

    def plot_ice_curve(self, ice_curve_y, ice_curve_x, title: str = None):
        if title is not None:
            title = f"ICE Curve for feature {self.feature_name}"

        plt.plot(ice_curve_x, ice_curve_y)
        plt.title(title)
        plt.ylim(self.ylim)
        plt.xlabel(f"{self.feature_name} range")
        plt.ylabel("Model Output")
        plt.xlim(self.xlim)
        plt.show()
