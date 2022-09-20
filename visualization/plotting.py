from abc import ABCMeta
from typing import Optional, Union

import matplotlib.pyplot as plt

from color import SAVE_DPI
from visualization.line_plots import plot_multi_line_graph


class BasePlotter(metaclass=ABCMeta):

    def __init__(self) -> None:
        self.seen_timesteps = 0

    def save(self,
             save_file_path: Optional[str] = None,
             save_dpi: Optional[int] = SAVE_DPI) -> None:
        pass


class FeatureImportancePlotter(BasePlotter):

    def __init__(self,
                 feature_names: Optional[list[str]] = None):
        super().__init__()
        self.feature_names = feature_names
        self.y_data = {}
        self.x_data = {}
        self.stored_facets = {}

    def _create_new_facet(self, facet_name: str):
        self.stored_facets.update(facet_name)
        self.y_data[facet_name] = []
        self.x_data[facet_name] = []


    def update(self,
               importance_values: dict[str, Union[int, float]],
               facet_name: Optional[str] = None,
               timestep: Optional[Union[int, float]] = None):
        self.seen_timesteps += 1
        if facet_name is not None and facet_name not in self.stored_facets:
            self._create_new_facet(facet_name)
        if timestep is None:
            timestep = self.seen_timesteps
        self.y_data[facet_name].append(importance_values)
        self.x_data[facet_name].append(timestep)

    def plot(self,
             figsize: Optional[tuple[int]] = None
             ) -> None:
        fig, axis = plt.subplot(figsize=figsize)
        axis = plot_multi_line_graph(axis=axis,
                                     y_data=self.y_data,
                                     names_to_highlight=self.feature_names,
                                     line_names=self.feature_names)
        plt.show()

