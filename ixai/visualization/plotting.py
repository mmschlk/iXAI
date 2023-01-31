import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Sequence, Dict, List, Tuple

import matplotlib.pyplot as plt

from ixai.visualization.config import SAVE_DPI
from ixai.visualization.line_plots import plot_multi_line_graph

__all__ = [
    "FeatureImportancePlotter"
]


class BasePlotter(metaclass=ABCMeta):

    def __init__(self) -> None:
        self.seen_timesteps = 0

    def __repr__(self):
        return f"Plotter with {self.seen_timesteps} items."

    def save(self,
             save_file_path: Optional[str] = None,
             save_dpi: Optional[int] = SAVE_DPI) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError


class FeatureImportancePlotter(BasePlotter):

    def __init__(self,
                 feature_names: Optional[List[str]] = None):
        super().__init__()
        self.feature_names = feature_names
        self.y_data = {}
        self.x_data = {}
        self.performance_data = {}
        self.stored_facets = set()
        self._default_performance_line_name = 'perf'

    def _create_new_facet(self, facet_name: str):
        self.stored_facets.add(facet_name)
        self.y_data[facet_name] = []
        self.x_data[facet_name] = []

    def update(self,
               importance_values: Dict[str, Union[int, float]],
               facet_name: Optional[str] = None,
               timestep: Optional[Union[int, float]] = None):
        if timestep is None:
            self.seen_timesteps = self.seen_timesteps + 1
        if facet_name is None:
            facet_name = 'importance_values'
        if facet_name not in self.stored_facets:
            self._create_new_facet(facet_name)
        self.y_data[facet_name].append(importance_values)
        self.x_data[facet_name].append(timestep)

    def update_performance(self, performance_value: float, facet_name: str):
        if facet_name not in self.performance_data:
            self.performance_data[facet_name] = []
        self.performance_data[facet_name].append({'perf': performance_value})

    def plot(
            self,
            figsize: Optional[Tuple[int, int]] = None,
            model_performances: Optional[Dict[str, Sequence]] = None,
            performance_kw: Optional[dict] = None,
            save_name: Optional[str] = None,
            **plot_kw,
    ) -> None:

        if 'names_to_highlight' not in plot_kw:
            plot_kw['names_to_highlight'] = self.feature_names
        if 'line_names' not in plot_kw:
            plot_kw['line_names'] = self.feature_names
        if 'x_label' not in plot_kw:
            plot_kw['x_label'] = 'Samples'
        if 'h_lines' not in plot_kw:
            plot_kw['h_lines'] = [{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}]
        if 'y_label' not in plot_kw:
            plot_kw['y_label'] = 'Feature Importance Values'
        if figsize is None:
            figsize = (5, 10)

        if model_performances is not None or len(self.performance_data) > 0:
            fig, (performance_axis, fi_axis) = plt.subplots(
                nrows=2, ncols=1, sharex='col', sharey='row',
                figsize=figsize, gridspec_kw={'height_ratios': [1, 4]}
            )
            plt.subplots_adjust(hspace=0.000, wspace=0.000)

            title = None
            if 'title' in plot_kw:
                title = copy.copy(plot_kw['title'])
                del plot_kw['title']
            v_lines = None
            if 'v_lines' in plot_kw:
                v_lines = plot_kw['v_lines']
            performance_kw = {} if performance_kw is None else performance_kw
            if model_performances is None:
                model_performances = self.performance_data
            if 'line_names' not in performance_kw:
                performance_kw['line_names'] = [self._default_performance_line_name]
            if 'color_list' not in performance_kw:
                performance_kw['color_list'] = ["red", "black"]
            if 'y_label' not in performance_kw:
                performance_kw['y_label'] = "Perf."

            performance_axis = plot_multi_line_graph(
                axis=performance_axis,
                y_data=model_performances,
                title=title,
                v_lines=v_lines,
                **performance_kw
            )
        else:
            fig, fi_axis = plt.subplots()

        if figsize is not None:
            fig.set_figheight(figsize[0])
            fig.set_figwidth(figsize[1])

        fi_axis = plot_multi_line_graph(
            axis=fi_axis,
            y_data=self.y_data,
            #x_data=self.x_data, TODO fix bug it's not plotting
            **plot_kw
        )
        if save_name is not None:
            plt.savefig(save_name, dpi=200)
        plt.show()
