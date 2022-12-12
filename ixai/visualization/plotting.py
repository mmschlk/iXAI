import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Sequence

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
                 feature_names: Optional[list[str]] = None):
        super().__init__()
        self.feature_names = feature_names
        self.y_data = {}
        self.x_data = {}
        self.stored_facets = set()

    def _create_new_facet(self, facet_name: str):
        self.stored_facets.add(facet_name)
        self.y_data[facet_name] = []
        self.x_data[facet_name] = []

    def update(self,
               importance_values: dict[str, Union[int, float]],
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

    def plot(
            self,
            figsize: Optional[tuple[int, int]] = None,
            model_performances: Optional[dict[str, Sequence]] = None,
            performance_kw: Optional[dict] = None,
            save_name: Optional[str] = None,
            **plot_kw,
    ) -> None:

        if 'names_to_highlight' not in plot_kw:
            plot_kw['names_to_highlight'] = self.feature_names
        if 'line_names' not in plot_kw:
            plot_kw['line_names'] = self.feature_names

        if model_performances is not None:
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
