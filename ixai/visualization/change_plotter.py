import copy
import typing

from matplotlib import pyplot as plt

from ixai.visualization.line_plots import plot_multi_line_graph
from ixai.visualization.plotting import BasePlotter
from ixai.visualization.waterfall_plots import plot_water_fall_graph


class ChangePlotter(BasePlotter):

    def __init__(self):
        super().__init__()
        self.y_data = {}
        self.x_data = {}
        self.stored_feature_names = set()

    @property
    def n_features_stored(self):
        return len(self.stored_feature_names)

    def _store_new_feature(self, feature_name: str):
        self.stored_feature_names.add(feature_name)
        self.y_data[feature_name] = []
        self.x_data[feature_name] = []

    def update(
            self,
            importance_values: dict,
    ):
        self.seen_timesteps += 1
        for feature_name, feature_value in importance_values.items():
            if feature_name not in self.stored_feature_names:
                self._store_new_feature(feature_name)
            self.y_data[feature_name].append(feature_value)
            self.x_data[feature_name].append(self.seen_timesteps)

    def plot(
            self,
            figsize: typing.Optional[typing.Tuple[int, int]] = None,
            save_name: typing.Optional[str] = None,
            model_performances: typing.Optional[typing.Dict[typing.Any, typing.Sequence]] = None,
            performance_kw: typing.Optional[dict] = None,
            **plot_kw
    ) -> None:

        n_features = self.n_features_stored
        line_names = list(self.stored_feature_names)
        if 'line_names' in plot_kw:
            line_names = plot_kw['line_names']
            n_features = len(line_names)

        if model_performances is not None:

            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, n_features, height_ratios=[1, 4])
            performance_axis = fig.add_subplot(gs[0, :])
            fi_axis = [fig.add_subplot(gs[1, i]) for i in range(0, n_features)]
            for i in range(0, len(fi_axis) - 1):
                fi_axis[i].sharey(fi_axis[i + 1])

            title = None
            if 'title' in plot_kw:
                title = copy.copy(plot_kw['title'])
                del plot_kw['title']
            performance_kw = {} if performance_kw is None else performance_kw
            performance_axis = plot_multi_line_graph(
                axis=performance_axis,
                y_data=model_performances,
                title=title,
                **performance_kw
            )
        else:
            fig, fi_axis = plt.subplots(1, n_features, sharey='all')

        if figsize is not None:
            fig.set_figheight(figsize[0])
            fig.set_figwidth(figsize[1])

        fi_axis = plot_water_fall_graph(
            axes=fi_axis,
            y_data=self.y_data,
            x_data=self.x_data,
            **plot_kw
        )

        plt.tight_layout()

        if model_performances is not None:
            plt.subplots_adjust(wspace=0.000, hspace=0.3)
        else:
            plt.subplots_adjust(wspace=0.000)

        if save_name is not None:
            plt.savefig(save_name, dpi=200)
        plt.show()
