import copy
import typing

from matplotlib import pyplot as plt

from ixai.visualization.color import BACKGROUND_COLOR

WATERFALL_COLORS = {False: 'red', True: 'green'}


def plot_water_fall_graph(
        axes: typing.List[plt.axis],
        y_data: typing.Dict[str, typing.List[float]],
        *,
        x_data: typing.Dict[str, typing.List[int]] = None,
        show_last_n: typing.Optional[int] = None,
        **kwargs
):
    if 'h_lines' in kwargs:
        for h_line_props in kwargs['h_lines']:
            for axis in axes:
                axis.axhline(**h_line_props)

    color_dict = WATERFALL_COLORS
    if 'color' in kwargs:
        color_dict = kwargs['color']

    line_names = list(y_data.keys())

    min_line_value = min([min(values) for values in x_data.values()])
    max_line_value = max([max(values) for values in x_data.values()])

    x_values = {i: i - min_line_value for i in range(min_line_value, max_line_value + 1)}  # {5:0, 6:1, 7:2}

    first_axis = axes[0]
    last_axis = axes[-1]

    for line_name, axis in zip(line_names, axes):
        line_values = [0., *y_data[line_name]]
        diffs = [line_values[i + 1] - line_values[i] for i in range(len(line_values) - 1)]
        bottoms = line_values[0:-1]
        colors = [color_dict[diffs[i] > 0] for i in range(len(diffs))]
        axis.bar(x_data[line_name], diffs, bottom=bottoms, color=colors)
        axis.set_xlabel(line_name)
        axis.grid(True, linestyle='dotted')
        axis.set_facecolor(BACKGROUND_COLOR)

    if 'ylabel' in kwargs:
        first_axis.set_ylabel(kwargs['ylabel'])

    if 'y_ticks' in kwargs:
        first_axis.set_yticks(kwargs['y_ticks'])
        first_axis.grid(True)
        for axis in axes[1:]:
            axis.set_yticks(kwargs['y_ticks'])
            axis.set_yticklabels([])

    if 'x_ticks' in kwargs:
        for axis in axes:
            axis.set_xticks(kwargs['x_ticks'])

    if 'y_min' in kwargs:
        plt.ylim(bottom=kwargs['y_min'])

    if 'y_max' in kwargs:
        plt.ylim(top=kwargs['y_max'])

    if 'title' in kwargs:
        plt.suptitle(kwargs['title'])

    plt.subplots_adjust(wspace=0, hspace=0)
    return axes
