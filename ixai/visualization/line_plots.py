import copy
from typing import Optional, Union, Sequence, Tuple, Dict, List

import pandas as pd
import matplotlib.pyplot as plt

from ixai.visualization.color import color_list_generator, get_color_with_generator, BACKGROUND_COLOR, BASE_COLOR, STD_ALPHA


def _validate_y_x_data(y_data: Union[dict, Sequence],
                       x_data: Optional[Union[dict, Sequence]] = None,
                       line_names: Optional[List[str]] = None) -> Tuple[dict, dict, List[str]]:
    if isinstance(y_data, dict):
        y_keys = set(y_data.keys())
    else:
        y_data = {'data': y_data}
        y_keys = {'data'}

    for y_key in y_keys:
        y_data_facet = y_data[y_key]
        if isinstance(y_data_facet, list) and isinstance(y_data_facet[0], dict):
            y_data[y_key] = pd.DataFrame(y_data_facet)
        elif not isinstance(y_data_facet, pd.Series) and not isinstance(y_data_facet, pd.DataFrame):
            if line_names is None:
                line_names = ['X' + str(i) for i in range(len(y_data_facet))]
            y_data[y_key] = pd.DataFrame(y_data_facet, columns=line_names)

    if x_data is not None and isinstance(x_data, dict):
        x_keys = list(x_data.keys())
    else:
        if x_data is None:
            x_data = {y_key: list(range(len(y_data[y_key]))) for y_key in y_keys}
        else:
            x_data = {y_key: x_data for y_key in y_keys}
        x_keys = list(x_data.keys())

    for y_key in y_keys:
        if y_key not in x_keys:
            x_data[y_key] = x_data[x_keys[0]]

    return copy.deepcopy(y_data), copy.deepcopy(x_data), copy.deepcopy(line_names)


def _validate_std(std: Optional[dict],
                  y_data: Optional[dict],
                  line_names: List[str]) -> Optional[dict]:
    if std is None:
        return None
    if not isinstance(std, dict):
        std_dict = {}
        if len(y_data) > 1:
            for n, y_key in enumerate(y_data.keys()):
                std_dict[y_key] = pd.DataFrame(std[n], columns=line_names)
        else:
            std_dict[list(y_data.keys())[0]] = pd.DataFrame(std, columns=line_names)
        std = std_dict
    else:
        for std_key in std.keys():
            if not isinstance(std[std_key], pd.Series) and not isinstance(std[std_key], pd.DataFrame):
                std[std_key] = pd.DataFrame(std[std_key], columns=line_names)
    return std


def plot_multi_line_graph(
        axis: plt.axis,
        y_data: Union[dict, Sequence],
        *,
        x_data: Optional[Union[dict, Sequence]] = None,
        line_names: Optional[Union[dict, Sequence]] = None,
        names_to_highlight: Optional[List[str]] = None,
        facet_not_to_highlight: Optional[List[str]] = None,
        line_styles: Optional[Dict[str, str]] = None,
        std: Optional[Union[dict, Sequence]] = None,
        title: Optional[str] = None,
        y_label: Optional[str] = None,
        x_label: Optional[str] = None,
        y_ticks: Optional[List[Union[int, float]]] = None,
        x_ticks: Optional[List[Union[int, float]]] = None,
        y_min: Optional[Union[int, float]] = None,
        y_max: Optional[Union[int, float]] = None,
        x_min: Optional[Union[int, float]] = None,
        x_max: Optional[Union[int, float]] = None,
        legend_style: Optional[dict] = None,
        secondary_legends: Optional[List[Dict[str, dict]]] = None,
        base_color: Optional[str] = None,
        color_list: Optional[List[str]] = None,
        h_lines: Optional[List[dict]] = None,
        v_lines: Optional[List[dict]] = None,
        markevery: Optional[Dict[str, int]] = None,
        tick_right: bool = False,
        fill_between_props: Optional[List[dict]] = None,
        xticklabels: Optional[List[str]] = None,
        text: Optional[List[dict]] = None
) -> plt.axis:

    if facet_not_to_highlight is None:
        facet_not_to_highlight = []

    if names_to_highlight is None:
        names_to_highlight = line_names if line_names is not None else []

    y_data, x_data, line_names = _validate_y_x_data(y_data=y_data,
                                                    x_data=x_data,
                                                    line_names=line_names)
    std = _validate_std(std=std, y_data=y_data, line_names=line_names)

    color_gens = {}
    for facet in y_data.keys():
        color_gens[facet] = color_list_generator(color_list=color_list)

    if line_styles is None:
        line_styles = {facet: '-' for facet in y_data.keys()}

    if markevery is None:
        markevery = {facet: 1 for facet in y_data.keys()}

    line_colors = {}

    for facet, y_facet in y_data.items():
        x_facet = x_data[facet]
        std_facet = None
        if std is not None:
            if facet in std:
                std_facet = std[facet]
        for line_name in line_names:
            color_line = get_color_with_generator(color_generator=color_gens[facet],
                                                  base_color=base_color,
                                                  item_id=line_name,
                                                  color_item_ids=names_to_highlight)
            line_colors[line_name] = color_line
            alpha = 1.
            if line_name not in names_to_highlight or facet in facet_not_to_highlight:
                alpha = 0.6
            axis.plot(x_facet[::markevery[facet]],
                      y_facet[line_name][::markevery[facet]],
                      ls=line_styles[facet],
                      c=color_line,
                      alpha=alpha,
                      linewidth=1)
            if std_facet is not None:
                std_alpha = STD_ALPHA
                if line_name not in names_to_highlight:
                    std_alpha = 0.
                axis.fill_between(x_facet[::markevery[facet]],
                                  y_facet[line_name][::markevery[facet]] - std_facet[line_name][::markevery[facet]],
                                  y_facet[line_name][::markevery[facet]] + std_facet[line_name][::markevery[facet]],
                                  color=color_line,
                                  alpha=std_alpha,
                                  linewidth=0.)

    if fill_between_props is not None:
        for fill_between_prop in fill_between_props:
            facet_1 = fill_between_prop['facet_1']
            facet_2 = fill_between_prop['facet_2']
            line_name_1 = fill_between_prop['line_name_1']
            line_name_2 = fill_between_prop['line_name_2']
            y_data_line_1 = y_data[facet_1][line_name_1][::markevery[facet_1]]
            y_data_line_2 = y_data[facet_2][line_name_2][::markevery[facet_2]]
            x_data_line = x_data[facet_1][::markevery[facet_1]]

            color = None if 'color' not in fill_between_prop.keys() else fill_between_prop['color']
            alpha = None if 'alpha' not in fill_between_prop.keys() else fill_between_prop['alpha']
            hatch = None if 'hatch' not in fill_between_prop.keys() else fill_between_prop['hatch']

            axis.fill_between(x_data_line, y_data_line_1, y_data_line_2,
                              color=color,
                              alpha=alpha,
                              hatch=hatch,
                              linewidth=0.)

    if legend_style is not None:
        for line_name in names_to_highlight:
            axis.plot([], label=line_name, color=line_colors[line_name])
        if len(names_to_highlight) < len(line_names):
            axis.plot([], label='others', color=BASE_COLOR)
        axis.legend(edgecolor="0.8", fancybox=False, **legend_style)

    if secondary_legends is not None:
        for seconday_legend in secondary_legends:
            ax2 = axis.twinx()
            legend_lines = []
            legend_labels = []
            for legend_item in seconday_legend['legend_items']:
                legend_labels.append(legend_item[0])
                legend_line, = ax2.plot([], label=legend_item[0], color=legend_item[2], ls=legend_item[1])
                legend_lines.append(legend_line)
            ax2.get_yaxis().set_visible(False)
            ax2.legend(legend_lines, legend_labels, edgecolor="0.8", fancybox=False, **seconday_legend['legend_props'])


    # name the y-axis
    if y_label is not None:
        if tick_right:
            axis.set_ylabel(y_label, rotation=-90)
            axis.yaxis.set_label_position("right")
        else:
            axis.set_ylabel(y_label)

    # name the x-axis
    if x_label is not None:
        axis.set_xlabel(x_label)

    # name the axis
    if title is not None:
        axis.set_title(title)

    # specify left limit of the x-axis
    if x_min is not None:
        axis.set_xlim(left=x_min)

    # specify right limit of the x-axis
    if x_max is not None:
        axis.set_xlim(right=x_max)

    # specify bottom limit of the y-axis
    if y_min is not None:
        axis.set_ylim(bottom=y_min)

    # specify top limit of the y-axis
    if y_max is not None:
        axis.set_ylim(top=y_max)

    # specify ticks of the y-axis
    if y_ticks is not None:
        axis.set_yticks(y_ticks)
    if tick_right:
        axis.yaxis.tick_right()

    # specify ticks of the x-axis
    if x_ticks is not None:
        axis.set_xticks(x_ticks)
        if xticklabels is not None:
            axis.set_xticklabels(xticklabels)

    # add vertical lines
    if v_lines is not None:
        for v_line_props in v_lines:
            axis.axvline(**v_line_props)

    # add horizontal lines
    if h_lines is not None:
        for h_line_props in h_lines:
            axis.axhline(**h_line_props)

    axis.set_facecolor(BACKGROUND_COLOR)

    if text is not None:
        for text_to_add in text:
            axis.text(**text_to_add)

    return axis


########################################################################################################################
# Stacking (vertical and horizontal) of line plots
########################################################################################################################


def stacked_plots(
        axes_function: Union[callable, List[callable]],
        data: Dict[int, Dict[int, dict]],  # TODO think about indexing
        ncols: int = 1,
        nrows: int = 1,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        figsize=(10, 5),
        title: Optional[str] = None
) -> Tuple[plt.figure, List[plt.axis]]:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row',
                             figsize=figsize,
                             gridspec_kw={'height_ratios': height_ratios,
                                          'width_ratios': width_ratios})
    plt.subplots_adjust(hspace=0.000, wspace=0.000)
    for row_index in range(nrows):
        for col_index in range(ncols):
            plot_data = data[row_index][col_index]
            axes[row_index, col_index] = axes_function(axis=axes[row_index, col_index], **plot_data)

    if title is not None:
        fig.suptitle(title)
    else:
        return fig, axes
