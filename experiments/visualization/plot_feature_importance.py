import copy
import os
from fractions import Fraction
import matplotlib.pyplot as plt
import pandas as pd

from increment_explain.visualization.config import SAVE_DPI
from increment_explain.visualization.line_plots import plot_multi_line_graph


def plot_stacked_line_plots(top_data, bot_data, top_styles, bot_styles,
                            figsize=(10, 5), top_portion=0.5, title=None, show=False,
                            save_path=None, top_margin=0.88):
    ratio = Fraction(top_portion).limit_denominator()
    top_ratio, bot_ratio = (ratio.numerator, ratio.denominator)
    top_styles = copy.deepcopy(top_styles)
    bot_styles = copy.deepcopy(bot_styles)
    fig, (top_axis, bot_axis) = plt.subplots(nrows=2, ncols=1, sharex='all',
                                             figsize=figsize,
                                             gridspec_kw={'height_ratios': [top_ratio, bot_ratio]})
    plt.subplots_adjust(hspace=0.000)
    bot_axis = plot_multi_line_graph(axis=bot_axis,
                                     y_data=bot_data['y_data'],
                                     #x_data=bot_data['x_data'],
                                     line_names=bot_data['line_names'],
                                     **bot_styles)
    top_axis = plot_multi_line_graph(axis=top_axis,
                                     y_data=top_data['y_data'],
                                     #x_data=top_data['x_data'],
                                     line_names=top_data['line_names'],
                                     **top_styles)
    plt.tight_layout()
    if title is not None:
        fig.suptitle(title)
        plt.subplots_adjust(hspace=0.000, wspace=0.000, top=top_margin)
    else:
        plt.subplots_adjust(hspace=0.000, wspace=0.000)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
        return None
    else:
        return fig, (top_axis, bot_axis)


def plot_double_stacked_line_plots(top_left_data, top_right_data, bot_left_data, bot_right_data,
                                   top_left_styles, top_right_styles, bot_left_styles, bot_right_styles,
                                   figsize=(10, 5), top_portion=1., left_proportion=1., title=None, show=False,
                                   different_scales=False,
                                   save_path=None, top_margin=0.88, wspace=0.0):
    ratio_vertical = Fraction(top_portion).limit_denominator()
    top_ratio, bot_ratio = (ratio_vertical.numerator, ratio_vertical.denominator)
    ratio_horizontal = Fraction(left_proportion).limit_denominator()
    left_ratio, right_ratio = (ratio_horizontal.numerator, ratio_horizontal.denominator)
    sharey = 'row' if not different_scales else 'none'
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='col', sharey=sharey,
                             dpi=SAVE_DPI,
                             figsize=figsize,
                             gridspec_kw={'height_ratios': [top_ratio, bot_ratio],
                                          'width_ratios': [left_ratio, right_ratio]})

    (top_left_axis, top_right_axis), (bot_left_axis, bot_right_axis) = axes
    top_left_axis = plot_multi_line_graph(axis=top_left_axis,
                                          y_data=top_left_data['y_data'],
                                          #x_data=top_left_data['x_data'],
                                          line_names=top_left_data['line_names'],
                                          **top_left_styles)
    top_right_axis = plot_multi_line_graph(axis=top_right_axis,
                                           y_data=top_right_data['y_data'],
                                           #x_data=top_right_data['x_data'],
                                           line_names=top_right_data['line_names'],
                                           **top_right_styles)
    bot_left_axis = plot_multi_line_graph(axis=bot_left_axis,
                                          y_data=bot_left_data['y_data'],
                                          #x_data=bot_left_data['x_data'],
                                          line_names=bot_left_data['line_names'],
                                          **bot_left_styles)
    bot_right_axis = plot_multi_line_graph(axis=bot_right_axis,
                                           y_data=bot_right_data['y_data'],
                                           #x_data=bot_right_data['x_data'],
                                           line_names=bot_right_data['line_names'],
                                           **bot_right_styles)
    plt.tight_layout()

    if title is not None:
        fig.suptitle(title)
        plt.subplots_adjust(hspace=0., wspace=wspace, top=top_margin)
    else:
        plt.subplots_adjust(hspace=0., wspace=wspace)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
        return None
    else:
        return fig, (top_left_axis, top_right_axis, bot_left_axis, bot_right_axis)


if __name__ == "__main__":
    pass
