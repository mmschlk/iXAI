import copy
import os
import pandas as pd

from experiments.visualization.plot_feature_importance import plot_double_stacked_line_plots, plot_stacked_line_plots


if __name__ == "__main__":
    DATA_DIR = "reoccurring_drift"

    SAVE_PATH_LEFT = os.path.join(DATA_DIR, "reoccurring_drift.png")

    # load data --------------------------------------------------------------------------------------------------------
    # left data
    marginal_loss_left_df = pd.read_csv(os.path.join(DATA_DIR, "marginal_loss_left.csv"))
    model_loss_left_df = pd.read_csv(os.path.join(DATA_DIR, "model_loss_left.csv"))
    inc_fi_values_left_df = pd.read_csv(os.path.join(DATA_DIR, "inc_fi_values_left.csv"))
    int_fi_values_left_df = pd.read_csv(os.path.join(DATA_DIR, "int_fi_values_left.csv"))
    inc_welford_fi_values_left_df = pd.read_csv(os.path.join(DATA_DIR, "inc_welford_fi_values_left.csv"))


    # plot data --------------------------------------------------------------------------------------------------------
    top_left_data = {'y_data': {"model_loss": model_loss_left_df, "marginal_loss": marginal_loss_left_df},
                     'line_names': ['loss']}
    bot_left_data = {'y_data': {'inc': inc_fi_values_left_df,
                                'int': int_fi_values_left_df,
                                'bat': inc_welford_fi_values_left_df
                                },
                     'line_names': list(inc_fi_values_left_df.columns)}

    # plot styles ------------------------------------------------------------------------------------------------------

    # top styles / performance row
    top_styles = {
        "y_min": 0, "y_max": 1,
        "color_list": ["red", "black"], "markevery": {"model_loss": 100, "marginal_loss": 100},
        "line_styles": {"model_loss": "solid", "marginal_loss": "dashed"},
        "fill_between_props": [{
            'facet_1': 'model_loss', 'facet_2': 'marginal_loss',
            'line_name_1': 'loss', 'line_name_2': 'loss',
            'hatch': '///',
            'color': 'red',
            'alpha': 0.1,
        }]
    }
    top_left_styles = copy.deepcopy(top_styles)
    top_left_styles["y_label"] = "loss"
    top_left_styles["title"] = f"iSAGE vs. interval SAGE (agrawal function 2 to 3)"
    top_left_styles["v_lines"] = [{'x': 25_000, 'ls': 'dotted', 'c': 'grey', 'linewidth': 1},
                                  {'x': 50_000, 'ls': 'dotted', 'c': 'grey', 'linewidth': 1},
                                  {'x': 75_000, 'ls': 'dotted', 'c': 'grey', 'linewidth': 1}]

    # bottom styles / feature importance row
    bot_styles = {
        "y_min": -0.03, "y_max": 0.31, "x_label": "Samples",
        "names_to_highlight": ['salary', 'commission', 'age', 'elevel'],
        "h_lines": [{'y': 0., 'ls': 'dotted', 'c': 'grey', 'linewidth': 1}],
        "line_styles": {'inc': 'solid', 'int': 'dashed', 'bat': 'dashdot', 'pfi': 'dotted'},
        "markevery": {'inc': 100, 'int': 1, 'bat': 100, 'pfi': 100},
        "x_ticks": [0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000],
        "y_ticks": [0, 0.1, 0.2],
    }
    bot_left_styles = copy.deepcopy(bot_styles)
    bot_left_styles["y_label"] = "SAGE values"
    bot_left_styles["v_lines"] = [{'x': 25_000, 'ls': 'dotted', 'c': 'grey', 'linewidth': 1},
                                  {'x': 50_000, 'ls': 'dotted', 'c': 'grey', 'linewidth': 1},
                                  {'x': 75_000, 'ls': 'dotted', 'c': 'grey', 'linewidth': 1}]


    # legends
    top_left_styles["secondary_legends"] = [{
        "legend_props": {"loc": 'center right', "ncol": 1, "fontsize": "small", 'title': 'loss'},
        "legend_items": [("mean prediction", "dashed", "red"),
                         ("model", "solid", "red")]
    }]
    bot_left_styles["legend_style"] = {
        "fontsize": "small", 'title': 'features', "ncol": 2,
        "loc": 'upper left', "bbox_to_anchor": (0, 1)}
    bot_left_styles["secondary_legends"] = [{
        "legend_props": {"loc": 'upper left', "ncol": 1, "fontsize": "small", "title": "explainer",
                         "bbox_to_anchor": (0.18, 1)},
        "legend_items": [("iSAGE", "solid", "black"),
                         ("interval SAGE", "dashed", "black"),
                         #("iPFI", "dotted", 'black'),
                         ("SAGE", "dashdot", "black")]
    }]


    plot_stacked_line_plots(
        figsize=(12.5, 5),
        top_data=top_left_data,
        top_styles=top_left_styles,
        bot_data=bot_left_data,
        bot_styles=bot_left_styles,
        show=True,
        save_path=SAVE_PATH_LEFT,
        top_portion=0.25
    )
