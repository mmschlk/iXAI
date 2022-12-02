import copy
import os
import pandas as pd

from experiments.visualization.plot_feature_importance import plot_double_stacked_line_plots, plot_stacked_line_plots

if __name__ == "__main__":
    SAVE_PATH = "conditional_marginal/conditional_marginal.png"

    # load data --------------------------------------------------------------------------------------------------------
    DATA_DIR = "conditional_marginal"
    # left data
    marginal_loss_left_df = pd.read_csv(os.path.join(DATA_DIR, "marginal_loss_left.csv"))
    model_loss_left_df = pd.read_csv(os.path.join(DATA_DIR, "model_loss_left.csv"))
    cond_fi_values_left_df = pd.read_csv(os.path.join(DATA_DIR, "cond_fi_values_left.csv"))
    marg_fi_values_left_df = pd.read_csv(os.path.join(DATA_DIR, "marg_fi_values_left.csv"))
    # right data

    marginal_loss_right_df = pd.read_csv(os.path.join(DATA_DIR, "marginal_loss_right.csv"))
    model_loss_right_df = pd.read_csv(os.path.join(DATA_DIR, "model_loss_right.csv"))
    cond_fi_values_right_df = pd.read_csv(os.path.join(DATA_DIR, "cond_fi_values_right.csv"))
    marg_fi_values_right_df = pd.read_csv(os.path.join(DATA_DIR, "marg_fi_values_right.csv"))

    # plot data --------------------------------------------------------------------------------------------------------
    top_left_data = {'y_data': {"model_loss": model_loss_left_df, "marginal_loss": marginal_loss_left_df},
                     'line_names': ['loss']}
    top_right_data = {'y_data': {"model_loss": model_loss_right_df, "marginal_loss": marginal_loss_right_df},
                      'line_names': ['loss']}
    bot_left_data = {'y_data': {'cond': cond_fi_values_left_df,
                                'marg': marg_fi_values_left_df},
                     'line_names': list(cond_fi_values_left_df.columns)}
    bot_right_data = {'y_data': {'cond': cond_fi_values_right_df,
                                 'marg': marg_fi_values_right_df},
                      'line_names': list(cond_fi_values_right_df.columns)}

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
    top_right_styles = copy.deepcopy(top_styles)
    top_left_styles = copy.deepcopy(top_styles)
    top_left_styles["y_label"] = "loss"
    top_left_styles["title"] = f"agrawal stream"
    top_right_styles["title"] = f"elec2 stream"
    #top_right_styles['y_ticks'] = []

    # bottom styles / feature importance row
    bot_styles = {
        "x_label": "Samples",
        "h_lines": [{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
        "line_styles": {'cond': 'solid', 'marg': 'dotted'},
        "markevery": {'cond': 30, 'marg': 30, },
    }
    bot_left_styles = copy.deepcopy(bot_styles)
    bot_left_styles["y_label"] = "SAGE values"
    bot_left_styles["legend_style"] = {"fontsize": "small", 'title': 'features'}
    bot_left_styles["names_to_highlight"] = ['salary', 'commission', 'age']
    bot_left_styles["y_min"] = -0.04
    bot_left_styles["y_max"] = 0.19
    bot_left_styles["y_ticks"] = [0., 0.05, 0.1, 0.15]
    bot_right_styles = copy.deepcopy(bot_styles)
    bot_right_styles["y_min"] = -0.09
    bot_right_styles["y_max"] = 0.598
    bot_right_styles["y_ticks"] = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
    bot_right_styles["legend_style"] = {"fontsize": "small", 'title': 'features'}
    bot_right_styles["names_to_highlight"] = ['nswprice', 'vicprice']

    # legends
    top_right_styles["secondary_legends"] = [{
        "legend_props": {"loc": 'upper left', "ncol": 1, "fontsize": "small", 'title': 'loss', "borderaxespad": 0,
                         "bbox_to_anchor": (1.02, 1), "frameon": False},
        "legend_items": [("mean prediction", "dashed", "red"),
                         ("model", "solid", "red")]
    }]
    bot_right_styles["secondary_legends"] = [
        {
        "legend_props": {"loc": 'upper left', "ncol": 1, "fontsize": "small", "title": "distribution", "borderaxespad": 0,
                         "bbox_to_anchor": (1.02, 1), "frameon": False},
        "legend_items": [("marg.", "solid", "black"),
                         ("cond.", "dotted", "black")]
    }]

    plot_double_stacked_line_plots(
        figsize=(12.5, 5),
        top_left_data=top_left_data,
        top_left_styles=top_left_styles,
        bot_left_data=bot_left_data,
        bot_left_styles=bot_left_styles,
        top_right_data=top_right_data,
        top_right_styles=top_right_styles,
        bot_right_data=bot_right_data,
        bot_right_styles=bot_right_styles,
        top_portion=0.25,
        show=True,
        save_path=SAVE_PATH,
        different_scales=True,
        wspace=0.08
    )