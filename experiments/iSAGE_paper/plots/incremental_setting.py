import copy
import os
import pandas as pd

from experiments.visualization.plot_feature_importance import plot_double_stacked_line_plots, plot_stacked_line_plots


if __name__ == "__main__":
    DATA_DIR = "incremental_setting_5"

    SAVE_PATH = os.path.join(DATA_DIR, "incremental_setting.png")
    SAVE_PATH_LEFT = os.path.join(DATA_DIR, "incremental_setting_left.png")
    SAVE_PATH_RIGHT = os.path.join(DATA_DIR, "incremental_setting_right.png")

    # load data --------------------------------------------------------------------------------------------------------
    # left data
    marginal_loss_left_df = pd.read_csv(os.path.join(DATA_DIR, "marginal_loss_left.csv"))
    model_loss_left_df = pd.read_csv(os.path.join(DATA_DIR, "model_loss_left.csv"))
    inc_fi_values_left_df = pd.read_csv(os.path.join(DATA_DIR, "inc_fi_values_left.csv"))
    int_fi_values_left_df = pd.read_csv(os.path.join(DATA_DIR, "int_fi_values_left.csv"))
    pfi_fi_values_left_df = pd.read_csv(os.path.join(DATA_DIR, "pfi_fi_values_left.csv"))
    # right data
    marginal_loss_right_df = pd.read_csv(os.path.join(DATA_DIR, "marginal_loss_right.csv"))
    model_loss_right_df = pd.read_csv(os.path.join(DATA_DIR, "model_loss_right.csv"))
    inc_fi_values_right_df = pd.read_csv(os.path.join(DATA_DIR, "inc_fi_values_right.csv"))
    int_fi_values_right_df = pd.read_csv(os.path.join(DATA_DIR, "int_fi_values_right.csv"))
    inc_welford_fi_values_right_df = pd.read_csv(os.path.join(DATA_DIR, "inc_welford_fi_values_right.csv"))

    # plot data --------------------------------------------------------------------------------------------------------
    top_left_data = {'y_data': {"model_loss": model_loss_left_df, "marginal_loss": marginal_loss_left_df},
                     'line_names': ['loss']}
    top_right_data = {'y_data': {"model_loss": model_loss_right_df, "marginal_loss": marginal_loss_right_df},
                      'line_names': ['loss']}
    bot_left_data = {'y_data': {'inc': inc_fi_values_left_df,
                                'int': int_fi_values_left_df,
                                },
                     'line_names': list(inc_fi_values_left_df.columns)}
    bot_right_data = {'y_data': {'inc': inc_fi_values_right_df,
                                'bat': inc_welford_fi_values_right_df
                                },
                      'line_names': list(inc_fi_values_right_df.columns)}

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
    top_left_styles["title"] = f"iSAGE vs. interval SAGE (agrawal function 2 to 3)"
    top_left_styles["v_lines"] = [{'x': 20_000, 'ls': 'dotted', 'c': 'grey', 'linewidth': 1}]
    top_right_styles["v_lines"] = [{'x': 20_000, 'ls': 'dotted', 'c': 'grey', 'linewidth': 1}]
    top_right_styles["title"] = f"iSAGE vs. SAGE (agrawal function 3 to 2)"

    # bottom styles / feature importance row
    bot_styles = {
        "y_min": -0.03, "y_max": 0.24, "x_label": "Samples",
        "names_to_highlight": ['salary', 'commission', 'age', 'elevel'],
        "h_lines": [{'y': 0., 'ls': 'dotted', 'c': 'grey', 'linewidth': 1}],
        "line_styles": {'inc': 'solid', 'int': 'dashed', 'bat': 'dashdot', 'pfi': 'dotted'},
        "markevery": {'inc': 100, 'int': 1, 'bat': 100, 'pfi': 100},
        "x_ticks": [0, 10_000, 20_000, 30_000, 40_000],
        "y_ticks": [0, 0.1, 0.2],
    }
    bot_left_styles = copy.deepcopy(bot_styles)
    bot_left_styles["y_label"] = "SAGE values"
    bot_left_styles["v_lines"] = [{'x': 20_000, 'ls': 'dotted', 'c': 'grey', 'linewidth': 1}]
    bot_right_styles = copy.deepcopy(bot_styles)
    bot_right_styles["v_lines"] = [{'x': 20_000, 'ls': 'dotted', 'c': 'grey', 'linewidth': 1}]




    # legends
    top_right_styles["secondary_legends"] = [{
        "legend_props": {"loc": 'upper left', "ncol": 1, "fontsize": "small", 'title': 'loss', "borderaxespad": 0,
                         "bbox_to_anchor": (1.02, 1), "frameon": False},
        "legend_items": [("mean prediction", "dashed", "red"),
                         ("model", "solid", "red")]
    }]
    bot_right_styles["legend_style"] = {
        "fontsize": "small", 'title': 'features', "borderaxespad": 0, "frameon": False,
        "loc": 'upper left', "bbox_to_anchor": (1.02, 1)}
    bot_right_styles["secondary_legends"] = [{
        "legend_props": {"loc": 'upper left', "ncol": 1, "fontsize": "small", "title": "explainer", "borderaxespad": 0,
                         "bbox_to_anchor": (1.02, 0.6), "frameon": False},
        "legend_items": [("iSAGE", "solid", "black"),
                         ("interval SAGE", "dashed", "black"),
                         #("iPFI", "dotted", 'black'),
                         ("SAGE", "dashdot", "black")]
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
        wspace=0.03
        #title=r"iSAGE on agrawal streams ($\alpha: 0.001$)"
    )

    '''
    plot_stacked_line_plots(
        figsize=(12.5, 5),
        top_data=top_left_data,
        top_styles=top_left_styles,
        bot_data=bot_left_data,
        bot_styles=bot_left_styles,
        show=True,
        save_path=SAVE_PATH_LEFT
    )

    plot_stacked_line_plots(
        figsize=(12.5, 5),
        top_data=top_right_data,
        top_styles=top_right_styles,
        bot_data=bot_right_data,
        bot_styles=bot_right_styles,
        show=True,
        save_path=SAVE_PATH_RIGHT
    )
    '''
