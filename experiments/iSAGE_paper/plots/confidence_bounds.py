import copy
import os
import pandas as pd

from experiments.visualization.plot_feature_importance import plot_double_stacked_line_plots


if __name__ == "__main__":
    DATA_DIR = "confidence_bounds"

    SAVE_PATH = os.path.join(DATA_DIR, "confidence_bounds.png")

    # load data --------------------------------------------------------------------------------------------------------
    marginal_loss_df = pd.read_csv(os.path.join(DATA_DIR, "marginal_loss_.csv"))
    model_loss_df = pd.read_csv(os.path.join(DATA_DIR, "model_loss_.csv"))
    sage_fi_values_df = pd.read_csv(os.path.join(DATA_DIR, "inc_fi_values_.csv"))
    confidence_bounds_1_df = pd.read_csv(os.path.join(DATA_DIR, "confidence_bounds_1_.csv"))
    confidence_bounds_2_df = pd.read_csv(os.path.join(DATA_DIR, "confidence_bounds_2_.csv"))

    # plot data --------------------------------------------------------------------------------------------------------
    top_left_data = {'y_data': {"model_loss": model_loss_df, "marginal_loss": marginal_loss_df},
                     'line_names': ['loss']}
    top_right_data = top_left_data
    bot_left_data = {'y_data': {'inc': sage_fi_values_df},
                     'line_names': list(sage_fi_values_df.columns)}
    bot_right_data = bot_left_data

    # plot styles ------------------------------------------------------------------------------------------------------

    # top styles / performance row
    top_styles = {
        "y_min": 0, "y_max": 1,
        "color_list": ["red", "black"], "markevery": {"model_loss": 100, "marginal_loss": 100},
        "line_styles": {"model_loss": "solid", "marginal_loss": "dashed"},
        "fill_between_props": [{'facet_1': 'model_loss', 'facet_2': 'marginal_loss','line_name_1': 'loss', 'line_name_2': 'loss','hatch': '///','color': 'red','alpha': 0.1}]
    }
    top_right_styles = copy.copy(top_styles)
    top_left_styles = copy.copy(top_styles)

    top_left_styles["y_label"] = "loss"
    top_left_styles["title"] = r"$\delta =$ " + f"{0.1}"
    top_right_styles["title"] = r"$\delta =$ " + f"{0.05}"

    # bottom styles / feature importance row
    bot_styles = {
        "y_min": -0.15, "y_max": 0.79, "x_label": "Samples",
        "names_to_highlight": ['nswprice', 'vicprice', 'nswdemand'],
        "h_lines": [{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
    }
    bot_left_styles = copy.copy(bot_styles)
    bot_left_styles["std"] = {'inc': confidence_bounds_1_df}
    bot_left_styles["y_label"] = "SAGE values"
    bot_right_styles = copy.copy(bot_styles)
    bot_right_styles["std"] = {'inc': confidence_bounds_2_df}

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
        top_portion=0.25, title="Confidence Bounds for elec2 data stream",
        show=True,
        save_path=SAVE_PATH
    )
