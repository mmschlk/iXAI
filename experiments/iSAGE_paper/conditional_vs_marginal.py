import math
import time

import numpy as np
import pandas as pd
from river.datasets.base import REG

from experiments.setup.data import get_dataset, get_concept_drift_dataset
from experiments.setup.explainer import get_incremental_sage_explainer, \
    get_interval_sage_explainer, get_incremental_pfi_explainer, get_imputer_and_storage
from experiments.setup.loss import get_loss_function, get_training_metric
from experiments.setup.model import get_model
from increment_explain.explainer.sage import IncrementalSageExplainer
from increment_explain.utils.converters import RiverToPredictionFunction
from increment_explain.utils.trackers import ExponentialSmoothingTracker
from increment_explain.visualization import FeatureImportancePlotter

DEBUG = True

DATASET_1_NAME = 'elec2'

DATASET_RANDOM_SEED = 1
SHUFFLE_DATA = False
N_SAMPLES_1 = None

MODEL_NAME = 'ARF'

N_INNER_SAMPLES: int = 2
SMOOTHING_ALPHA: float = 0.001
RESERVOIR_LENGTH: int = 100

TRAINING_METRIC_WINDOW = 500

MODEL_PARAMS = {
    'ARF': {
        'n_models': 25,
    }
}


if __name__ == "__main__":

    # Get Data ---------------------------------------------------------------------------------------------------------
    dataset = get_dataset(
        dataset_name=DATASET_1_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES_1)
    feature_names = dataset.feature_names
    cat_feature_names = dataset.cat_feature_names
    num_feature_names = dataset.num_feature_names
    n_samples = dataset.n_samples
    stream = dataset.stream

    # Get Model / Loss and Training Procedure --------------------------------------------------------------------------
    task = dataset.task
    loss_function = get_loss_function(task=task)
    rolling_training_metric, training_metric = get_training_metric(task=task, rolling_window=TRAINING_METRIC_WINDOW)
    model, model_function = get_model(
        model_name=MODEL_NAME, task=task, feature_names=feature_names, **MODEL_PARAMS[MODEL_NAME])

    # Get imputer and explainers ---------------------------------------------------------------------------------------
    # conditional
    imputer_cond, storage_cond = get_imputer_and_storage(
        model_function=model_function,
        feature_removal_distribution="conditional",
        reservoir_kind=None,
        reservoir_length=None,
        cat_feature_names=cat_feature_names,
        num_feature_names=num_feature_names
    )
    incremental_sage_cond = IncrementalSageExplainer(
        model_function=model_function,
        loss_function=loss_function,
        imputer=imputer_cond,
        storage=storage_cond,
        feature_names=feature_names,
        smoothing_alpha=SMOOTHING_ALPHA,
        n_inner_samples=N_INNER_SAMPLES
    )

    # marginal
    imputer_marg, storage_marg = get_imputer_and_storage(
        model_function=model_function,
        feature_removal_distribution="marginal joint",
        reservoir_kind="geometric",
        reservoir_length=RESERVOIR_LENGTH,
        cat_feature_names=None,
        num_feature_names=None
    )
    incremental_sage_marg = IncrementalSageExplainer(
        model_function=model_function,
        loss_function=loss_function,
        imputer=imputer_marg,
        storage=storage_marg,
        feature_names=feature_names,
        smoothing_alpha=SMOOTHING_ALPHA,
        n_inner_samples=N_INNER_SAMPLES
    )

    # warm-up because of errors ------------------------------------------------------------------------------------
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        model.learn_one(x_i, y_i)
        if n > 10:
            break

    plotter = FeatureImportancePlotter(
        feature_names=feature_names
    )

    time_marg = 0.
    time_cond = 0.
    time_learning = 0.

    sage_cond_fi_values = []
    sage_marg_fi_values = []

    model_loss_tracker_cond = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
    marginal_loss_tracker_cond = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
    model_loss_cond = []
    marginal_loss_cond = []

    model_loss_tracker_marg = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
    marginal_loss_tracker_marg = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
    model_loss_marg = []
    marginal_loss_marg = []

    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        # predicting
        y_i_pred = model_function(x_i)
        loss_i = loss_function(y_true=y_i, y_prediction=y_i_pred)
        model_loss_tracker_cond.update(loss_i)
        model_loss_cond.append({"loss": model_loss_tracker_cond.get()})
        loss_marginal_i = loss_function(y_true=y_i, y_prediction=incremental_sage_cond._marginal_prediction_tracker.get())
        marginal_loss_tracker_cond.update(loss_marginal_i)
        marginal_loss_cond.append({"loss": marginal_loss_tracker_cond.get()})

        model_loss_tracker_marg.update(loss_i)
        model_loss_marg.append({"loss": model_loss_tracker_marg.get()})
        loss_marginal_i = loss_function(y_true=y_i, y_prediction=incremental_sage_cond._marginal_prediction_tracker.get())
        marginal_loss_tracker_marg.update(loss_marginal_i)
        marginal_loss_marg.append({"loss": marginal_loss_tracker_marg.get()})

        # sage inc cond
        start_time = time.time()
        fi_values = incremental_sage_cond.explain_one(x_i, y_i)
        time_cond += time.time() - start_time
        sage_cond_fi_values.append(fi_values)
        plotter.update(fi_values, facet_name='cond')

        # sage inc marg
        start_time = time.time()
        fi_values = incremental_sage_marg.explain_one(x_i, y_i)
        time_marg += time.time() - start_time
        sage_marg_fi_values.append(fi_values)
        plotter.update(fi_values, facet_name='marg')

        # learning
        start_time = time.time()
        model.learn_one(x_i, y_i)
        time_learning += time.time() - start_time

        if DEBUG and n % 1000 == 0:
            print(f"{n}: x_i                 {x_i}\n"
                  f"{n}: conditional\n"
                  f"{n}: marginal-prediction {incremental_sage_cond._marginal_prediction_tracker.get()}\n"
                  f"{n}: model-loss          {model_loss_tracker_cond.get()}\n"
                  f"{n}: marginal-loss       {marginal_loss_tracker_cond.get()}\n"
                  f"{n}: diff                {marginal_loss_tracker_cond.get() - model_loss_tracker_cond.get()}\n"
                  f"{n}: sum-sage            {sum(list(incremental_sage_cond.importance_values.values()))}\n"
                  f"{n}: inc-sage            {incremental_sage_cond.importance_values}\n"
                  f"{n}: marginal\n"
                  f"{n}: marginal-prediction {incremental_sage_marg._marginal_prediction_tracker.get()}\n"
                  f"{n}: model-loss          {model_loss_tracker_marg.get()}\n"
                  f"{n}: marginal-loss       {marginal_loss_tracker_marg.get()}\n"
                  f"{n}: diff                {marginal_loss_tracker_marg.get() - model_loss_tracker_marg.get()}\n"
                  f"{n}: sum-sage            {sum(list(incremental_sage_marg.importance_values.values()))}\n"
                  f"{n}: inc-sage            {incremental_sage_marg.importance_values}\n"
                  )

        if n >= n_samples:
            break

    # Store the results in a database ----------------------------------------------------------------------------------

    data_folder = "conditional_marginal"
    facet = "right.csv"

    name = "cond_fi_values"
    df = pd.DataFrame(sage_cond_fi_values)
    df.to_csv(f"plots/{data_folder}/{'_'.join((name, facet))}", index=False)

    name = "marg_fi_values"
    df = pd.DataFrame(sage_marg_fi_values)
    df.to_csv(f"plots/{data_folder}/{'_'.join((name, facet))}", index=False)

    name = "model_loss"
    df = pd.DataFrame(model_loss_cond)
    df.to_csv(f"plots/{data_folder}/{'_'.join((name, facet))}", index=False)

    name = "marginal_loss"
    df = pd.DataFrame(marginal_loss_cond)
    df.to_csv(f"plots/{data_folder}/{'_'.join((name, facet))}", index=False)

    performance_kw = {"y_min": 0, "y_max": 1, "y_label": "cross\nentropy", "color_list": ["red", "black"],
                      "line_names": ["loss"], "names_to_highlight": ['loss'],
                      "line_styles": {"model_loss": "solid", "marginal_loss": "dashed"},
                      "markevery": {"model_loss": 100, "marginal_loss": 100},
                      "legend": {"legend_props": {"loc": 7, "ncol": 1, "fontsize": "small"},
                                 "legend_items": [("marginal loss", "dashed", "red"),
                                                  ("model loss", "solid", "red")]}}

    plotter.plot(
        figsize=(5, 10),
        model_performances={"model_loss": model_loss_cond, "marginal_loss": marginal_loss_cond},
        performance_kw=performance_kw,
        title=f'Agrawal Stream with Concept Drift',
        y_label='SAGE values',
        x_label='Samples',
        names_to_highlight=['salary', 'commission', 'age', 'elevel'],
        h_lines=[{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
        line_styles={'cond': '-', 'marg': '--'},
        legend_style={"fontsize": "small"},
        markevery={'cond': 100, 'marg': 100},
        y_min=-0.049,
        y_max=0.325,
        save_name="result_data/conditional_vs_marginal.png"
    )
