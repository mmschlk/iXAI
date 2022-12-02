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
from increment_explain.explainer import IncrementalPFI
from increment_explain.explainer.sage import IntervalSageExplainer, IncrementalSageExplainer
from increment_explain.utils.converters import RiverToPredictionFunction
from increment_explain.utils.trackers import ExponentialSmoothingTracker
from increment_explain.visualization import FeatureImportancePlotter

DEBUG = True

DATASET_1_NAME = 'agrawal 2'
DATASET_2_NAME = 'agrawal 1'
DATASET_RANDOM_SEED = 1
SHUFFLE_DATA = False
N_SAMPLES_1 = 20000
N_SAMPLES_2 = 20000

CONCEPT_DRIFT_POSITION = 20000
CONCEPT_DRIFT_WIDTH = 1
CONCEPT_DRIFT_SWITCHING_FEATURES = None # {'salary': 'age'}

MODEL_NAME = 'ARF'

N_INNER_SAMPLES: int = 5
N_INTERVAL_LENGTH: int = 5000
CONFIDENCE_BOUNDS_DELTA: float = 0.1
SMOOTHING_ALPHA: float = 0.001
FEATURE_REMOVAL_DISTRIBUTION: str = 'marginal joint'  # ['marginal joint', 'marginal product', 'conditional']
RESERVOIR_LENGTH: int = 100
RESERVOIR_KIND: str = 'geometric'  # ['geometric', 'uniform']

TRAINING_METRIC_WINDOW = 500

MODEL_PARAMS = {
    'ARF': {
        'n_models': 25,
    }
}


if __name__ == "__main__":

    # Get Data ---------------------------------------------------------------------------------------------------------
    dataset_1 = get_dataset(
        dataset_name=DATASET_1_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES_1)
    dataset_2 = get_dataset(
        dataset_name=DATASET_2_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES_2)
    dataset = get_concept_drift_dataset(
        dataset_1_name=DATASET_1_NAME, dataset_2_name=DATASET_2_NAME,
        dataset_1=dataset_1, dataset_2=dataset_2,
        position=CONCEPT_DRIFT_POSITION, width=CONCEPT_DRIFT_WIDTH,
        features_to_switch=CONCEPT_DRIFT_SWITCHING_FEATURES,
    )
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
    imputer, storage = get_imputer_and_storage(
        model_function=model_function,
        feature_removal_distribution=FEATURE_REMOVAL_DISTRIBUTION,
        reservoir_kind=RESERVOIR_KIND,
        reservoir_length=RESERVOIR_LENGTH,
        cat_feature_names=cat_feature_names,
        num_feature_names=num_feature_names
    )

    incremental_sage = IncrementalSageExplainer(
        model_function=model_function,
        loss_function=loss_function,
        imputer=imputer,
        storage=storage,
        feature_names=feature_names,
        smoothing_alpha=SMOOTHING_ALPHA,
        n_inner_samples=N_INNER_SAMPLES
    )

    incremental_welford_sage = IncrementalSageExplainer(
        model_function=model_function,
        loss_function=loss_function,
        dynamic_setting=False,
        feature_names=feature_names,
        smoothing_alpha=SMOOTHING_ALPHA,
        n_inner_samples=N_INNER_SAMPLES
    )

    interval_explainer = IntervalSageExplainer(
        model_function=model_function,
        loss_function=loss_function,
        feature_names=feature_names,
        n_inner_samples=N_INNER_SAMPLES,
        interval_length=N_INTERVAL_LENGTH
    )

    # warm-up because of errors ----------------------------------------------------------------------------------------
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        model.learn_one(x_i, y_i)
        if n > 10:
            break

    plotter = FeatureImportancePlotter(
        feature_names=feature_names
    )

    time_sage = 0.
    time_interval = 0.
    time_learning = 0.

    sage_fi_values = []
    sage_welford_fi_values = []
    interval_fi_values = []

    model_loss_tracker = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
    marginal_loss_tracker = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
    model_loss = []
    marginal_loss = []

    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        # predicting
        y_i_pred = model_function(x_i)
        loss_i = loss_function(y_true=y_i, y_prediction=y_i_pred)
        model_loss_tracker.update(loss_i)
        model_loss.append({"loss": model_loss_tracker.get()})
        loss_marginal_i = loss_function(y_true=y_i, y_prediction=incremental_sage._marginal_prediction_tracker.get())
        marginal_loss_tracker.update(loss_marginal_i)
        marginal_loss.append({"loss": marginal_loss_tracker.get()})

        # sage inc
        start_time = time.time()
        fi_values = incremental_welford_sage.explain_one(x_i, y_i)
        time_sage += time.time() - start_time
        sage_welford_fi_values.append(fi_values)
        plotter.update(fi_values, facet_name='inc-w')

        # sage inc
        start_time = time.time()
        fi_values = incremental_sage.explain_one(x_i, y_i)
        time_sage += time.time() - start_time
        sage_fi_values.append(fi_values)
        plotter.update(fi_values, facet_name='inc')

        # sage int
        start_time = time.time()
        fi_values = interval_explainer.explain_one(x_i, y_i)
        time_interval += time.time() - start_time
        interval_fi_values.append(fi_values)
        plotter.update(fi_values, facet_name='int')

        # learning
        start_time = time.time()
        model.learn_one(x_i, y_i)
        time_learning += time.time() - start_time

        if DEBUG and n % 1000 == 0:
            print(f"{n}: x_i                 {x_i}\n"
                  f"{n}: marginal-prediction {incremental_sage._marginal_prediction_tracker.get()}\n"
                  f"{n}: model-loss          {model_loss_tracker.get()}\n"
                  f"{n}: marginal-loss       {marginal_loss_tracker.get()}\n"
                  f"{n}: diff                {marginal_loss_tracker.get() - model_loss_tracker.get()}\n"
                  f"{n}: sum-sage            {sum(list(incremental_sage.importance_values.values()))}\n"
                  f"{n}: inc-sage            {incremental_sage.importance_values}\n"
                  f"{n}: inc-sage-bnd        {incremental_sage.get_confidence_bound(delta=CONFIDENCE_BOUNDS_DELTA)}\n"
                  f"{n}: inc-sage-var        {incremental_sage.variances}\n\n"
                  f"{n}: int-sage            {interval_explainer.importance_values}\n"
                  f"{n}: inc-sage-welford    {incremental_welford_sage.importance_values}\n"
                  )

        if n >= n_samples:
            break

    # Store the results in a database ----------------------------------------------------------------------------------

    facet = "right.csv"
    data_folder = "incremental_setting_5"

    name = "inc_fi_values"
    df = pd.DataFrame(sage_fi_values)
    df.to_csv(f"plots/{data_folder}/{'_'.join((name, facet))}", index=False)

    name = "inc_welford_fi_values"
    df = pd.DataFrame(sage_welford_fi_values)
    df.to_csv(f"plots/{data_folder}/{'_'.join((name, facet))}", index=False)

    name = "int_fi_values"
    df = pd.DataFrame(interval_fi_values)
    df.to_csv(f"plots/{data_folder}/{'_'.join((name, facet))}", index=False)

    name = "model_loss"
    df = pd.DataFrame(model_loss)
    df.to_csv(f"plots/{data_folder}/{'_'.join((name, facet))}", index=False)

    name = "marginal_loss"
    df = pd.DataFrame(marginal_loss)
    df.to_csv(f"plots/{data_folder}/{'_'.join((name, facet))}", index=False)

    performance_kw = {"y_min": 0, "y_max": 1, "y_label": "loss", "color_list": ["red", "black"],
                      "line_names": ["loss"], "names_to_highlight": ['loss'],
                      "line_styles": {"model_loss": "solid", "marginal_loss": "dashed"},
                      "markevery": {"model_loss": 100, "marginal_loss": 100},
                      "legend": {"legend_props": {"loc": 7, "ncol": 1, "fontsize": "small"},
                                 "legend_items": [("marginal loss", "dashed", "red"),
                                                  ("model loss", "solid", "red")]}}

    plotter.plot(
        figsize=(5, 10),
        model_performances={"model_loss": model_loss, "marginal_loss": marginal_loss},
        performance_kw=performance_kw,
        title=f'Agrawal Stream with Concept Drift',
        y_label='SAGE values',
        x_label='Samples',
        names_to_highlight=['salary', 'commission', 'age', 'elevel'],
        v_lines=[{'x': CONCEPT_DRIFT_POSITION, 'ls': '--', 'c': 'black', 'linewidth': 1}],
        h_lines=[{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
        line_styles={'inc': '-', 'int': '--', 'inc-w': 'dotted'},
        legend_style={"fontsize": "small"},
        markevery={'inc': 10, 'int': 1, 'inc-w': 10},
        y_min=-0.05,
        y_max=0.42,
    )
