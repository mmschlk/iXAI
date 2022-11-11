import math
import time

import numpy as np
import pandas as pd
from river.datasets.base import REG
from river.drift import ADWIN

from experiments.setup.data import get_dataset, get_concept_drift_dataset
from experiments.setup.explainer import get_incremental_sage_explainer, \
    get_interval_sage_explainer, get_incremental_pfi_explainer
from experiments.setup.loss import get_loss_function, get_training_metric
from experiments.setup.model import get_model
from increment_explain.utils.converters import RiverToPredictionFunction
from increment_explain.utils.trackers import ExponentialSmoothingTracker
from increment_explain.visualization import FeatureImportancePlotter

DEBUG = True

DATASET_1_NAME = 'elec2'

DATASET_RANDOM_SEED = 1
SHUFFLE_DATA = False
N_SAMPLES_1 = None

CONCEPT_DRIFT_SWITCHING_FEATURES = None  # {'salary': 'age'}

MODEL_NAME = 'ARF'

N_INNER_SAMPLES = 1

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
    n_samples = dataset.n_samples
    stream = dataset.stream

    # Get Model / Loss and Training Procedure --------------------------------------------------------------------------
    task = dataset.task
    loss_function = get_loss_function(task=task)
    rolling_training_metric, training_metric = get_training_metric(task=task, rolling_window=TRAINING_METRIC_WINDOW)
    model, model_function = get_model(
        model_name=MODEL_NAME, task=task, feature_names=feature_names, **MODEL_PARAMS[MODEL_NAME])

    # Get explainers ---------------------------------------------------------------------------------------------------
    incremental_sage = get_incremental_sage_explainer(
        feature_names=feature_names,
        model_function=model_function,
        loss_function=loss_function,
        n_inner_samples=N_INNER_SAMPLES
    )

    # Change Detectors -------------------------------------------------------------------------------------------------
    fi_detectors = {feature_name: ADWIN(delta=0.001) for feature_name in feature_names}
    change_points = []
    change_points_fi = []

    # warm-up because of errors ----------------------------------------------------------------------------------------
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        model.learn_one(x_i, y_i)
        if n > 10:
            break

    # storing ----------------------------------------------------------------------------------------------------------
    plotter = FeatureImportancePlotter(
        feature_names=feature_names
    )
    model_performance = []

    # run --------------------------------------------------------------------------------------------------------------
    for (n, (x_i, y_i)) in enumerate(stream, start=1):

        # predicting
        if task == REG:
            y_i_pred = model.predict_one(x_i)
        else:
            y_i_pred = model.predict_proba_one(x_i)
        rolling_training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model_performance.append(rolling_training_metric.get())

        # sage inc
        fi_values = incremental_sage.explain_one(x_i, y_i)

        # storing data
        plotter.update(fi_values, facet_name='inc')

        # change detection
        detected_drifts = 0
        for feature_name in feature_names:
            fi_detector = fi_detectors[feature_name]
            fi_detector.update(fi_values[feature_name])
            if fi_detector.drift_detected:
                detected_drifts += 1

        if detected_drifts >= 1:
            change_points.append(n)
            change_points_fi.append({**incremental_sage.importance_values, 'number_of_drifts': detected_drifts})

        # learning
        model.learn_one(x_i, y_i)

        if DEBUG and n % 1000 == 0:
            print(f"{n}: x_i               {x_i}\n"
                  f"{n}: inc-sage          {incremental_sage.importance_values}\n"
                  f"{n}: inc-sage-var      {incremental_sage.variances}\n"
                  )

        if n >= n_samples:
            break

    # Store the results in a database ----------------------------------------------------------------------------------

    #v_lines = [{'x': CONCEPT_DRIFT_POSITION, 'ls': '--', 'c': 'black', 'linewidth': 1}]
    v_lines = []
    v_lines.extend([{'x': fi_drift, 'ls': '--', 'c': 'gray', 'linewidth': 1}
                    for fi_drift in change_points])
    plotter.plot(
        figsize=(5, 10),
        model_performance={"cross\nentropy": model_performance},
        title=f'Agrawal Stream with Concept Drift',
        y_label='SAGE values',
        x_label='Samples',
        names_to_highlight=['nswprice', 'vicprice', 'nswdemand'],
        v_lines=v_lines,
        h_lines=[{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
        line_styles={'inc': '-'},
        legend_style={},
        markevery={'inc': 10},
        y_min=-0.1,
        y_max=0.75
    )


    change_points_fi = pd.DataFrame(change_points_fi)
    change_points_fi['t'] = change_points
    change_points_fi.to_csv("changes_detected.csv", index=False)


