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

DATASET_1_NAME = 'agrawal 1'
DATASET_2_NAME = 'agrawal 2'
DATASET_3_NAME = 'agrawal 1'
DATASET_4_NAME = 'agrawal 2'
DATASET_RANDOM_SEED = 1
SHUFFLE_DATA = False
N_SAMPLES = 25000

MODEL_NAME = 'ARF'

N_INNER_SAMPLES: int = 4
N_INTERVAL_LENGTH: int = 5000
CONFIDENCE_BOUNDS_DELTA: float = 0.1
SMOOTHING_ALPHA: float = 0.001
FEATURE_REMOVAL_DISTRIBUTION: str = 'marginal joint'  # ['marginal joint', 'marginal product', 'conditional']
RESERVOIR_LENGTH: int = 100
RESERVOIR_KIND: str = 'geometric'  # ['geometric', 'uniform']

TRAINING_METRIC_WINDOW = 500

MODEL_PARAMS = {
    'ARF': {
        'n_models': 15,
    }
}


if __name__ == "__main__":

    # Get Data ---------------------------------------------------------------------------------------------------------
    dataset = get_dataset(
        dataset_name=DATASET_1_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=10)
    dataset_1 = get_dataset(
        dataset_name=DATASET_1_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES)
    dataset_2 = get_dataset(
        dataset_name=DATASET_2_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES)
    dataset_3 = get_dataset(
        dataset_name=DATASET_3_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES)
    dataset_4 = get_dataset(
        dataset_name=DATASET_4_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES)
    feature_names = dataset.feature_names
    cat_feature_names = dataset.cat_feature_names
    num_feature_names = dataset.num_feature_names
    stream_0 = dataset
    n_samples = N_SAMPLES * 4
    stream_1 = dataset_1.stream
    stream_2 = dataset_2.stream
    stream_3 = dataset_3.stream
    stream_4 = dataset_4.stream

    # Get Model / Loss and Training Procedure --------------------------------------------------------------------------
    task = dataset_1.task
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
    for (n, (x_i, y_i)) in enumerate(stream_0, start=1):
        model.learn_one(x_i, y_i)


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

    for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
        # predicting
        y_i_pred = model_function(x_i)
        loss_i = loss_function(y_true=y_i, y_prediction=y_i_pred)
        model_loss_tracker.update(loss_i)
        model_loss.append({"loss": model_loss_tracker.get()})
        loss_marginal_i = loss_function(y_true=y_i, y_prediction=incremental_sage._marginal_prediction_tracker.get())
        marginal_loss_tracker.update(loss_marginal_i)
        marginal_loss.append({"loss": marginal_loss_tracker.get()})

        # sage inc
        fi_values = incremental_welford_sage.explain_one(x_i, y_i)
        sage_welford_fi_values.append(fi_values)

        # sage inc
        fi_values = incremental_sage.explain_one(x_i, y_i)
        sage_fi_values.append(fi_values)

        # sage int
        fi_values = interval_explainer.explain_one(x_i, y_i)
        interval_fi_values.append(fi_values)

        # learning
        model.learn_one(x_i, y_i)

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

        if n >= N_SAMPLES:
            break

    for (n, (x_i, y_i)) in enumerate(stream_2, start=1):
        # predicting
        y_i_pred = model_function(x_i)
        loss_i = loss_function(y_true=y_i, y_prediction=y_i_pred)
        model_loss_tracker.update(loss_i)
        model_loss.append({"loss": model_loss_tracker.get()})
        loss_marginal_i = loss_function(y_true=y_i, y_prediction=incremental_sage._marginal_prediction_tracker.get())
        marginal_loss_tracker.update(loss_marginal_i)
        marginal_loss.append({"loss": marginal_loss_tracker.get()})

        # sage inc
        fi_values = incremental_welford_sage.explain_one(x_i, y_i)
        sage_welford_fi_values.append(fi_values)

        # sage inc
        fi_values = incremental_sage.explain_one(x_i, y_i)
        sage_fi_values.append(fi_values)

        # sage int
        fi_values = interval_explainer.explain_one(x_i, y_i)
        interval_fi_values.append(fi_values)

        # learning
        model.learn_one(x_i, y_i)

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

        if n >= N_SAMPLES:
            break

    for (n, (x_i, y_i)) in enumerate(stream_3, start=1):
        # predicting
        y_i_pred = model_function(x_i)
        loss_i = loss_function(y_true=y_i, y_prediction=y_i_pred)
        model_loss_tracker.update(loss_i)
        model_loss.append({"loss": model_loss_tracker.get()})
        loss_marginal_i = loss_function(y_true=y_i, y_prediction=incremental_sage._marginal_prediction_tracker.get())
        marginal_loss_tracker.update(loss_marginal_i)
        marginal_loss.append({"loss": marginal_loss_tracker.get()})

        # sage inc
        fi_values = incremental_welford_sage.explain_one(x_i, y_i)
        sage_welford_fi_values.append(fi_values)

        # sage inc
        fi_values = incremental_sage.explain_one(x_i, y_i)
        sage_fi_values.append(fi_values)

        # sage int
        fi_values = interval_explainer.explain_one(x_i, y_i)
        interval_fi_values.append(fi_values)

        # learning
        model.learn_one(x_i, y_i)

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

        if n >= N_SAMPLES:
            break

    for (n, (x_i, y_i)) in enumerate(stream_4, start=1):
        # predicting
        y_i_pred = model_function(x_i)
        loss_i = loss_function(y_true=y_i, y_prediction=y_i_pred)
        model_loss_tracker.update(loss_i)
        model_loss.append({"loss": model_loss_tracker.get()})
        loss_marginal_i = loss_function(y_true=y_i, y_prediction=incremental_sage._marginal_prediction_tracker.get())
        marginal_loss_tracker.update(loss_marginal_i)
        marginal_loss.append({"loss": marginal_loss_tracker.get()})

        # sage inc
        fi_values = incremental_welford_sage.explain_one(x_i, y_i)
        sage_welford_fi_values.append(fi_values)

        # sage inc
        fi_values = incremental_sage.explain_one(x_i, y_i)
        sage_fi_values.append(fi_values)

        # sage int
        fi_values = interval_explainer.explain_one(x_i, y_i)
        interval_fi_values.append(fi_values)

        # learning
        model.learn_one(x_i, y_i)

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

        if n >= N_SAMPLES:
            break

    # Store the results in a database ----------------------------------------------------------------------------------

    facet = "left.csv"
    data_folder = "reoccurring_drift"

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