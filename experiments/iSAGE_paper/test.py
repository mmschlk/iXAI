import time

import pandas as pd

from experiments.setup.data import get_dataset
from experiments.setup.explainer import get_imputer_and_storage
from experiments.setup.loss import get_loss_function, get_training_metric
from experiments.setup.model import get_model
from increment_explain.explainer.sage import IncrementalSageExplainer
from increment_explain.utils.tracker.exponential_smoothing import ExponentialSmoothingTracker
from increment_explain.visualization import FeatureImportancePlotter

DEBUG = True

DATASET_1_NAME = 'stream_bike'

DATASET_RANDOM_SEED = 1
SHUFFLE_DATA = False
N_SAMPLES_1 = None

MODEL_NAME = 'ARF'

N_INNER_SAMPLES: int = 1
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
    dataset = get_dataset(
        dataset_name=DATASET_1_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES_1)
    feature_names = dataset.feature_names
    feature_names.remove('moment')
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

    # warm-up because of errors ----------------------------------------------------------------------------------------
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        del x_i['moment']
        print(x_i)
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

    model_loss_tracker = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
    marginal_loss_tracker = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
    model_loss = []
    marginal_loss = []

    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        del x_i['moment']
        # predicting
        y_i_pred = model_function(x_i)
        loss_i = loss_function(y_true=y_i, y_prediction=y_i_pred)
        model_loss_tracker.update(loss_i)
        model_loss.append({"loss": model_loss_tracker.get()})
        loss_marginal_i = loss_function(y_true=y_i, y_prediction=incremental_sage._marginal_prediction_tracker.get())
        marginal_loss_tracker.update(loss_marginal_i)
        marginal_loss.append({"loss": marginal_loss_tracker.get()})

        # sage inc
        #start_time = time.time()
        #fi_values = incremental_welford_sage.explain_one(x_i, y_i)
        #time_sage += time.time() - start_time
        #sage_welford_fi_values.append(fi_values)
        #plotter.update(fi_values, facet_name='inc-w')

        # sage inc
        #start_time = time.time()
        #fi_values = incremental_sage.explain_one(x_i, y_i)
        #time_sage += time.time() - start_time
        #sage_fi_values.append(fi_values)
        #plotter.update(fi_values, facet_name='inc')

        # learning
        start_time = time.time()
        model.learn_one(x_i, y_i)
        time_learning += time.time() - start_time

        if DEBUG and n % 1000 == 0:
            print(f"{n}: x_i                 {x_i}\n"
                  f"{n}: x_i                 {model_loss_tracker.get()}\n"
                  f"{n}: inc-sage            {incremental_sage.importance_values}\n"
                  f"{n}: inc-sage-welford    {incremental_welford_sage.importance_values}\n"
                  )

        if n >= n_samples:
            break

    # Store the results in a database ----------------------------------------------------------------------------------

    facet = "left.csv"
    data_folder = "static_sage_in_dynamic_setting_insects"

    name = "inc_fi_values"
    df = pd.DataFrame(sage_fi_values)
    df.to_csv(f"plots/{data_folder}/{'_'.join((name, facet))}", index=False)

    name = "inc_welford_fi_values"
    df = pd.DataFrame(sage_welford_fi_values)
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
        h_lines=[{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
        line_styles={'inc': '-', 'inc-w': '--'},
        legend_style={"fontsize": "small"},
        markevery={'inc': 10, 'inc-w': 10},
        y_min=-0.05,
        y_max=0.42,
    )
