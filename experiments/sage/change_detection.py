import time

from river.drift import ADWIN, KSWIN, DDM, HDDM_A

from experiments.setup.data import get_dataset, get_concept_drift_dataset
from experiments.setup.explainer import get_incremental_sage_explainer, \
    get_batch_sage_explainer, get_interval_sage_explainer, get_incremental_pfi_explainer
from experiments.setup.loss import get_loss_function, get_training_metric
from experiments.setup.model import get_model

from matplotlib import pyplot as plt

from increment_explain.utils.trackers import SlidingWindowTracker
from increment_explain.visualization import FeatureImportancePlotter

DATASET_1_NAME = 'agrawal 1'
DATASET_2_NAME = 'agrawal 2'
DATASET_RANDOM_SEED = 1
SHUFFLE_DATA = False
N_SAMPLES_1 = 5000
N_SAMPLES_2 = 5000
N_SAMPLES = N_SAMPLES_1 + N_SAMPLES_2

CONCEPT_DRIFT_POSITION = int(0.5 * N_SAMPLES)
CONCEPT_DRIFT_WIDTH = int(1)
CONCEPT_DRIFT_SWITCHING_FEATURES = None #'salary_age'

MODEL_NAME = 'ARF'

N_INNER_SAMPLES = 1
N_INTERVAL_LENGTH = 2000

TRAINING_METRIC_WINDOW = 100

MODEL_PARAMS = {
    'ARF': {
        'n_models': 25,
    }
}

if __name__ == "__main__":

    dataset_1 = get_dataset(dataset_name=DATASET_1_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES_1)
    dataset_2 = get_dataset(dataset_name=DATASET_2_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES_2)

    dataset = get_concept_drift_dataset(
        dataset_1_name=DATASET_1_NAME, dataset_2_name=DATASET_2_NAME,
        dataset_1=dataset_1, dataset_2=dataset_2,
        position=CONCEPT_DRIFT_POSITION, width=CONCEPT_DRIFT_WIDTH,
        features_to_switch=CONCEPT_DRIFT_SWITCHING_FEATURES
    )

    feature_names = dataset.feature_names
    n_samples = dataset.n_samples
    stream = dataset.stream

    task = dataset.task

    loss_function = get_loss_function(task=task)
    rolling_training_metric, training_metric = get_training_metric(task=task, rolling_window=TRAINING_METRIC_WINDOW)

    model, model_function = get_model(
        model_name=MODEL_NAME, task=task, feature_names=feature_names, **MODEL_PARAMS[MODEL_NAME])

    performance_drift_detector = ADWIN(delta=0.001)
    #fi_detectors = {feature_name: KSWIN(alpha=0.0001, window_size=500) for feature_name in feature_names}
    fi_detectors = {feature_name: ADWIN(delta=0.00001) for feature_name in feature_names}
    moving_average_large = SlidingWindowTracker(k=2000)
    moving_average_small = SlidingWindowTracker(k=500)


    incremental_sage = get_incremental_sage_explainer(
        feature_names=feature_names,
        model_function=model_function,
        loss_function=loss_function,
        n_inner_samples=N_INNER_SAMPLES,
    )

    plotter = FeatureImportancePlotter(
        feature_names=feature_names
    )

    # warm-up because of errors ------------------------------------------------------------------------------------
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        model.learn_one(x_i, y_i)
        if n > 10:
            break

    # Concept 1 ----------------------------------------------------------------------------------------------------

    performance_drifts = []
    fi_drifts = []

    moving_average_large_values = []
    moving_average_small_values = []

    model_performance = []
    time_pfi = 0.
    time_sage = 0.
    time_interval = 0.
    time_learning = 0.

    sage_fi_values = []
    interval_fi_values = []
    pfi_fi_values = []

    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        # predicting
        y_i_pred = model.predict_one(x_i)
        rolling_training_metric.update(y_true=y_i, y_pred=y_i_pred)
        current_performance = rolling_training_metric.get()
        model_performance.append({'performance': current_performance})

        # sage inc
        start_time = time.time()
        fi_values = incremental_sage.explain_one(x_i, y_i)
        time_sage += time.time() - start_time
        plotter.update(importance_values=fi_values, facet_name='inc-sage')

        # concept drift detectors
        performance_drift_detector.update(int(y_i == y_i_pred))
        if performance_drift_detector.drift_detected:
            performance_drifts.append(n)

        detected_drifts = 0
        for feature_name in ['salary']:

            moving_average_large.update(fi_values[feature_name])
            moving_average_large_values.append(moving_average_large.mean)
            moving_average_small.update(fi_values[feature_name])
            moving_average_small_values.append(moving_average_small.mean)

            fi_detector = fi_detectors[feature_name]
            fi_detector.update(fi_values[feature_name])
            if fi_detector.drift_detected:
                detected_drifts += 1
                if detected_drifts >= 1:
                    fi_drifts.append(n)

        # learning
        start_time = time.time()
        model.learn_one(x_i, y_i)
        time_learning += time.time() - start_time

        if n % 1000 == 0:
            print(f"{n}: x_i         {x_i}\n"
                  f"{n}: performance {rolling_training_metric.get()}\n"
                  f"{n}: inc-sage    {incremental_sage.importance_values}\n")
        if n >= n_samples:
            break

    v_lines = [{'x': CONCEPT_DRIFT_POSITION, 'ls': '--', 'c': 'black', 'linewidth': 1}]
    v_lines.extend([{'x': performance_drift, 'ls': '--', 'c': 'red', 'linewidth': 1}
                    for performance_drift in performance_drifts])
    v_lines.extend([{'x': fi_drift, 'ls': '--', 'c': 'gray', 'linewidth': 1}
                   for fi_drift in fi_drifts])

    plotter.plot(
        figsize=(5, 10),
        model_performance={'performance': model_performance},
        title=f'Agrawal Stream concept drift stream',
        y_label='SAGE values',
        x_label='Samples',
        names_to_highlight=['salary', 'commission', 'age', 'elevel'],
        v_lines=v_lines,
        h_lines=[{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
        line_styles={'inc-sage': 'solid', 'int-sage': 'dashed', 'inc-pfi': 'dotted'},
        legend_style={}
    )

    fig, axis = plt.subplots(nrows=1, ncols=1)
    axis.plot(moving_average_large_values, color="blue")
    axis.plot(moving_average_small_values, color="red")
    axis.axvline(**{'x': CONCEPT_DRIFT_POSITION, 'ls': '--', 'c': 'black', 'linewidth': 1})
    plt.show()
