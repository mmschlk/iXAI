import time

import matplotlib.pyplot as plt
from river.utils import Rolling

from explainer.sage import IncrementalSAGE
from data.synth.synthetic import SyntheticDataset

import copy
import numpy as np

from river.ensemble import AdaptiveRandomForestRegressor, AdaptiveRandomForestClassifier
from river import metrics
from river.stream import iter_array

from increment_explain.utils import RiverToPredictionFunction, PredictionFunctionToRiverInput
from increment_explain.visualization import plot_multi_line_graph


def _dataset_target_function_regression_1(x):
    return np.dot(x, _WEIGHTS_1)


def _dataset_target_function_classification_1(x):
    return int(np.dot(x, _WEIGHTS_1) >= _CLASSIFICATION_THRESHOLD_1)


def _dataset_target_function_regression_2(x):
    return np.dot(x, _WEIGHTS_2)


def _dataset_target_function_classification_2(x):
    return int(np.dot(x, _WEIGHTS_2) >= _CLASSIFICATION_THRESHOLD_2)


if __name__ == "__main__":

    DEBUG = True
    RANDOM_SEED = 1

    # Setup Data -------------------------------------------------------------------------------------------------------

    # Problem Definition
    _N_FEATURES = 10
    _WEIGHTS_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    _WEIGHTS_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 10]
    _CLASSIFICATION_THRESHOLD_1 = 27.5
    _CLASSIFICATION_THRESHOLD_2 = 5
    CLASSIFICATION = False
    REAL_MODEL = True

    # Samples per stream
    N_STREAM_1 = 10000
    N_STREAM_2 = 10000
    BATCH_SIZE = 1

    # Default Values
    _DEFAULT_VALUES = [0.5 for _ in range(_N_FEATURES)]

    if CLASSIFICATION:
        _DATASET_TARGET_FUNCTION_1 = _dataset_target_function_classification_1
        _DATASET_TARGET_FUNCTION_2 = _dataset_target_function_classification_2
    else:
        _DATASET_TARGET_FUNCTION_1 = _dataset_target_function_regression_1
        _DATASET_TARGET_FUNCTION_2 = _dataset_target_function_regression_2

    # Get Data Streams
    dataset_1 = SyntheticDataset(n_numeric=_N_FEATURES, n_features=_N_FEATURES, n_samples=N_STREAM_1*BATCH_SIZE,
                                 noise_std=0, target_function=_DATASET_TARGET_FUNCTION_1, random_seed=RANDOM_SEED,
                                 classification=CLASSIFICATION)
    x_stream_1 = dataset_1.x_data[:N_STREAM_1]
    y_stream_1 = dataset_1.y_data[:N_STREAM_1]
    dataset_2 = SyntheticDataset(n_numeric=_N_FEATURES, n_features=_N_FEATURES, n_samples=N_STREAM_2*BATCH_SIZE,
                                 noise_std=0, target_function=_DATASET_TARGET_FUNCTION_2, random_seed=RANDOM_SEED,
                                 classification=CLASSIFICATION)
    x_stream_2 = dataset_2.x_data[:N_STREAM_2]
    y_stream_2 = dataset_2.y_data[:N_STREAM_2]

    # Get the feature names for both streams
    feature_names = dataset_1.feature_names

    # Get Model and Performance Metric
    if CLASSIFICATION:
        metric = metrics.Accuracy()
        if REAL_MODEL:
            model = AdaptiveRandomForestClassifier(seed=RANDOM_SEED)
            model_function = RiverToPredictionFunction(model, feature_names).predict
        else:
            model_function = PredictionFunctionToRiverInput(prediction_function=_dataset_target_function_classification_1)
    else:
        metric = metrics.MAE()
        if REAL_MODEL:
            model = AdaptiveRandomForestRegressor(seed=RANDOM_SEED)
            model_function = RiverToPredictionFunction(model, feature_names).predict
        else:
            model_function = PredictionFunctionToRiverInput(prediction_function=_dataset_target_function_regression_1)
    metric = Rolling(metric, window_size=1000)
    time_incremental = 0.0000001
    time_batch = 0.0000001
    incremental_SAGE_values = []

    # IncrementalSAGE Stream 1 -----------------------------------------------------------------------------------------
    time.sleep(0.05)
    explainer = IncrementalSAGE(
        model_fn=model_function,
        feature_names=feature_names,
        loss_function='mse',
        empty_prediction=None,
        sub_sample_size=10,
        # default_values=_DEFAULT_VALUES
        default_values=None
    )
    for (n, (x_i, y_i)) in enumerate(iter_array(x_stream_1, y_stream_1, feature_names=feature_names)):
        n += 1
        if REAL_MODEL:
            model.learn_one(x_i, y_i)
        start_time = time.time()
        y_i_pred = model_function(x_i)
        metric.update(y_true=y_i, y_pred=y_i_pred[0])
        SAGE_values = explainer.explain_one(x_i=x_i, y_i=y_i)
        incremental_SAGE_values.append(copy.deepcopy(SAGE_values))
        time_incremental += time.time() - start_time
        if n >= N_STREAM_1:
            break
        if DEBUG and n % 1000 == 0:
            print(n)
            print("Score:", metric.get())
            print("Sage Values", SAGE_values)
    SAGE_values = explainer.SAGE_trackers
    # sage_values_mean = [round(sage_value.mean, 2) for _, sage_value in SAGE_values.items()]
    # sage_values_std = [round(sage_value.std, 2) for _, sage_value in SAGE_values.items()]
    sage_values_mean = [round(sage_value(), 2) for _, sage_value in SAGE_values.items()]
    print("IncrementalSAGE Explanation(")
    print("  (Mean):", sage_values_mean)
    print(")")
    print("Time:", round(time_incremental, 2))
    """
    # Original SAGE Stream 1 -------------------------------------------------------------------------------------------
    time.sleep(0.05)
    start_time = time.time()
    imputer = sage.DefaultImputer(model=model_function, values=np.asarray(_DEFAULT_VALUES))
    estimator = sage.PermutationEstimator(imputer, 'mse')
    sage_values = estimator(x_stream_1, y_stream_1,
                            n_permutations=N_STREAM_1,
                            detect_convergence=False,
                            batch_size=1)
    time_batch += time.time() - start_time
    sage_values_mean = [round(sage_value, 2) for sage_value in sage_values.values]
    sage_values_std = [round(sage_value, 2) for sage_value in sage_values.std]
    print("Batch SAGE Explanation(")
    print("  (Mean):", sage_values_mean)
    print("  (Std):", sage_values_std)
    print(")")
    print("Time:", round(time_batch, 2))

    # Kernel SAGE Stream 1 -------------------------------------------------------------------------------------------
    time.sleep(0.05)
    start_time = time.time()
    imputer = sage.DefaultImputer(model=model_function, values=np.asarray(_DEFAULT_VALUES))
    estimator = sage.KernelEstimator(imputer, 'mse')
    sage_values = estimator(x_stream_1, y_stream_1,
                            n_samples=N_STREAM_1,
                            detect_convergence=False,
                            batch_size=1)
    time_batch += time.time() - start_time
    sage_values_mean = [round(sage_value, 2) for sage_value in sage_values.values]
    sage_values_std = [round(sage_value, 2) for sage_value in sage_values.std]
    print("Batch Kernel SAGE Explanation(")
    print("  (Mean):", sage_values_mean)
    print("  (Std):", sage_values_std)
    print(")")
    print("Time:", round(time_batch, 2))
    """

    # Concept Drift ----------------------------------------------------------------------------------------------------
    if not REAL_MODEL:
        if CLASSIFICATION:
            model_function = PredictionFunctionToRiverInput(prediction_function=_dataset_target_function_classification_2)
        else:
            model_function = PredictionFunctionToRiverInput(prediction_function=_dataset_target_function_regression_2)

    # IncrementalSAGE Stream 2 -----------------------------------------------------------------------------------------
    time.sleep(0.05)
    explainer.model_fn = model_function
    start_time = time.time()
    for (n, (x_i, y_i)) in enumerate(iter_array(x_stream_2, y_stream_2, feature_names=feature_names)):
        n += 1
        if REAL_MODEL:
            model.learn_one(x_i, y_i)
        start_time = time.time()
        y_i_pred = model_function(x_i)
        metric.update(y_true=y_i, y_pred=y_i_pred[0])
        SAGE_values = explainer.explain_one(x_i=x_i, y_i=y_i)
        incremental_SAGE_values.append(copy.deepcopy(SAGE_values))
        time_incremental += time.time() - start_time
        if n >= N_STREAM_2:
            break
        if DEBUG and n % 1000 == 0:
            print(n)
            print("Score:", metric.get())
            print("Sage Values", SAGE_values)
    SAGE_values = explainer.SAGE_trackers
    end_time = time.time()
    sage_values_mean = [round(sage_value(), 2) for _, sage_value in SAGE_values.items()]
    print("IncrementalSAGE Explanation(")
    print("  (Mean):", sage_values_mean)
    print(")")
    print("Time:", round(time_incremental, 2))

    """
    # Original SAGE Stream 2 -------------------------------------------------------------------------------------------
    time.sleep(0.05)
    start_time = time.time()
    imputer = sage.DefaultImputer(model=model_function, values=np.asarray(_DEFAULT_VALUES))
    estimator = sage.PermutationEstimator(imputer, 'mse')
    sage_values = estimator(x_stream_2, y_stream_2,
                            n_permutations=N_STREAM_2,
                            detect_convergence=False,
                            batch_size=1)
    time_batch += time.time() - start_time
    sage_values_mean = [round(sage_value, 2) for sage_value in sage_values.values]
    sage_values_std = [round(sage_value, 2) for sage_value in sage_values.std]
    print("Batch SAGE Explanation(")
    print("  (Mean):", sage_values_mean)
    print("  (Std):", sage_values_std)
    print(")")
    print("Time:", round(time_batch, 2))

    time.sleep(0.05)
    start_time = time.time()
    imputer = sage.DefaultImputer(model=model_function, values=np.asarray(_DEFAULT_VALUES))
    estimator = sage.KernelEstimator(imputer, 'mse')
    sage_values = estimator(x_stream_2, y_stream_2,
                            n_samples=N_STREAM_1,
                            detect_convergence=False,
                            batch_size=1)
    time_batch += time.time() - start_time
    sage_values_mean = [round(sage_value, 2) for sage_value in sage_values.values]
    sage_values_std = [round(sage_value, 2) for sage_value in sage_values.std]
    print("Batch KernelSAGE Explanation(")
    print("  (Mean):", sage_values_mean)
    print("  (Std):", sage_values_std)
    print(")")
    print("Time:", round(time_batch, 2))

    # Summary Time
    print()
    print("Time Dif. (inc / batch):", time_incremental / time_batch)
    """

    # Plotting ---------------------------------------------------------------------------------------------------------

    fig, axis = plt.subplots()
    axis = plot_multi_line_graph(axis=axis,
                                 y_data=incremental_SAGE_values,
                                 names_to_highlight=feature_names,
                                 line_names=feature_names)
    plt.show()
