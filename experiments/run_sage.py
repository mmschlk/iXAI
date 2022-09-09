import time

import river.compat
from river.utils import Rolling
from tqdm import tqdm

from explainer.sage import IncrementalSAGE
from data.synthetic import SyntheticDataset

import copy
import numpy as np

from river.ensemble import AdaptiveRandomForestRegressor, AdaptiveRandomForestClassifier
from river.linear_model import Perceptron, LinearRegression
from river import metrics
from river.stream import iter_array

import sage

from utils.converters import RiverToPredictionFunction


def _dataset_target_function_regression(x):
    return np.dot(x, _WEIGHTS)


def _dataset_target_function_classification(x):
    return int(np.dot(x, _WEIGHTS) >= _CLASSIFICATION_THRESHOLD)


def _model_function_dict(x_dict):
    x_list = list(x_dict.values())
    return _DATASET_TARGET_FUNCTION(x_list)


if __name__ == "__main__":

    RANDOM_SEED = 36

    # Problem Definition
    _N_FEATURES = 10
    _WEIGHTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 10]
    _CLASSIFICATION_THRESHOLD = 27.5  # 5
    CLASSIFICATION = False

    # Empty prediction
    _EMPTY_PREDICTION = None  #_EMPTY_PREDICTION = np.dot([0.5 for _ in range(_N_FEATURES)], _WEIGHTS)

    # Default Values
    _DEFAULT_VALUES = [0.5 for _ in range(_N_FEATURES)]

    if CLASSIFICATION:
        _DATASET_TARGET_FUNCTION = _dataset_target_function_classification
    else:
        _DATASET_TARGET_FUNCTION = _dataset_target_function_regression

    N_TRAINING = 1
    N_EXPLAIN = 10000

    # Incremental Definitions
    USE_INCREMENTAL_SAGE = True
    TRAIN_TIME = N_TRAINING
    PRINT_INCREMENTAL = True

    # Original SAGE Definitions
    USE_SAGE = True

    n_samples = max(N_EXPLAIN, N_TRAINING)
    dataset = SyntheticDataset(n_numeric=_N_FEATURES, n_features=_N_FEATURES, n_samples=n_samples,
                               noise_std=0, target_function=_DATASET_TARGET_FUNCTION, random_seed=RANDOM_SEED,
                               classification=CLASSIFICATION)
    stream = dataset.to_stream()
    feature_names = dataset.feature_names
    """
    if CLASSIFICATION:
        metric = metrics.Accuracy()
        model = AdaptiveRandomForestClassifier(seed=RANDOM_SEED)
    else:
        metric = metrics.MAE()
        model = AdaptiveRandomForestRegressor(seed=RANDOM_SEED)
    metric = Rolling(metric, window_size=1000)

    for (n, (x_i, y_i)) in enumerate(stream):
        n += 1
        y_i_pred = model.predict_one(x_i)
        model.learn_one(x_i, y_i)
        metric.update(y_true=y_i, y_pred=y_i_pred)
        if n % 1000 == 0:
            print(n, "Score:", metric.get())
        if n >= TRAIN_TIME:
            break

    print("Score:", metric.get())

    model_function = RiverToPredictionFunction(model, feature_names).predict
    """
    x_data = dataset.x_data
    y_data = dataset.y_data

    print(_WEIGHTS)
    print(np.mean(x_data, axis=0))
    print(np.mean(y_data, axis=0))

    x_explain = x_data[:N_EXPLAIN]
    y_explain = y_data[:N_EXPLAIN]

    if CLASSIFICATION:
        metric = metrics.Accuracy()
        model = AdaptiveRandomForestClassifier(seed=RANDOM_SEED)
    else:
        metric = metrics.MAE()
        model = AdaptiveRandomForestRegressor(seed=RANDOM_SEED)
    metric = Rolling(metric, window_size=1000)
    model_function = RiverToPredictionFunction(model, feature_names).predict

    # IncrementalSAGE
    if USE_INCREMENTAL_SAGE:
        start_time = time.time()
        explainer = IncrementalSAGE(
            model_fn=model_function,
            feature_names=feature_names,
            loss_function='mse',
            empty_prediction=_EMPTY_PREDICTION,
            sub_sample_size=1,
            default_values=_DEFAULT_VALUES
        )
        SAGE_values_run = []
        for (n, (x_i, y_i)) in enumerate(iter_array(x_explain, y_explain, feature_names=feature_names)):
            n += 1
            model.learn_one(x_i, y_i)
            y_i_pred = model_function(x_i)
            metric.update(y_true=y_i, y_pred=y_i_pred[0])
            SAGE_values = explainer.explain_one(x_i=x_i, y_i=y_i)
            SAGE_values_run.append(copy.deepcopy(SAGE_values))
            if n >= N_EXPLAIN:
                break
            if PRINT_INCREMENTAL and n % 1000 == 0:
                print(n)
                print("Score:", metric.get())
                print("Sage Values", SAGE_values)
        end_time = time.time()
        sage_values_mean = [round(sage_value.mean, 2) for _, sage_value in SAGE_values.items()]
        sage_values_std = [round(sage_value.std, 2) for _, sage_value in SAGE_values.items()]
        time_inc_sage = round(end_time - start_time, 2)

        print("IncrementalSAGE Explanation(")
        print("  (Mean):", sage_values_mean)
        print("  (Std):", sage_values_std)
        print(")")
        print("Time:", time_inc_sage)


        #print(np.sum((y_explain - np.mean(x_explain[:, -1]))**2) / len(x_explain))

        # Original Sage
        if USE_SAGE:
            start_time = time.time()
            imputer = sage.DefaultImputer(model=model_function, values=np.asarray(_DEFAULT_VALUES))
            estimator = sage.PermutationEstimator(imputer, 'mse')
            sage_values = estimator(x_explain, y_explain,
                                    n_permutations=N_EXPLAIN,
                                    detect_convergence=False,
                                    batch_size=1)
            end_time = time.time()
            sage_values_mean = [round(sage_value, 2) for sage_value in sage_values.values]
            sage_values_std = [round(sage_value, 2) for sage_value in sage_values.std]
            print("Batch SAGE Explanation(")
            print("  (Mean):", sage_values_mean)
            print("  (Std):", sage_values_std)
            print(")")
            time_sage = round(end_time - start_time, 2)
            print("Time:", time_sage)

    print()
    print("Time Dif. (inc / batch):", time_inc_sage / time_sage)