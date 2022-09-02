import river.compat
from tqdm import tqdm

from explainer.sage import IncrementalSAGE
from data.synthetic import SyntheticDataset

import copy
import numpy as np

from river.ensemble import AdaptiveRandomForestRegressor
from river import metrics
from river.stream import iter_array

import sage

from utils.converters import RiverToPredictionFunction


if __name__ == "__main__":

    # Problem Definition
    _N_FEATURES = 10
    #_WEIGHTS = [i for i in range(_N_FEATURES)]
    _WEIGHTS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 10]
    #_WEIGHTS = [1, 0, 0]
    #_EMPTY_PREDICTION = np.dot([0.5 for _ in range(_N_FEATURES)], _WEIGHTS)
    #_EMPTY_PREDICTION = 0.5
    _EMPTY_PREDICTION = None

    def _dataset_target_function(x):
        return np.dot(x, _WEIGHTS)

    def _model_function_dict(x_dict):
        x_list = list(x_dict.values())
        return _dataset_target_function(x_list)

    # Incremental Definitions
    USE_INCREMENTAL_SAGE = True
    TRAIN_TIME = 1
    N_INCREMENTAL_SAGE_EXPLAIN = 10000

    # Original SAGE Definitions
    USE_SAGE = True
    N_SAGE_EXPLAIN = 1000

    n_samples = max(N_SAGE_EXPLAIN, N_INCREMENTAL_SAGE_EXPLAIN)
    dataset = SyntheticDataset(n_numeric=_N_FEATURES, n_features=_N_FEATURES, n_samples=n_samples,
                               noise_std=0.01, target_function=_dataset_target_function)
    stream = dataset.to_stream()
    feature_names = dataset.feature_names
    model = AdaptiveRandomForestRegressor()

    metric = metrics.MAE()

    for (n, (x_i, y_i)) in enumerate(stream):
        n += 1
        y_i_pred = model.predict_one(x_i)
        model.learn_one(x_i, y_i)
        metric.update(y_true=y_i, y_pred=y_i_pred)
        if n % 1000 == 0:
            print(n, "Score", metric.get())
        if n >= TRAIN_TIME:
            break

    x_data = dataset.x_data
    y_data = dataset.y_data

    print(np.mean(x_data, axis=0))

    # model_function = RiverToPredictionFunction(model, feature_names).predict

    # Original Sage
    if USE_SAGE:
        model_function = _dataset_target_function
        x_explain = x_data[:N_SAGE_EXPLAIN]
        y_explain = y_data[:N_SAGE_EXPLAIN]
        imputer = sage.MarginalImputer(model=model_function, data=x_explain)
        estimator = sage.PermutationEstimator(imputer, 'mse')
        sage_values = estimator(x_explain, y_explain, n_permutations=10000, detect_convergence=False, batch_size=10)
        print(sage_values)
        print(_WEIGHTS)

    # IncrementalSAGE
    if USE_INCREMENTAL_SAGE:
        model_function = _model_function_dict
        x_explain = x_data[:N_INCREMENTAL_SAGE_EXPLAIN]
        y_explain = y_data[:N_INCREMENTAL_SAGE_EXPLAIN]
        explainer = IncrementalSAGE(
            model_fn=_model_function_dict,
            feature_names=feature_names,
            loss_function='mse',
            empty_prediction=_EMPTY_PREDICTION,
            sub_sample_size=10
        )
        SAGE_values_run = []
        for (n, (x_i, y_i)) in enumerate(iter_array(x_explain, y_explain, feature_names=feature_names)):
            n += 1
            y_i_pred = _model_function_dict(x_i)
            SAGE_values = explainer.explain_one(x_i=x_i, y_i=y_i)
            SAGE_values_run.append(copy.deepcopy(SAGE_values))
            if n >= N_INCREMENTAL_SAGE_EXPLAIN:
                break
            if n % 1000 == 0:
                print(n, SAGE_values)
        print(SAGE_values_run[-1])

        print(np.sum((y_explain - np.mean(x_explain[:, -1]))**2) / len(x_explain))
