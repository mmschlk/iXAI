from data.synthetic import SyntheticDataset
import numpy as np
from river.ensemble import AdaptiveRandomForestRegressor
from river import metrics
# from storage.sequence_storage import SequenceStorage
# from storage.interval_storage import IntervalStorage
# from storage.uniform_reservoir_storage import UniformReservoirStorage
# from storage.geometric_reservoir_storage import GeometricReservoirStorage
from storage.batch_storage import BatchStorage
from imputer.marginal_imputer import MarginalImputer
# from imputer.default_imputer import DefaultImputer
from explainer.incremental_pfi import IncrementalPFI
# from utils.converters import RiverToPredictionFunction


storage = BatchStorage(store_targets=True)
imputer = MarginalImputer(sampling_strategy='product')
# imputer = DefaultImputer(values={'N_1': 5, 'N_2': 3, 'N_10': 2})


def _model_function_dict(x_dict):
    x_list = list(x_dict.values())
    return _dataset_target_function(x_list)


if __name__ == "__main__":

    _N_FEATURES = 10
    _WEIGHTS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 10]

    def _dataset_target_function(x):
        return np.dot(x, _WEIGHTS)

    # def _model_function_dict(x_dict):
    #     x_list = list(x_dict.values())
    #     return _dataset_target_function(x_list)

    TRAIN_TIME = 1000
    TEST_TIME = 1002

    n_samples = 10000
    dataset = SyntheticDataset(n_numeric=_N_FEATURES, n_features=_N_FEATURES,
                               n_samples=n_samples, noise_std=0.01,
                               target_function=_dataset_target_function)
    stream = dataset.to_stream()
    feature_names = dataset.feature_names
    model = AdaptiveRandomForestRegressor()

    metric = metrics.MAE()
    feature_list = ['N_1', 'N_2', 'N_10']
    values = []
    for (n, (x_i, y_i)) in enumerate(stream):
        n += 1
        y_i_pred = model.predict_one(x_i)
        storage.update(x_i, y_i)
        model.learn_one(x_i, y_i)
        metric.update(y_true=y_i, y_pred=y_i_pred)
        if n % 1000 == 0:
            print(n, "Score", metric.get())
        if n >= TRAIN_TIME:
            break

    all_data = storage.get_data()
    print(len(all_data[0]))
    print(len(all_data[1]))
    explainer = IncrementalPFI(model, feature_list, storage, imputer, 'mse')

    for (n, (x_i, y_i)) in enumerate(stream, (TRAIN_TIME + 1)):
        n += 1
        y_i_pred = model.predict_one(x_i)
        model.learn_one(x_i, y_i)
        inc_pfi_values = explainer.explain_one(x_i, y_i)
        print(inc_pfi_values)
        metric.update(y_true=y_i, y_pred=y_i_pred)
        if n % 10 == 0:
            print(n, "Score", metric.get())
        if n >= TEST_TIME:
            break