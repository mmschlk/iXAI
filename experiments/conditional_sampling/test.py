import numpy as np
from matplotlib import pyplot as plt
from river import metrics
from river.utils import Rolling

from data.stream.synth import Agrawal
from data.stream.synth._base import BaseSyntheticDataset
from experiments.setup.loss import get_loss_function
from increment_explain.imputer import MarginalImputer
from increment_explain.imputer.tree_imputer import TreeImputer
from increment_explain.storage import UniformReservoirStorage, GeometricReservoirStorage
from increment_explain.storage.tree_storage import TreeStorage
from increment_explain.explainer import IncrementalPFI
from river.datasets.base import REG

from scipy.stats import multivariate_normal as mvn

def model_function_placeholder():
    pass


def get_covariance_stream(size):
    cov = np.array([[1, 0.5,.2, .1],[.5,1.,.1,.1],[0.2,.1,1,.3],[0.1,.1,.3,1]])
    x_data = mvn.rvs(mean=[0.,0.,0.,0.], cov=cov, size=size)
    feature_names = ['_'.join(('N', i)) for i in range(1, 5)]
    y_data = np.array([1 for _ in range(size)])
    dataset = BaseSyntheticDataset(feature_names=feature_names, x_data=x_data, y_data=y_data)
    return dataset.to_stream()


if __name__ == "__main__":

    RANDOM_SEED = 1
    CLASSIFICATION_FUNCTIONS = (1, 2)

    N_STREAM_1 = 10000

    for i in range(1):

        # Setup Data ---------------------------------------------------------------------------------------------------

        dataset_1 = Agrawal(classification_function=CLASSIFICATION_FUNCTIONS[0], random_seed=RANDOM_SEED, n_samples=N_STREAM_1)
        n_samples = N_STREAM_1
        stream_1 = dataset_1.stream
        feature_names = dataset_1.feature_names
        cat_feature_names = dataset_1.cat_feature_names
        num_feature_names = dataset_1.num_feature_names

        loss_function = get_loss_function(task=REG)

        # Model and training setup
        model_function = model_function_placeholder

        # Instantiating objects
        storage = TreeStorage(cat_feature_names=cat_feature_names, num_feature_names=num_feature_names)
        imputer = TreeImputer(model_function, storage_object=storage, direct_predict_numeric=False, use_storage=True)

        res_storage = GeometricReservoirStorage(store_targets=False, size=100, constant_probability=1.0)
        imputer_marginal = MarginalImputer(model_function, storage_object=res_storage, sampling_strategy='joint')

        age_perf = []
        commission_perf = []

        salaray_values = []
        commission_values = []
        commission_predictions = []
        commission_marginal_predictions = []

        age_predictions = []
        age_marginal_predictions = []

        # Concept 1 ----------------------------------------------------------------------------------------------------
        for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
            storage.update(x_i, y_i)
            res_storage.update(x_i, y_i)

            salaray_values.append(x_i['salary'])
            commission_values.append(x_i['commission'])

            age_predictions.append(imputer._sample_from_storages(feature_name='age', x_i=x_i))
            age_marginal_predictions.append(imputer_marginal._sample(res_storage, feature_subset=['age'])['age'])

            commission_predictions.append(imputer._sample_from_storages(feature_name='commission', x_i=x_i))
            commission_marginal_predictions.append(imputer_marginal._sample(res_storage, feature_subset=['commission'])['commission'])

            age_perf.append(storage.performances['age'].get())
            commission_perf.append(storage.performances['commission'].get())

            if n % 1000 == 0:
                print(f"{n}")
            if n >= n_samples:
                break

        plt.plot(age_perf)
        plt.title('Performance age')
        plt.ylim((0, 1.1))
        plt.show()

        plt.plot(commission_perf)
        plt.title('Performance commission')
        plt.ylim((0, 1.1))
        plt.show()

        plt.scatter(salaray_values[200:], commission_values[200:])
        plt.title('agrawal stream (original data distribution)')
        plt.ylim((-30_000, 125_000))
        plt.xlabel("salary")
        plt.ylabel("commission")
        plt.show()

        plt.scatter(salaray_values[200:], commission_predictions[200:])
        plt.title('agrawal stream (conditional sampling)')
        plt.ylim((-30_000, 125_000))
        plt.xlabel("salary")
        plt.ylabel("commission")
        plt.show()

        plt.scatter(salaray_values[200:], commission_marginal_predictions[200:])
        plt.title('agrawal stream (marginal sampling)')
        plt.ylim((-30_000, 125_000))
        plt.xlabel("salary")
        plt.ylabel("commission")
        plt.show()

        plt.scatter(salaray_values[200:], age_predictions[200:])
        plt.title('agrawal stream (conditional sampling)')
        plt.ylim((-0, 100))
        plt.xlabel("salary")
        plt.ylabel("age")
        plt.show()

        plt.scatter(salaray_values[200:], age_marginal_predictions[200:])
        plt.title('agrawal stream (marginal sampling)')
        plt.ylim((-0, 100))
        plt.xlabel("salary")
        plt.ylabel("age")
        plt.show()

