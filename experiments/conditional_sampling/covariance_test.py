import copy
import random

import numpy as np
import pandas as pd
import river
from matplotlib import pyplot as plt
from river import metrics
from river.utils import Rolling

from data.stream.synth import Agrawal
from data.stream.synth._base import BaseSyntheticDataset
from experiments.setup.loss import get_loss_function
from increment_explain.imputer import MarginalImputer
from increment_explain.imputer.tree_imputer import TreeImputer
from increment_explain.storage import UniformReservoirStorage
from increment_explain.storage.tree_storage import TreeStorage
from increment_explain.explainer import IncrementalPFI
from river.datasets.base import REG

from scipy.stats import multivariate_normal as mvn

def model_function_placeholder(x_i):
    return 1.0


def get_covariance_stream(size):
    cov = np.array([[1,0.8,0,0],[0.8,1,0,0],[0,0,1,0],[0,0,0,1]])
    x_data = mvn.rvs(mean=[0.,0.,0.,0.], cov=cov, size=size)
    feature_names = ['_'.join(('N', str(i))) for i in range(1, 5)]
    x_data = pd.DataFrame(x_data, columns=feature_names)
    x_data['C_1'] = np.array([random.randint(1, 5) for _ in range(size)])
    y_data = pd.DataFrame(np.array([1 for _ in range(size)]))
    stream = river.stream.iter_pandas(x_data, y_data, feature_names=feature_names)
    return stream, feature_names


if __name__ == "__main__":

    RANDOM_SEED = 1
    N_STREAM_1 = 10000

    for i in range(1):

        # Setup Data ---------------------------------------------------------------------------------------------------

        stream_1, feature_names = get_covariance_stream(size=N_STREAM_1)
        n_samples = N_STREAM_1
        feature_names = feature_names
        cat_feature_names = ['C_1']
        num_feature_names = ['N_1', 'N_2', 'N_3', 'N_4']

        loss_function = get_loss_function(task=REG)

        # Model and training setup
        model_function = model_function_placeholder

        # Instantiating objects
        storage = TreeStorage(cat_feature_names=cat_feature_names, num_feature_names=num_feature_names)
        # storage = SequenceStorage(store_targets=True)
        imputer = TreeImputer(model_function, storage_object=storage, direct_predict_numeric=True, use_storage=True)
        # imputer = DefaultImputer(model, values={'N_1': 5, 'N_2': 3, 'N_10': 2})



        x_data = []
        C_1_sampled = []
        N_1_sampled = []
        performances = []

        # Concept 1 ----------------------------------------------------------------------------------------------------
        for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
            #print(x_i)
            storage.update(x_i, y_i)
            x_data.append(x_i)

            C_1_sampled.append(imputer._sample_from_storages(feature_name='C_1', x_i=x_i))
            N_1_sampled.append(imputer._sample_from_storages(feature_name='N_1', x_i=x_i))

            performance = {}
            for feature_name, model in storage._storage_x.items():
                performance[feature_name] = storage.performances[feature_name].get()
            performances.append(performance)
            if n % 1000 == 0:
                print(f"{n}")
            if n >= n_samples:
                break

        x_data = pd.DataFrame(x_data)
        plt.scatter(x_data['N_2'], x_data['N_1'])
        plt.title('N_2 to N_1')
        plt.show()

        plt.scatter(x_data['N_4'], x_data['N_1'])
        plt.title('N_4 to N_1')
        plt.show()

        performances = pd.DataFrame(performances)

        plt.plot(performances['N_1'])
        plt.title('Performance N_1')
        plt.ylim((0, 1.1))
        plt.show()

        plt.plot(performances['N_4'])
        plt.title('Performance N_4')
        plt.ylim((0, 1.1))
        plt.show()

        rf_1 = storage._storage_x['N_1']
        rf_2 = storage._storage_x['N_2']
        rf_3 = storage._storage_x['N_3']
        rf_4 = storage._storage_x['N_4']

        rf_1.predict_one({'N_2': 4})

        plt.scatter(x_data['N_2'], N_1_sampled)
        plt.title('N_2 to N_1 sampled')
        plt.show()

        plt.scatter(x_data['N_4'], N_1_sampled)
        plt.title('N_4 to N_1 sampled')
        plt.show()
