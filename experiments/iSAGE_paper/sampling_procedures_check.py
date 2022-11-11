import random

import numpy as np
import pandas as pd
import river
from matplotlib import pyplot as plt

from data.stream.synth import Agrawal
from experiments.setup.loss import get_loss_function
from increment_explain.imputer import MarginalImputer
from increment_explain.imputer.tree_imputer import TreeImputer
from increment_explain.storage import GeometricReservoirStorage
from increment_explain.storage.tree_storage import TreeStorage
from river.datasets.base import REG

from scipy.stats import multivariate_normal as mvn


def model_function_placeholder(x_i):
    return 1.0


def get_covariance_stream(size):

    mvn.rvs(mean=[0., 0.], cov=[[1, 0.8], [0.8, 1]], size=5000)

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

    # Model and training setup
    model_function = model_function_placeholder
    loss_function = get_loss_function(task=REG)

    sample_from_leaf_reservoirs = True  # let the model predict (False) or sample from leaf reservoirs (True)

    n_samples_agrawal = 5000
    n_samples_covariance_test = 5000

    # agrawal Test -----------------------------------------------------------------------------------------------------
    dataset_1 = Agrawal(classification_function=0, random_seed=RANDOM_SEED, n_samples=n_samples_agrawal)
    n_samples = n_samples_agrawal
    stream_1 = dataset_1.stream
    feature_names = dataset_1.feature_names
    cat_feature_names = dataset_1.cat_feature_names
    num_feature_names = dataset_1.num_feature_names

    # init imputer and storage
    storage_tree = TreeStorage(cat_feature_names=cat_feature_names, num_feature_names=num_feature_names)
    imputer_cond = TreeImputer(model_function, storage_object=storage_tree, direct_predict_numeric=True, use_storage=True)
    storage_reservoir = GeometricReservoirStorage(store_targets=False, size=100, constant_probability=1.0)
    imputer_marg = MarginalImputer(model_function, storage_object=storage_reservoir, sampling_strategy='joint')

    x_data_agrawal = []

    commission_sampled_cond = []
    commission_sampled_marg = []

    age_sampled_cond = []
    age_sampled_marg = []

    performances_agrawal = []

    for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
        storage_tree.update(x_i, y_i)
        storage_reservoir.update(x_i, y_i)

        x_data_agrawal.append(x_i)

        if sample_from_leaf_reservoirs:
            commission_sampled_cond.append(imputer_cond._sample_from_storages(feature_name='commission', x_i=x_i))
            age_sampled_cond.append(imputer_cond._sample_from_storages(feature_name='age', x_i=x_i))
        else:
            commission_sampled_cond.append(imputer_cond._sample(feature_name='commission', x_i=x_i))
            age_sampled_cond.append(imputer_cond._sample(feature_name='age', x_i=x_i))

        commission_sampled_marg.append(imputer_marg._sample(storage_reservoir, feature_subset=['commission'])['commission'])
        age_sampled_marg.append(imputer_marg._sample(storage_reservoir, feature_subset=['age'])['age'])

        performance = {}
        for feature_name, model in storage_tree._storage_x.items():
            performance[feature_name] = storage_tree.performances[feature_name].get()
        performances_agrawal.append(performance)

        if n >= n_samples:
            break

    # Covariance Test --------------------------------------------------------------------------------------------------
    stream_1, feature_names = get_covariance_stream(size=n_samples_covariance_test)
    n_samples = n_samples_covariance_test
    feature_names = feature_names
    cat_feature_names = ['C_1']
    num_feature_names = ['N_1', 'N_2', 'N_3', 'N_4']

    # init imputer
    storage_tree = TreeStorage(cat_feature_names=cat_feature_names, num_feature_names=num_feature_names)
    imputer_cond = TreeImputer(model_function, storage_object=storage_tree, direct_predict_numeric=True, use_storage=True)
    storage_reservoir = GeometricReservoirStorage(store_targets=False, size=100, constant_probability=1.0)
    imputer_marg = MarginalImputer(model_function, storage_object=storage_reservoir, sampling_strategy='joint')

    x_data_covariance_test = []

    C_1_sampled_cond = []
    C_1_sampled_marg = []

    N_1_sampled_cond = []
    N_1_sampled_marg = []

    performances_covariance_test = []

    # Concept 1 ----------------------------------------------------------------------------------------------------
    for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
        storage_tree.update(x_i, y_i)
        storage_reservoir.update(x_i, y_i)

        x_data_covariance_test.append(x_i)

        if sample_from_leaf_reservoirs:
            C_1_sampled_cond.append(imputer_cond._sample_from_storages(feature_name='C_1', x_i=x_i))
            N_1_sampled_cond.append(imputer_cond._sample_from_storages(feature_name='N_1', x_i=x_i))
        else:
            C_1_sampled_cond.append(imputer_cond._sample(feature_name='C_1', x_i=x_i))
            N_1_sampled_cond.append(imputer_cond._sample(feature_name='N_1', x_i=x_i))

        C_1_sampled_marg.append(imputer_marg._sample(storage_reservoir, feature_subset=['C_1'])['C_1'])
        N_1_sampled_marg.append(imputer_marg._sample(storage_reservoir, feature_subset=['N_1'])['N_1'])

        performance = {}
        for feature_name, model in storage_tree._storage_x.items():
            performance[feature_name] = storage_tree.performances[feature_name].get()
        performances_covariance_test.append(performance)

        if n >= n_samples:
            break

    # plotting ---------------------------------------------------------------------------------------------------------
    fig_size_vertical = 3
    fig_size_horizontal = 9

    # plot agrawal -----------------------------------------------------------------------------------------------------
    markevery_step = 30
    markevery_ground_truth_step = 15

    alpha_hue = np.asarray([i for i in range(1, n_samples + 1)]) / n_samples
    alpha_hue = alpha_hue[::markevery_step]

    performances_agrawal = pd.DataFrame(performances_agrawal)
    x_data_agrawal = pd.DataFrame(x_data_agrawal)

    fig, axis = plt.subplots(figsize=(fig_size_horizontal*2/3, fig_size_vertical), nrows=1, ncols=2, sharey="all")
    axis_1 = axis[0]
    axis_2 = axis[1]

    axis_1.scatter(
        x_data_agrawal['commission'].loc[::markevery_ground_truth_step],
        x_data_agrawal['salary'].loc[::markevery_ground_truth_step],
        marker="o", c="black", alpha=alpha_hue
    )
    axis_1.scatter(
        commission_sampled_marg[::markevery_step],
        x_data_agrawal['salary'].loc[::markevery_step],
        marker="+", c="blue", alpha=alpha_hue
    )
    axis_1.scatter(
        commission_sampled_cond[::markevery_step],
        x_data_agrawal['salary'].loc[::markevery_step],
        marker="x", c="red", alpha=alpha_hue
    )
    axis_1.axis(xmin=-5_000, xmax=90_000, ymin=15_000, ymax=155_000)
    axis_1.set_xlabel(r"$X_{commission}$")
    axis_1.set_ylabel(r"$X_{salary}$")
    axis_1.set_yticks([25_000, 50_000, 75_000, 100_000, 125_000, 150_000])
    axis_1.set_yticklabels(["25k", "50k", "75k", "100k", "125k", "150k"])
    axis_1.set_xticks([0, 20_000, 40_000, 60_000, 80_000])
    axis_1.set_xticklabels(["0", "20k", "40k", "60k", "80k"])
    axis_1.set_title(r"dependent")

    axis_2.scatter(
        x_data_agrawal['age'].loc[::markevery_ground_truth_step],
        x_data_agrawal['salary'].loc[::markevery_ground_truth_step],
        marker="o", c="black", alpha=alpha_hue
    )
    axis_2.scatter(
        age_sampled_marg[::markevery_step],
        x_data_agrawal['salary'].loc[::markevery_step],
        marker="+", c="blue", alpha=alpha_hue
    )
    axis_2.scatter(
        age_sampled_cond[::markevery_step],
        x_data_agrawal['salary'].loc[::markevery_step],
        marker="x", c="red", alpha=alpha_hue
    )
    axis_2.axis(xmin=5, xmax=85)
    axis_2.set_xlabel(r"$X_{age}$")
    axis_2.set_xlim([15, 85])
    axis_2.set_xticks([20, 30, 40, 50, 60, 70, 80])
    axis_2.set_title(r"independent")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0, left=0.12, right=0.995)
    plt.savefig("agrawal_conditonal_vs_marginal.png", dpi=200)
    plt.show()

    # plot covariance test ---------------------------------------------------------------------------------------------
    performances_covariance_test = pd.DataFrame(performances_covariance_test)
    x_data_covariance_test = pd.DataFrame(x_data_covariance_test)

    markevery_step = 30
    markevery_ground_truth_step = 15

    alpha_hue = np.asarray([i for i in range(1, n_samples + 1)]) / n_samples
    alpha_hue = alpha_hue[::markevery_step]

    fig_2, axes_2 = plt.subplots(figsize=(fig_size_horizontal*1/3, fig_size_vertical), nrows=1, ncols=1)
    axis_3 = axes_2

    axis_3.scatter(
        x_data_covariance_test['N_1'].loc[::markevery_ground_truth_step],
        x_data_covariance_test['N_2'].loc[::markevery_ground_truth_step],
        marker="o", c="black", alpha=alpha_hue
    )
    axis_3.scatter(
        N_1_sampled_marg[::markevery_step],
        x_data_covariance_test['N_2'].loc[::markevery_step],
        marker="+", c="blue", alpha=alpha_hue
    )
    axis_3.scatter(
        N_1_sampled_cond[::markevery_step],
        x_data_covariance_test['N_2'].loc[::markevery_step],
        marker="x", c="red", alpha=alpha_hue
    )
    axis_3.axis(xmin=-3.8, xmax=3.8, ymin=-3.8, ymax=3.8)
    axis_3.set_xlabel(r"$X_{2}$")
    axis_3.set_ylabel(r"$X_{1}$")
    axis_3.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    axis_3.set_yticks([-3, -2, -1, 0, 1, 2, 3])

    plt.title(r"$\rho_{X_{1}, X_{2}} = 0.8$")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0, left=0.18, right=0.99)
    plt.savefig("covariance_conditonal_vs_marginal.png", dpi=200)
    plt.show()
