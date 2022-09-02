"""
This file gathers Tests for the Shap Implementations
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>

import time

import numpy as np
import random
from pytest import approx

from data.synthetic import SyntheticDataset


class TestSyntheticDataset:
    # TODO add test for target_function and noise levels

    def test_create_dataset(self):
        n_samples_test = 1000
        n_features_test = 20
        n_numeric_test = 10

        dataset = SyntheticDataset(n_features=n_features_test, n_numeric=n_numeric_test, n_samples=n_samples_test)

        x_data = dataset.x_data
        y_data = dataset.y_data

        assert len(dataset) == n_samples_test
        assert x_data.shape[0] == n_samples_test and len(y_data) == n_samples_test
        assert x_data.shape[1] == n_features_test

    def test_to_stream(self):
        n_samples_test = 1000
        n_features_test = 10
        n_numeric_test = 8

        dataset = SyntheticDataset(n_features=n_features_test, n_numeric=n_numeric_test, n_samples=n_samples_test)
        stream = dataset.to_stream()

        counter = 0
        for i, x_i in enumerate(stream):
            counter += 1
            if i >= n_samples_test - 1:
                break

        assert counter == n_samples_test

    def test_feature_names(self):
        n_samples_test = 1000
        n_features_test = 10
        n_numeric_test = 8

        dataset = SyntheticDataset(n_features=n_features_test, n_numeric=n_numeric_test, n_samples=n_samples_test)

        feature_names = dataset.feature_names

        assert len(feature_names) == n_features_test
        assert feature_names[3] == "N_4"
        assert feature_names[-1] == "C_2"
