"""
This file gathers Tests for the Shap Implementations
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>
import time

import numpy as np
import random
from pytest import approx

import shap
from explainer.shap import KernelSHAP


# =============================================================================
# Utility Classes
# =============================================================================


class DummyModel:
    def __init__(self, n_features=5, variant='regression', feature_weights=None, randomness=False):
        self.variant = variant
        self.feature_weights = feature_weights if feature_weights is not None else [1 for _ in range(n_features)]
        self.randomness = int(randomness)

    def predict(self, x_predict):
        predictions = []
        if self.variant == 'regression':
            for x_i in x_predict:
                prediction = np.dot(x_i, self.feature_weights) + random.random() * 0.1 * self.randomness
                predictions.append(prediction)
        else:
            for x_i in x_predict:
                prediction = int(np.dot(x_i, self.feature_weights) > 0.5) * 0.1 * self.randomness
                predictions.append(prediction)
        return np.asarray(predictions)


# =============================================================================
# Test Cases
# =============================================================================


class TestKernelSHAP:
    def test_determine_complete_subsets(self):
        np.random.seed(42)

        n_features = 10
        feature_names = [str(i) for i in range(n_features)]

        sampling_budget = 900

        kernel_explainer = KernelSHAP(model=None, feature_names=feature_names, random_state=None)
        complete_subsets, incomplete_subsets, budget = \
            kernel_explainer._determine_complete_subsets(budget=sampling_budget)

        assert set(complete_subsets) == {1, 9, 2, 8, 3, 7}
        assert set(incomplete_subsets) == {4, 5, 6}
        assert budget == 550

        sampling_budget = 1022

        kernel_explainer = KernelSHAP(model=None, feature_names=feature_names, random_state=None)
        complete_subsets, incomplete_subsets, budget = \
            kernel_explainer._determine_complete_subsets(budget=sampling_budget)

        assert set(complete_subsets) == {1, 2, 3, 4, 5, 6, 7, 8, 9}
        assert set(incomplete_subsets) == set([])
        assert budget == 0

    def test_sample_with_budget(self):
        np.random.seed(42)

        n_features = 20
        feature_names = [str(i) for i in range(n_features)]
        x_explain = np.asarray([0.25 for _ in range(n_features)])
        x_reference = np.asarray([[0.5 for _ in range(n_features)]])
        sampling_budget = 1024

        kernel_explainer = KernelSHAP(model=None, feature_names=feature_names, random_state=None)
        permuted_samples, masks = kernel_explainer.sample_with_budget(x_orig=x_explain,
                                                                      reference_set=x_reference,
                                                                      budget=sampling_budget,
                                                                      random_seed=42)
        n_sampled = len(permuted_samples)
        assert n_sampled == sampling_budget

    def test_explain_one_equal_contribution(self):
        np.random.seed(42)

        n_features = 20
        feature_names = [str(i) for i in range(n_features)]
        x_explain = np.asarray([0.25 for _ in range(n_features)])
        x_reference = np.asarray([[0.5 for _ in range(n_features)]])

        model = DummyModel(n_features=n_features, variant='regression')

        kernel_explainer = KernelSHAP(model=model.predict, feature_names=feature_names, random_state=None)
        explanation = kernel_explainer.explain_one(x_explain, reference_set=x_reference, budget=1024, random_seed=42)

        shap_values = explanation[0:n_features]
        shap_values = shap_values.flatten()
        shap_values = np.round(shap_values, 2)

        theoretical_shap_values = np.asarray([-0.25 for _ in range(n_features)])

        assert np.array_equal(np.round(shap_values, 2), theoretical_shap_values)
        assert approx(explanation[-1][0]) == 10

    def test_explain_one_non_equal_contribution(self):
        np.random.seed(42)

        n_features = 5
        feature_names = [str(i) for i in range(n_features)]
        x_explain = np.asarray([0.25, 0.25, 0.25, 0.25, 0.25])
        x_reference = np.asarray([[0.5 for _ in range(n_features)]])
        feature_weights = [10, 1, 1, 1, 1]

        model = DummyModel(n_features=n_features, variant='regression', feature_weights=feature_weights)

        kernel_explainer = KernelSHAP(model=model.predict, feature_names=feature_names, random_state=None)
        explanation = kernel_explainer.explain_one(x_explain, reference_set=x_reference, budget=1024,
                                                   random_seed=42)

        shap_values = explanation[0:n_features]
        shap_values = shap_values.flatten()
        shap_values = np.round(shap_values, 2)

        theoretical_shap_values = np.asarray([-2.5, -0.25, -0.25, -0.25, -0.25])

        assert np.array_equal(shap_values, theoretical_shap_values)
        assert approx(explanation[-1][0]) == 7

    def test_original_shap_vs_reimplementation(self):
        np.random.seed(42)
        random.seed(42)

        x_explain = np.asarray([0.71, 0.82, 0.91, 0.64, 0.75, 0.5, 0.5, 0.5, 0.30, 0.10, 0.71, 0.82, 0.91, 0.64, 0.75, 0.5, 0.5, 0.5, 0.30, 0.10])

        n_features = len(x_explain)
        feature_names = [str(i) for i in range(n_features)]
        x_reference = np.asarray([[0.5 for _ in range(n_features)]])
        feature_weights = [1 + i for i in range(n_features)]

        model = DummyModel(n_features=n_features, variant='regression', feature_weights=feature_weights, randomness=True)

        # Incomplete Sampling (budget too small)
        kernel_explainer = KernelSHAP(model=model.predict, feature_names=feature_names, random_state=None)
        explanation = kernel_explainer.explain_one(x_explain, reference_set=x_reference, budget=100,
                                                   random_seed=42)
        shap_values = explanation[0:n_features].flatten()
        original_shap = shap.KernelExplainer(model=model.predict, data=x_reference)
        original_shap_values = original_shap.shap_values(X=x_explain, nsamples=100)

        assert approx(sum(shap_values), abs=0.2) == sum(original_shap_values)

        # Complete Sampling (budget large enough to compute exact)
        start_time = time.time()
        explanation = kernel_explainer.explain_one(x_explain, reference_set=x_reference, budget=2000,
                                                   random_seed=42)
        reimplementation_runtime = time.time() - start_time
        shap_values = explanation[0:n_features].flatten()

        start_time = time.time()
        original_shap_values = original_shap.shap_values(X=x_explain, nsamples=2000)
        original_runtime = time.time() - start_time

        assert approx(sum(shap_values), abs=0.2) == sum(original_shap_values)
