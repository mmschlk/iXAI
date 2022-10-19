"""
This module gathers SHAP Explanation Methods
"""

# Authors: Maximilian Muschalik <maximilian.muschalik@lmu.de>
#          Fabian Fumagalli <ffumagalli@techfak.uni-bielefeld.de>

import copy
import random
from abc import ABCMeta
from abc import abstractmethod
from typing import Optional, Iterator, Any, Sequence, Callable
from collections import Counter
import itertools

import numpy as np
from scipy.special import binom

from increment_explain.utils import get_n_feature_masks, get_all_feature_masks

__all__ = [
    "KernelSHAP",
    "ImprovedSHAP",
    "IncrementalSHAP"
]

# =============================================================================
# Types and constants
# =============================================================================

EPS = 1e-10


# =============================================================================
# Base Explainer Class
# =============================================================================


class BaseSHAPExplainer(metaclass=ABCMeta):
    """Base class for SHAP-based explainer algorithms.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
            self,
            *,
            model: Callable,
            feature_names: list[str],
            random_state: Optional[int] = None
    ):
        self.model_fn = model
        self.feature_names = feature_names
        self.random_state = random_state
        self.n_features = len(feature_names)
        self.paired_subsets, self.unpaired_subset = self._get_paired_subsets(self.n_features)
        self.n_subsets = self.n_features - 1
        self.sampling_weight_vector = np.array(
            [(self.n_features - 1.0) / (i * (self.n_features - i)) for i in range(1, self.n_subsets + 1)])
        self.sampling_weight_vector /= np.sum(self.sampling_weight_vector)
        self.sampling_weight_dict = \
            {subset_size: self.sampling_weight_vector[subset_size - 1] for subset_size in range(1, self.n_features)}

    @abstractmethod
    def explain_one(self, x, reference_set):
        pass

    @staticmethod
    def _shapley_kernel(m: int, s: int) -> float:
        if s == 0 or s == m:
            return 10000.
        return 1 / ((m - 1) * binom(m - 2, s - 1))

    @staticmethod
    def _shapley_weights(m: int, s: int) -> float:
        if s > m - 1 or s < 0:
            return 0
        else:
            return 1 / (binom(m - 1, s) * m)

    @staticmethod
    def _powerset(iterable: Iterator[Any]) -> Iterator[Any]:
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

    @staticmethod
    def _get_paired_subsets(n_features: int) -> tuple[list[tuple[int, int]], Optional[int]]:
        subset_sizes = list(range(1, n_features))
        n_paired_subsets = int(len(subset_sizes) / 2)
        paired_subsets = [(subset_sizes[subset_size - 1], subset_sizes[-subset_size])
                          for subset_size in range(1, n_paired_subsets + 1)]
        unpaired_subset = None
        if n_paired_subsets < len(subset_sizes) / 2:
            unpaired_subset = int(np.ceil(len(subset_sizes) / 2))
        return paired_subsets, unpaired_subset

    def _determine_complete_subsets(self, budget: int) -> tuple[list[int], list[int], int]:
        complete_subsets = []
        incomplete_subsets = list(range(1, self.n_subsets + 1))
        weight_vector = copy.copy(self.sampling_weight_vector)
        allowed_budget = weight_vector * budget
        for subset_size_1, subset_size_2 in self.paired_subsets:
            subset_budget = int(binom(self.n_features, subset_size_1))
            if allowed_budget[subset_size_1 - 1] >= subset_budget:
                complete_subsets.extend((subset_size_1, subset_size_2))
                incomplete_subsets.remove(subset_size_1)
                incomplete_subsets.remove(subset_size_2)
                weight_vector[subset_size_1 - 1] = 0
                weight_vector[subset_size_2 - 1] = 0
                weight_vector /= np.sum(weight_vector)
                budget -= subset_budget * 2
            else:
                return complete_subsets, incomplete_subsets, budget
            allowed_budget = weight_vector * budget
        if self.unpaired_subset is not None:
            subset_budget = int(binom(self.n_features, self.unpaired_subset))
            if budget - subset_budget >= 0:
                complete_subsets.append(self.unpaired_subset)
                incomplete_subsets.remove(self.unpaired_subset)
                budget -= subset_budget
        return complete_subsets, incomplete_subsets, budget

    def _sample_from_weights(self,
                             budget: int,
                             subset_sizes: list[int],
                             *,
                             random_seed: Optional[int] = None
                             ) -> dict[int, set[tuple[int]]]:
        if not subset_sizes:
            return {}
        random.seed(random_seed)
        subset_weight_vector = np.asarray(
            [self.sampling_weight_vector[subset_size - 1] for subset_size in subset_sizes])
        subset_weight_vector /= np.sum(subset_weight_vector)
        subset_sizes_samples = random.choices(subset_sizes, k=budget, weights=subset_weight_vector)
        subset_sizes_samples = Counter(subset_sizes_samples)
        masks = {}
        for subset_size, n_samples in subset_sizes_samples.items():
            masks[subset_size] = get_n_feature_masks(subset_size, self.n_features, n=n_samples)
        return masks

    def _sample_complete_subsets(self, subset_sizes: list[int]) -> dict[int, set[tuple[int]]]:
        masks = {}
        for subset_size in subset_sizes:
            masks[subset_size] = get_all_feature_masks(subset_size, self.n_features)
        return masks

    def _sample_masks(self,
                      budget: int = 1024,
                      *,
                      random_seed: Optional[int] = None
                      ) -> np.ndarray:
        all_masks = {}
        complete_subsets, incomplete_subsets, budget = self._determine_complete_subsets(budget)
        all_masks.update(self._sample_complete_subsets(complete_subsets))
        all_masks.update(self._sample_from_weights(budget, incomplete_subsets, random_seed=random_seed))
        all_masks = np.asarray([mask for key in all_masks.keys() for mask in all_masks[key]])
        return all_masks

    @staticmethod
    def _generate_samples(masks: np.ndarray,
                          x_orig: Any,
                          reference_set: Sequence[Any]
                          ) -> np.ndarray:
        n_masks = len(masks)
        if hasattr(x_orig, "values"):
            x_orig = x_orig.values
        x_orig = np.repeat([x_orig], repeats=n_masks, axis=0)
        reference_set = np.asarray(reference_set)
        if len(reference_set) > 1:
            permuted_samples = np.asarray(random.choices(reference_set, k=n_masks))
        else:
            permuted_samples = np.repeat(reference_set, repeats=n_masks, axis=0)
        masks = masks.astype(bool)
        permuted_samples[masks] = x_orig[masks]
        return permuted_samples

    def sample_with_budget(self,
                           budget: int,
                           x_orig: np.ndarray,
                           reference_set: Sequence[np.ndarray],
                           random_seed: Optional[int] = None
                           ) -> tuple[np.ndarray, np.ndarray]:
        masks = self._sample_masks(budget, random_seed=random_seed)
        permuted_samples = self._generate_samples(masks=masks, x_orig=x_orig, reference_set=reference_set)
        return permuted_samples, masks

    @staticmethod
    def _get_empty_sample(reference_set: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray(random.choice(reference_set))

    def _predict_with_samples(self, samples):
        if hasattr(samples, "shape"):
            if len(samples.shape) < 2:
                try:
                    samples = samples.reshape((1, samples.shape[0]))
                except AttributeError:
                    pass
        try:
            predictions = self.model_fn(samples)
        except Exception as error:
            raise error
        return predictions


# =============================================================================
# Public Explainer Classes
# =============================================================================


class KernelSHAP(BaseSHAPExplainer):

    def __init__(
            self,
            *,
            model: object,
            feature_names: list[str],
            random_state: Optional[int] = None
    ):
        super(KernelSHAP, self).__init__(
            model=model,
            feature_names=feature_names,
            random_state=random_state
        )

    def explain_one(self,
                    x,
                    reference_set: Sequence[Any],
                    budget: int = 1024,
                    random_seed: Optional[int] = None
                    ) -> Any:
        # TODO refactor into base class and methods
        # masks = S_shap
        empty_sample = self._get_empty_sample(reference_set)
        permuted_samples, masks = self.sample_with_budget(budget=budget,
                                                          x_orig=x,
                                                          reference_set=reference_set,
                                                          random_seed=random_seed)

        y_full = self._predict_with_samples(x)
        y_empty = self._predict_with_samples(empty_sample)
        y_permuted = self._predict_with_samples(permuted_samples)

        subset_sizes = np.sum(masks, axis=1)  # = S_sizes
        subset_weights = [self.sampling_weight_dict[subset_size] for subset_size in subset_sizes]  # = weights_shap

        empty_mask = np.zeros(shape=self.n_features, dtype=int)
        full_mask = np.ones(shape=self.n_features, dtype=int)

        tmp = np.concatenate(([empty_mask], masks, [full_mask]))
        S_shap_bar = np.zeros((np.shape(tmp)[0], self.n_features + 1))
        S_shap_bar[:, -1] = 1
        S_shap_bar[:, :-1] = tmp
        weights_shap_bar = np.concatenate(([10000], subset_weights, [10000]))
        y_all = np.concatenate((y_empty, y_permuted, y_full))
        y_all = np.reshape(y_all, (y_all.shape[0], 1))

        tmp = np.linalg.inv(np.dot(np.dot(S_shap_bar.T, np.diag(weights_shap_bar)), S_shap_bar))
        shap_values = np.dot(tmp, np.dot(np.dot(S_shap_bar.T, np.diag(weights_shap_bar)), y_all))

        return shap_values


class ImprovedSHAP(BaseSHAPExplainer):

    def __init__(
            self, *,
            model,
            feature_names,
            random_state
    ):
        super(ImprovedSHAP, self).__init__(
            model=model,
            feature_names=feature_names,
            random_state=random_state
        )

    def explain_one(self, x, reference_set):
        pass


class IncrementalSHAP(BaseSHAPExplainer):

    def __init__(
            self, *,
            model,
            feature_names,
            random_state
    ):
        super(IncrementalSHAP, self).__init__(
            model=model,
            feature_names=feature_names,
            random_state=random_state
        )

    def explain_one(self, x, reference_set):
        pass


class DummyModel:
    def __init__(self, variant='regression'):
        self.variant = variant

    def predict(self, x_predict):
        if self.variant == 'regression':
            return [sum(x_i) for x_i in x_predict]
        else:
            return [sum(x_i) / len(x_i) for x_i in x_predict]


if __name__ == "__main__":
    pass
