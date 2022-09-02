from typing import Sequence

import numpy as np
import warnings
from math import factorial
from scipy.special import binom
from sympy.utilities.iterables import multiset_permutations


def _generate_all_permutations_from_sequence(sequence: Sequence[int]) -> set[tuple[int]]:
    permutations = set()
    for permutation in multiset_permutations(sequence):
        permutations.add(tuple(permutation))
    return permutations


def _generate_permutations_from_sequence(sequence: Sequence[int], n: int) -> set[tuple[int]]:
    permutations = set()
    length = 0
    stopper = 0
    while length < n:
        permutation = tuple(np.random.permutation(sequence))
        if permutation not in permutations:
            permutations.add(permutation)
            length += 1
        stopper += 1
        if stopper > n * 5:
            warnings.warn(f"Permutation generation was stopped prematurely. Only {len(permutations)} permutations were"
                          f" sampled instead of n = {n}.")
            break
    return permutations


def get_n_permutations(samples: Sequence[int], n: int) -> set[tuple[int]]:
    k = len(samples)
    if k < 15:
        total_permutations = factorial(k)
        if total_permutations <= n:
            return _generate_all_permutations_from_sequence(sequence=samples)
        if total_permutations <= 2 * n:
            warnings.warn("The number of samples 'n' is half the size of k! The sampling procedure may be slow.")
    permutations = _generate_permutations_from_sequence(samples, n)
    return permutations


def get_n_feature_masks(subset_size: int, n_features: int, n: int) -> set[tuple[int]]:
    initial_mask = list(np.repeat(1, subset_size)) + list(np.repeat(0, n_features - subset_size))
    total_permutations = binom(n_features, subset_size)
    if total_permutations <= n:
        return get_all_feature_masks(subset_size=subset_size, n_features=n_features)
    if total_permutations <= 2 * n:
        warnings.warn("The number of samples 'n' is half the size of binom(n_features, subset_size)! "
                      "The sampling procedure may be slow.")
    masks = _generate_permutations_from_sequence(initial_mask, n)
    return masks


def get_all_feature_masks(subset_size: int, n_features: int) -> set[tuple[int]]:
    initial_mask = list(np.repeat(1, subset_size)) + list(np.repeat(0, n_features - subset_size))
    masks = _generate_all_permutations_from_sequence(sequence=initial_mask)
    return masks


if __name__ == "__main__":
    samples_test = list(np.repeat(1, 2)) + list(np.repeat(0, 20))
    permutations_test = get_n_permutations(samples_test, 2)
    print(permutations_test)

    samples_test = list(np.repeat(1, 2)) + list(np.repeat(0, 3))
    permutations_test = get_n_permutations(samples_test, 100)
    print(permutations_test)

    print(get_n_feature_masks(2, 20, 10))

    all_masks = get_all_feature_masks(3, 10)
    print(len(all_masks), binom(10, 3), all_masks)
