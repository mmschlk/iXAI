"""
This module contains the Batch SAGE explainer.
"""

import random
from typing import Callable, Any, Sequence, Union, Dict, Optional, List

import numpy as np
from river.metrics.base import Metric
from tqdm import tqdm

from ixai.explainer.base import _get_mean_model_output
from ixai.imputer import BaseImputer, MarginalImputer
from ixai.storage import BatchStorage
from ixai.storage.base import BaseStorage
from ixai.utils.validators import validate_model_function, validate_loss_function


class BatchSage:
    """Batch SAGE Explainer

    Computes SAGE importance values according to its original definition in
    https://arxiv.org/abs/2004.00668. A Storage is updated with all observations from a stream and
    an explanation is created with access to all of these observations at once. This can be
    computationally challenging for large amounts of observations.

    Args:
        model_function (Callable[[Any], Any]): The Model function to be explained (e.g.
            model.predict_one (river), model.predict_proba (sklearn)).
        loss_function (Union[Metric, Callable[[Any, Dict], float]]): The loss function for which
            the importance values are calculated. This can either be a callable function or a
            predefined river.metric.base.Metric.<br>
            - river.metric.base.Metric: Any Metric implemented in river (e.g.
                river.metrics.CrossEntropy() for classification or river.metrics.MSE() for
                regression).<br>
            - callable function: The loss_function needs to follow the signature of
                loss_function(y_true, y_pred) and handle the output dimensions of the model
                function. Smaller values are interpreted as being better if not overriden with
                `loss_bigger_is_better=True`. `y_pred` is passed as a dict.
        feature_names (Sequence[Union[str, int, float]]): List of feature names to be explained
            for the model.
        storage (BaseStorage, optional): Optional incremental data storage Mechanism.
            Defaults to `GeometricReservoirStorage(size=100)` for dynamic modelling settings
            (`dynamic_setting=True`) and `UniformReservoirStorage(size=100)` in static modelling
            settings (`dynamic_setting=False`).
        imputer (BaseImputer, optional): Incremental imputing strategy to be used. Defaults to
            `MarginalImputer(sampling_strategy='joint')`.
        n_inner_samples (int): Number of model evaluation per feature and explanation step
            (observation). Defaults to 1.

    Attributes:
        feature_names (Sequence[Union[str, int, float]]): The feature names of the dataset.
        n_inner_samples (int): Number of model evaluation per feature and explanation step
            (observation).
    """
    def __init__(
            self,
            model_function: Callable[[Any], Any],
            feature_names: Sequence[Union[str, int, float]],
            loss_function: Union[Metric, Callable[[Any, Dict], float]],
            n_inner_samples: int = 1,
            storage: Optional[BaseStorage] = None,
            imputer: Optional[BaseImputer] = None,
    ):
        self.feature_names = feature_names
        self.n_inner_samples = n_inner_samples
        self._model_function = validate_model_function(model_function)
        self._loss_function = validate_loss_function(loss_function)
        self._storage: BaseStorage = storage
        if self._storage is None:
            self._storage = BatchStorage(store_targets=True)
        self._imputer: BaseImputer = imputer
        if self._imputer is None:
            self._imputer = MarginalImputer(
                sampling_strategy='joint', model_function=self._model_function,
                storage_object=self._storage)
        self.importance_values = {feature_name: 0. for feature_name in self.feature_names}

    def update_storage(self, x_i: dict, y_i: Any):
        """Updates the data storage with one observation (x_i, y_i).

        Args:
            x_i (dict): The input features of the current observation.
            y_i (Any): Target label of the current observation.
        """
        self._storage.update(x=x_i, y=y_i)

    def explain_one(
            self,
            x_i,
            y_i,
            n_inner_samples: Optional[int] = None,
            original_sage: bool = False,
            verbose: bool = True
    ) -> dict:
        """Explain one observation (x_i, y_i) with all data stored.

        Args:
            x_i (dict): The input features of the current observation as a dict of feature names to
                feature values.
            y_i (Any): Target label of the current observation.
            n_inner_samples (int, optional): Number of model evaluation per feature for the current
                explanation step (observation). Defaults to `None`.
            original_sage (bool): Flag indicating if the original definition of SAGE is used
                (`True`) or not (`False`). Defaults to `False`.

        Returns:
            (dict): The current SAGE feature importance scores.
        """
        self._storage.update(x_i, y_i)
        x_data, y_data = self._storage.get_data()
        if original_sage:
            self.explain_many_original(
                x_data=x_data, y_data=y_data, n_inner_samples=n_inner_samples, verbose=verbose)
        else:
            self.explain_many(
                x_data=x_data, y_data=y_data, n_inner_samples=n_inner_samples, verbose=verbose)
        return self.importance_values

    def explain_many(
            self,
            x_data: List[dict],
            y_data: List[Any],
            n_inner_samples: Optional[int] = None,
            verbose: bool = True
    ) -> dict:
        """Explain one observation (x_i, y_i) with all data stored.

        Args:
            x_data (List[dict]): A list of input data to be explained, as dicts mapping from
                feature names to feature values.
            y_data (List[Any]): Target label of the current observation.
            n_inner_samples (int, optional): Number of model evaluation per feature for the current
                explanation step (observation). Defaults to `None`.
            verbose (bool): Flag indicating if the explanation should print to console (`True`) or
                not (`False`).

        Returns:
            (dict): The current SAGE feature importance scores.
        """
        if n_inner_samples is None:
            n_inner_samples = self.n_inner_samples
        sage_values = {feature: 0. for feature in self.feature_names}
        n_data = len(x_data)
        all_predictions = self._model_function(x_data)
        marginal_prediction = _get_mean_model_output(all_predictions)
        for n, (x_i, y_i) in tqdm(enumerate(zip(x_data, y_data), start=1), total=n_data,
                                  disable=not verbose):
            permutation_chain = np.random.permutation(self.feature_names)
            loss_previous = self._loss_function(y_true=y_i, y_prediction=marginal_prediction)
            features_not_in_s = set(self.feature_names)
            for feature in permutation_chain:
                features_not_in_s.remove(feature)
                predictions = self._imputer.impute(
                    feature_subset=features_not_in_s,
                    x_i=x_i,
                    n_samples=n_inner_samples
                )
                y = _get_mean_model_output(predictions)
                feature_loss = self._loss_function(y_true=y_i, y_prediction=y)
                marginal_contribution = loss_previous - feature_loss
                sage_values[feature] += marginal_contribution
                loss_previous = feature_loss
            n_data = n
        self.importance_values = {feature: sage_value / n_data
                                  for feature, sage_value in sage_values.items()}
        return self.importance_values

    def explain_many_original(
            self,
            x_data: List[dict],
            y_data: List[Any],
            n_inner_samples: Optional[int] = None,
            verbose: bool = True
    ) -> dict:
        """Explain one observation (x_i, y_i) with all data stored according to the original
        definition in https://arxiv.org/abs/2004.00668.

        Args:
            x_data (List[dict]): A list of input data to be explained, as dicts mapping from
                feature names to feature values.
            y_data (List[Any]): Target label of the current observation.
            n_inner_samples (int, optional): Number of model evaluation per feature for the current
                explanation step (observation). Defaults to `None`.
            verbose (bool): Flag indicating if the explanation should print to console (`True`) or
                not (`False`).

        Returns:
            (dict): The current SAGE feature importance scores.
        """
        if n_inner_samples is None:
            n_inner_samples = self.n_inner_samples
        sage_values = {feature: 0. for feature in self.feature_names}
        n_data = len(x_data)
        all_predictions = self._model_function(x_data)
        marginal_prediction = _get_mean_model_output(all_predictions)
        for n, (x_i, y_i) in tqdm(enumerate(zip(x_data, y_data), start=1), total=n_data,
                                  disable=not verbose):
            permutation_chain = np.random.permutation(self.feature_names)
            x_s = {}
            loss_previous = self._loss_function(y_true=y_i, y_prediction=marginal_prediction)
            for feature in permutation_chain:
                x_s[feature] = x_i[feature]
                predictions = []
                for _ in range(1, n_inner_samples + 1):
                    x_marginal = x_data[random.randint(0, n_data - 1)]
                    x_marginal = {**x_marginal, **x_s}
                    predictions.append(self._model_function(x_marginal))
                y = _get_mean_model_output(predictions)
                feature_loss = self._loss_function(y_true=y_i, y_prediction=y)
                marginal_contribution = loss_previous - feature_loss
                sage_values[feature] += marginal_contribution
                loss_previous = feature_loss
            n_data = n
        self.importance_values = {feature: sage_value / n_data
                                  for feature, sage_value in sage_values.items()}
        return self.importance_values
