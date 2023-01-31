"""
This module contains the interval SAGE explainer.
"""
from typing import Callable, Any, Sequence, Union, Dict, Optional

from river.metrics.base import Metric

from ixai.explainer.sage.batch import BatchSage
from ixai.imputer import BaseImputer, MarginalImputer
from ixai.storage import IntervalStorage


class IntervalSage(BatchSage):
    """Interval SAGE Explainer

    Computes SAGE importance values according to its original definition in
    https://arxiv.org/abs/2004.00668 at set time intervals. A Storage of the last n (specified by
    `storage_length`) observations are kept on which the explanations are created.

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
        storage (IntervalStorage, optional): Optional incremental data storage Mechanism.
            Defaults to `IntervalStorage(size=interval_length)`.
        imputer (BaseImputer, optional): Incremental imputing strategy to be used. Defaults to
            `MarginalImputer(sampling_strategy='joint')`.
        n_inner_samples (int): Number of model evaluation per feature and explanation step
            (observation). Defaults to 1.
        interval_length (int): Length of the explanation interval after which the explanations
            are created. Defaults to 1000.

    Attributes:
        feature_names (Sequence[Union[str, int, float]]): The feature names of the dataset.
        n_inner_samples (int): Number of model evaluation per feature and explanation step
            (observation).
        seen_samples (int): Number of observations seen.
    """
    def __init__(
            self,
            model_function: Callable[[Any], Any],
            feature_names: Sequence[Union[str, int, float]],
            loss_function: Union[Metric, Callable[[Any, Dict], float]],
            n_inner_samples: int = 1,
            interval_length: int = 1000,
            storage_length: int = 1000,
            storage: IntervalStorage = None,
            imputer: BaseImputer = None,
    ):
        if storage is None:
            storage = IntervalStorage(store_targets=True, size=storage_length)
        assert isinstance(storage, IntervalStorage), f"Only 'IntervalStorage' expected not " \
                                                     f"{type(storage)}."

        if imputer is None:
            imputer = MarginalImputer(
                model_function=model_function, sampling_strategy='joint', storage_object=storage)

        super().__init__(
            model_function=model_function,
            feature_names=feature_names,
            loss_function=loss_function,
            n_inner_samples=n_inner_samples,
            storage=storage,
            imputer=imputer
        )
        self.interval_length = interval_length
        self.seen_samples = 0

    def explain_one(
            self,
            x_i: dict,
            y_i: Any,
            n_inner_samples: Optional[int] = None,
            update_storage: bool = True,
            force_explain: bool = False,
            verbose: bool = True
    ) -> dict:
        """Explain one observation (x_i, y_i) if enough time between the last explanation and now
            has passed (`interval_length`).

        Args:
            x_i (dict): The input features of the current observation as a dict of feature names to
                feature values.
            y_i (Any): Target label of the current observation.
            n_inner_samples (int, optional): Number of model evaluation per feature for the current
                explanation step (observation). Defaults to `None`.
            update_storage (bool): Flag if the underlying incremental data storage mechanism is to
                be updated with the new observation (`True`) or not (`False`). Defaults to `True`.
            force_explain (bool): Overrides the `interval_length` restriction and explains the
                current sample. This does not override the set `interval_length` globally, such that
                the explainer is still run in the same rhythm as before.
            verbose (bool): Flag indicating if the explanation should print to console (`True`) or
                not (`False`).

        Returns:
            (dict): The current SAGE feature importance scores.
        """
        if update_storage:
            self._storage.update(x=x_i, y=y_i)
        self.seen_samples += 1
        if not force_explain and self.seen_samples % self.interval_length != 0:
            return self.importance_values
        x_data, y_data = self._storage.get_data()
        super().explain_many(x_data=x_data, y_data=y_data, n_inner_samples=n_inner_samples,
                             verbose=verbose)
        return self.importance_values
