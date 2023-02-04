from typing import Callable

from .base import BaseImputer


class DefaultImputer(BaseImputer):
    """Imputer that always imputes with default values (e.g. mean values).

    Args:
        model_function (Callable): Model function to impute values for.
        values (dict): Mapping from feature names to default values for the features.
    """
    def __init__(self, model_function: Callable, values: dict):
        self.values = values
        super().__init__(
            model_function=model_function
        )

    def impute(self, feature_subset: list, x_i: dict, n_samples: int = 1):
        """Imputes a subset of features with default values

        Args:
            feature_subset (list): Feature names to impute.
            x_i (dict): Instance to impute values in.
            n_samples (int): Number of imputed instances to create (effectively cloning it).
                Defaults to 1.
        """
        sampled_values = {feature: self.values[feature] for feature in feature_subset}
        prediction = self.model_function({**x_i, **sampled_values})
        prediction = [prediction for _ in range(n_samples)]
        return prediction
