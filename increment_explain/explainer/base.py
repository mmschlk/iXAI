from abc import ABCMeta
from abc import abstractmethod
from increment_explain.utils import ExponentialSmoothingTracker, WelfordTracker


class BaseIncrementalExplainer(metaclass=ABCMeta):
    """Base class for incremental explainer algorithms.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
            self,
            model_function,
            feature_names
    ):
        self.model_function = model_function
        self.feature_names = feature_names
        self.number_of_features = len(feature_names)
        self.seen_samples = 0

    def __repr__(self):
        return f"Explainer for {self.number_of_features} features after {self.seen_samples} samples."


class BaseIncrementalFeatureImportance(BaseIncrementalExplainer):

    @abstractmethod
    def __init__(
            self,
            model_function,
            feature_names,
            dynamic_setting: bool = False,
            smoothing_alpha: float = 0.001
    ):
        super().__init__(model_function, feature_names)
        self.number_of_features = len(feature_names)
        self.seen_samples = 0
        self._smoothing_alpha = smoothing_alpha
        if dynamic_setting:
            assert 0. < smoothing_alpha <= 1., f"The smoothing parameter needs to be in the range of ']0,1]' and not " \
                                               f"'{self._smoothing_alpha}'."
            self._marginal_prediction = ExponentialSmoothingTracker(alpha=self._smoothing_alpha)
            self.importance_trackers = {
                feature_name: ExponentialSmoothingTracker(alpha=self._smoothing_alpha) for feature_name in feature_names
            }
        else:
            self._marginal_prediction = WelfordTracker()
            self.importance_trackers = {
                feature_name: WelfordTracker() for feature_name in feature_names
            }

    @property
    def importance_values(self):
        return {
            feature_name: float(self.importance_trackers[feature_name].tracked_value)
            for feature_name in self.feature_names
        }

    @abstractmethod
    def explain_one(self, x, y):
        pass

    @staticmethod
    def _normalize_importance_values(
            importance_values: dict[str, float],
            mode: str
    ) -> dict[str, float]:
        """Given a data point, it updates the storage.

        Args:
            importance_values: importance values as dictionary of feature-name, value pairs
            mode: normalization mode, possible values are `delta` and `sum`.

        Returns:
            None
        """
        # TODO change documentation
        importance_values_list = list(importance_values.values())
        if mode == 'delta':
            factor = max(importance_values_list) - min(importance_values_list)
        elif mode == 'sum':
            factor = sum(importance_values_list)
        else:
            raise NotImplementedError(f"mode must be either 'sum', or 'delta' not '{mode}'")
        normalized_importance_values = {feature: importance_value / factor
                                        for feature, importance_value in importance_values.items()}
        return normalized_importance_values
