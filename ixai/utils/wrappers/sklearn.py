import typing
import warnings

from ixai.utils.wrappers.base import Wrapper


class SklearnWrapper(Wrapper):
    """Wrapper for sklearn prediction functions.

    This wrapper turns any prediction function ouput into an iterable (list or np.ndarray) output. And allows
    for dict inputs.

    Examples:
        Basic usage:

        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> model_function = SklearnWrapper(model.predict)

        For classifiers returning probas:

        >>> model_function = SklearnWrapper(model.predict_proba)

        If the dict-inputs may be in a different orderings

        >>> feature_orderings: list = ['feature_1', 'feature_2', 'feature_3']
        >>> model_function = SklearnWrapper(model.predict, feature_names=feature_orderings)

    Note:
        If the sklearn model is trained with access to the feature names (e.g. trained on a pandas DataFrame) it will
        usually raise a warning, if unnamed feature values are provided (e.g. in the form of a np.ndarray). Since
        instantiating a pandas DataFrame for each input is computationally more expensive, the specific warning is
        manually suppressed in this Wrapper.
    """

    def __init__(self, prediction_function: typing.Callable, feature_names: typing.Optional[list] = None):
        super().__init__(prediction_function, feature_names)

    def __call__(self, x: typing.Union[typing.List[dict], dict]) -> typing.Union[dict, typing.List[dict]]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names, but")
            if isinstance(x, dict):
                x_input = self.convert_1d_input_to_arr(x)
                return self.convert_arr_output_to_dict(self._prediction_function(x_input))
            x_input = self.convert_2d_input_to_arr(x)
            y_predictions = self._prediction_function(x_input)
            y_prediction = [self.convert_arr_output_to_dict(y_predictions[i]) for i in range(len(y_predictions))]
            return y_prediction
