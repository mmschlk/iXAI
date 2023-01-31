import abc
import typing

import numpy as np


class Wrapper(metaclass=abc.ABCMeta):
    """Base Wrapper

    Warning: This class should not be used directly.
    Use derived classes instead.

    Attributes:
        default_label (str): Default dict key for single value model outputs.
    """

    def __init__(self, prediction_function, feature_names):
        self._feature_names: list = feature_names
        self._prediction_function = prediction_function
        self.default_label = 'output'

    @abc.abstractmethod
    def __call__(self, x: typing.Union[typing.List[dict], dict]) -> typing.Union[dict, typing.List[dict]]:
        raise NotImplementedError

    # {features: values} -> [[values]]
    def convert_1d_input_to_arr(self, x_dict: dict) -> np.ndarray:
        """Transforms a 1d input (a single dict) into an array of shape (1, N_Features).

        Note:
            If the `feature_names` parameter is not None in the initialization of the Wrapper. The transformed output
            array is resorted in this order and to only contain the specified features.

        Args:
            x_dict (dict): Input dictionary of type `{feature_name: feature_value}`.

        Returns:
            (np.ndarray): The transformed input array of shape (1, N_Features).

        Examples:
            Basic usage:

            >>> x_input = {'feature_1': 'value_1', 'feature_2': 2}
            >>> x_array = Wrapper.convert_1d_input_to_arr(x_input)
            >>> print(x_array, x_array.shape)
            >>> [['value_1', 2]], (1, 2)
        """
        if self._feature_names is not None:
            x_dict = {feature: x_dict[feature] for feature in self._feature_names}
        return np.asarray(list(x_dict.values())).reshape(1, -1)

    # [{features: values_1}, {features: values_2}] -> [[values], [values_2]]
    def convert_2d_input_to_arr(self, x_dicts: typing.Sequence[dict]) -> np.ndarray:
        """Transforms a 2d input (a list of dicts) into an array of shape (N_Instances, N_Features).

        Note:
            If the `feature_names` parameter is not None in the initialization of the Wrapper. The transformed output
            array is resorted in this order and to only contain the specified features.

        Args:
            x_dicts (list[dict]): List of input dictionaries of type `{feature_name: feature_value}`.

        Returns:
            (np.ndarray): The transformed input array of shape (N_Instances, N_Features).

        Examples:
            Basic usage:

            >>> x_inputs = [{'feature_1': 'value_1', 'feature_2': 2}, {'feature_1': 'value_1', 'feature_2': 3}]
            >>> x_array = Wrapper.convert_1d_input_to_arr(x_inputs)
            >>> print(x_array, x_array.shape)
            >>> [['value_1', 2], ['value_1', 3]], (2, 2)
        """
        x_input = []
        for i in range(len(x_dicts)):
            if self._feature_names is not None:
                x_input_i = [x_dicts[i][feature] for feature in self._feature_names]
            else:
                x_input_i = list(x_dicts[i].values())
            x_input.append(x_input_i)
        return np.asarray(x_input)

    # [[output]] or [output] -> {'output': output}, [[output_1, output_2]] -> {0: output_1, 1: output_2}
    def convert_arr_output_to_dict(self, y_prediction: np.ndarray) -> dict:
        """Transforms a prediction output into a dict of outputs.

        Note:
            If the `feature_names` parameter is not None in the initialization of the Wrapper. The transformed output
            array is resorted in this order and to only contain the specified features.

        Args:
            y_prediction (np.ndarray): Output array from model, of shape (1, N_Outputs) or (N_Outputs)

        Returns:
            (dict): The transformed output dictionary. If the output dimension of the model is 1 (e.g. regression,
                classification labels) the output dict contains one element mapping from 'output' key to the model
                output (e.g. original output was `[[1]]` or `[1]` then `{'output': 1}` is the transformed output).

        Examples:
            Onedimensional Model Output (Labels):

            >>> model_output = np.asarray([1])
            >>> Wrapper.convert_arr_output_to_dict(model_output)
            >>> {'output': 1}
            >>> model_output = np.asarray([[1]])
            >>> Wrapper.convert_arr_output_to_dict(model_output)
            >>> {'output': 1}

            Multidimnesnional Model Output (Probas):

            >>> model_output = np.asarray([[0.05, 0.5, 0.45]])
            >>> Wrapper.convert_arr_output_to_dict(model_output)
            >>> {0: 0.05, 1: 0.5, 2: 0.45}

        Raises:
            ValueError: If the model outputs are strings.
        """
        try:
            return {self.default_label: float(y_prediction)}
        except TypeError:  # y_prediction is not a size-1 array or real_valued number
            y_prediction = y_prediction.flatten()
            y_prediction = {i: y_prediction[i] for i in range(y_prediction.shape[0])}
        except ValueError as e:  # y_prediction is probably a size-1 array containing a string
            raise ValueError(f"Prediction is probably a string: {y_prediction}. Only numeric values allowed. "
                             f"Exception is raised: {e}.")
        return y_prediction
