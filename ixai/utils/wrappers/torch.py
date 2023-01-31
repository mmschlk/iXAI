import typing
import warnings

from ixai.utils.wrappers.base import Wrapper

try:
    import torch
except ImportError:
    pass


class TorchWrapper(Wrapper):
    """Wrapper for torch link functions.

    This wrapper turns any torch ouput tensor into a dict output and allows for dict inputs.

    Examples:
        Basic usage:

        >>> torch_module: torch.nn.Module = torch_module
        >>> module_device = 'cpu'
        >>> model_function = TorchWrapper(torch_module, device=module_device)

        For classifiers returning class_labels:

        >>> def link_function_class(x):
        >>>     return torch.argmax(torch.softmax(torch_module(x), dim=-1), dim=-1)
        >>> model_function = TorchWrapper(link_function_class)

        For classifiers returning probas:

        >>> def link_function_probas(x):
        >>>     return torch.softmax(torch_module(x), dim=-1)
        >>> model_function = TorchWrapper(link_function_probas)

        If the dict-inputs may be in a different orderings:

        >>> feature_orderings: list = ['feature_1', 'feature_2', 'feature_3']
        >>> model_function = TorchWrapper(link_function_probas, feature_names=feature_orderings)
    """

    def __init__(
            self,
            link_function: typing.Union["Module", typing.Callable],
            feature_names: typing.Optional[list] = None,
            device: str = 'cpu'
    ):
        """
        Args:
            link_function (Union[torch.nn.Module, Callable]): The function linking from the model input to the output.
            device: (str): Torch device flag where the model is running. Defaults to `'cpu'`.
            feature_names (list[str], optional): A ordered list of feature names what features should be provided.
        """
        super().__init__(link_function, feature_names)
        self._device: str = device

    def __call__(self, x: typing.Union[typing.List[dict], dict]) -> typing.Union[dict, typing.List[dict]]:
        """Runs the torch model with the given input dictionary.

        Args:
            x (Union[list[dict], dict]): Input features in the form of a dict (1d-input) mapping from feature names to
            feature values or a list of such dicts.

        Returns:
            (Union[list[dict], dict]): The model output as a dictionary following river conventions.

        Examples:
            Basic usage:

            >>> def link_function_probas(x):
            >>>     return torch.softmax(torch_module(x), dim=-1)
            >>> model_function: typing.Callable = TorchWrapper(link_function_probas)
            >>> input_dict = {'feature_1': 1, 'feature_2': 0}
            >>> model_function(input_dict)
            >>> {0: 0.45, 1: 0.05, 2: 0.5}
        """
        if isinstance(x, dict):
            x_input = self.convert_1d_input_to_arr(x)
            x_input = torch.tensor(x_input, device=self._device, dtype=torch.float32)
            output = self._prediction_function(x_input).detach().cpu().numpy()
            return self.convert_arr_output_to_dict(output)
        x_input = self.convert_2d_input_to_arr(x)
        x_input = torch.tensor(x_input, device=self._device, dtype=torch.float32)
        y_predictions = self._prediction_function(x_input).detach().cpu().numpy()
        y_prediction = [self.convert_arr_output_to_dict(y_predictions[i]) for i in range(len(y_predictions))]
        return y_prediction


class TorchSupervisedLearningWrapper(Wrapper):
    """Basic wrapper for torch classification models.

    Warning: This wrapper entails only very basic functionality.
    This wrapper is only intend for basic supervised learning tasks solved with torch.

    This wrapper turns any prediction function output into an iterable (list or np.ndarray) output.
    """

    def __init__(self, model, optimizer, loss_function, task, n_classes: int = 1, class_labels: list = None):
        warnings.warn("TorchSupervisedLearningWrapper is deprecated and will be removed in future"
                      " releases.", DeprecationWarning)
        super().__init__(prediction_function=None, feature_names=None)
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self._supervised = True
        self.task = task
        self.n_classes = n_classes
        if class_labels is not None:
            self.n_classes = len(class_labels)
            self.class_labels = class_labels
        else:
            self.class_labels = [i for i in range(self.n_classes)]
        self.device = 'cpu'

    def _dict_to_tensor(self, x_i):
        return torch.tensor([list(x_i.values())], dtype=torch.float32, device=self.device)

    def _tensor_to_probas(self, prediction_tensor):
        probas_tensor = torch.softmax(prediction_tensor, dim=-1)
        probas_dict = {class_label: float(probas_tensor[0][i])
                       for class_label, i in zip(self.class_labels, range(self.n_classes))}
        return probas_dict

    def predict_one(self, x_i):
        x_tensor = self._dict_to_tensor(x_i)
        if self.task == 'Regression':
            pred = float(self.model(x_tensor))
        else:
            pred = int(torch.argmax(self.model(x_tensor)))
        return pred

    def predict_proba_one(self, x_i):
        x_tensor = self._dict_to_tensor(x_i)
        y_prediction = self.model(x_tensor)
        y_prediction = self._tensor_to_probas(y_prediction)
        return y_prediction

    def learn_one(self, x, y):
        x_tensor = torch.tensor([list(x.values())], dtype=torch.float32)
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(x_tensor)
        y_tensor = torch.zeros(1, self.n_classes, dtype=torch.float32)
        y_tensor[0, int(y)] = 1.
        loss = self.loss_function(y_pred, y_tensor)
        loss.backward()
        self.optimizer.step()

    def __call__(self, x_i):
        return self.predict_one(x_i)
