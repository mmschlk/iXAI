from increment_explain.utils.wrappers.base import Wrapper

try:
    import torch
except ImportError:
    pass


class TorchSupervisedLearningWrapper(Wrapper):
    """Basic wrapper for torch classification models.

    Warning: This wrapper entails only very basic functionality.
    This wrapper is only intend for basic supervised learning tasks solved with torch.

    This wrapper turns any prediction function output into an iterable (list or np.ndarray) output.
    """

    def __init__(self, model, optimizer, loss_function, n_classes: int = 1, class_labels: list[str] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self._supervised = True
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
        return int(torch.argmax(self.model(x_tensor)))

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
