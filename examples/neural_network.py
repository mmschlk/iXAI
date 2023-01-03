import river.metrics
import torch
from river.utils import Rolling
from river import preprocessing, compose
from river.datasets import Elec2

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ixai.explainer import IncrementalPFI
from ixai.explainer.sage import IncrementalSage, IntervalSage
from ixai.imputer import MarginalImputer
from ixai.storage import GeometricReservoirStorage
from ixai.utils.wrappers.torch import TorchSupervisedLearningWrapper, TorchWrapper

N_SAMPLES = 10_000


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(N_INPUT, 100)
        self.layer_2 = nn.Linear(100, 10)
        self.layer_3 = nn.Linear(10, N_CLASSES)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


if __name__ == "__main__":

    # Get Data ---------------------------------------------------------------------------------------------------------
    stream = Elec2()
    feature_names = list([x_0 for x_0, _ in stream.take(1)][0].keys())

    N_INPUT = len(feature_names)
    N_CLASSES = 3
    network = Net()
    network_loss_function = nn.CrossEntropyLoss()
    network_optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    loss_metric = river.metrics.CrossEntropy()
    training_metric = Rolling(river.metrics.Accuracy(), window_size=1000)

    model = TorchSupervisedLearningWrapper(
        model=network, loss_function=network_loss_function, optimizer=network_optimizer, n_classes=N_CLASSES,
        task='Classification')
    scaler = preprocessing.StandardScaler()
    model = compose.Pipeline(scaler, model)


    def network_link_function(x):
        return torch.softmax(network(x), dim=-1)
    model_function = TorchWrapper(network_link_function)

    # Get imputer and explainers ---------------------------------------------------------------------------------------
    storage = GeometricReservoirStorage(
        size=200,
        store_targets=False
    )

    imputer = MarginalImputer(
        model_function=model_function,
        storage_object=storage,
        sampling_strategy="joint"
    )

    incremental_sage = IncrementalSage(
        model_function=model_function,
        loss_function=loss_metric,
        imputer=imputer,
        storage=storage,
        feature_names=feature_names,
        smoothing_alpha=0.001,
        n_inner_samples=5,
        loss_bigger_is_better=True
    )

    interval_sage = IntervalSage(
        model_function=model_function,
        loss_function=loss_metric,
        feature_names=feature_names,
        interval_length=2000,
        n_inner_samples=5
    )

    incremental_pfi = IncrementalPFI(
        model_function=model_function,
        loss_function=loss_metric,
        imputer=imputer,
        storage=storage,
        feature_names=feature_names,
        smoothing_alpha=0.001,
        n_inner_samples=5
    )

    for (n, (x_i, y_i)) in enumerate(stream, start=1):

        # here we manually 'extend' the binary classification problem to a multi-class classification problem
        # by artificially introducing another class whenever the 'period' feature exceeds 0.5 -> this does not make
        # any sense and is just for illustrative purposes
        if x_i['period'] >= 0.5:
            y_i = 2

        # predicting
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)

        # transforming for explanation's model function
        x_i_transformed = scaler.transform_one(x_i)
        _ = incremental_sage.explain_one(x_i_transformed, y_i)

        # sage inc
        _ = incremental_sage.explain_one(x_i, y_i)
        _ = interval_sage.explain_one(x_i, y_i)
        _ = incremental_pfi.explain_one(x_i, y_i, update_storage=False)

        # learning
        model.learn_one(x_i, y_i)

        if n % 1000 == 0:
            print(f"{n}: perf                {training_metric.get()}\n"
                  f"{n}: x_i                 {x_i}\n"
                  f"{n}: inc-sage            {incremental_sage.importance_values}\n"
                  f"{n}: int-sage            {interval_sage.importance_values}\n"
                  f"{n}: pfi-sage            {incremental_pfi.importance_values}\n"
                  f"{n}: diff                {incremental_sage.marginal_loss - incremental_sage.model_loss}\n"
                  f"{n}: marginal-loss       {incremental_sage.marginal_loss}\n"
                  f"{n}: model-loss          {incremental_sage.model_loss}\n"
                  f"{n}: sum-sage            {sum(list(incremental_sage.importance_values.values()))}\n")

        if n >= N_SAMPLES:
            break
