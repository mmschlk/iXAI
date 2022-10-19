import pandas as pd

from river.stream import iter_pandas
from river.ensemble import AdaptiveRandomForestRegressor
from river.metrics import MSE
from river.utils import Rolling

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.utils import shuffle

if __name__ == "__main__":

    data = fetch_california_housing()
    data_x = pd.DataFrame(data['data'], columns=data['feature_names'])
    data_y = pd.Series(data['target'], name='target')
    data_x, data_y = shuffle(data_x, data_y)  # of course don't shuffle real stream data

    # model
    model = AdaptiveRandomForestRegressor(n_models=50)  # most important hyperparameter is n_models

    # performance metric
    metric = MSE()  # real metric can be everything from river.metrics
    metric_rolling = Rolling(metric, window_size=200)  # metric wrapper to summarize individual prediction losses

    model_performance = []  # for plotting

    # main training loop data_x and data_y needs to be pd.DataFrames and / or pd.Series
    for n, (x_i, y_i) in enumerate(iter_pandas(data_x, data_y), start=1):
        y_i_pred = model.predict_one(x_i)
        metric_rolling.update(y_true=y_i, y_pred=y_i_pred)
        model_performance.append(metric.get())
        model.learn_one(x_i, y_i)
        if n % 1000 == 0:
            print(f"{n}: performance {metric_rolling.get()}\n"
                  f"{n}: y_i         {y_i}\n"
                  f"{n}: y_i_pred    {y_i_pred}\n"
                  f"{n}: x_i         {x_i}\n")

    plt.plot(model_performance)
    plt.ylim((0, 1))
    plt.title(f"Model Performance")
    plt.xlabel(f"Time / Samples")
    plt.ylabel(f"Performance")
    plt.show()
