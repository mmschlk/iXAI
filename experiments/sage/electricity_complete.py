import pandas as pd
import os

import numpy as np


import matplotlib.pyplot as plt

from increment_explain.imputer import MarginalImputer
from increment_explain.storage import GeometricReservoirStorage

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from river.stream import iter_array
from river.ensemble import AdaptiveRandomForestRegressor
from river.metrics import MAE
from river.utils import Rolling

from increment_explain.explainer.sage import IncrementalSageExplainer
from increment_explain.visualization import FeatureImportancePlotter
from increment_explain.utils.converters import RiverToPredictionFunction

from sage.utils import MSELoss

if __name__ == "__main__":

    data = pd.read_csv("C:\\1_Workspaces\\1_Phd_Projects\\IncrementExplain\\data\\data_sets\\energydata_complete.csv")
    data.dropna()
    n_samples = len(data)

    data_y = data.pop("Appliances").values
    data_y = data_y.astype(float)
    print(data_y)



    data_x = data.drop(columns=['date'])
    feature_names = list(data_x.columns)
    print(feature_names)
    data_x = data_x.values
    print(data_x)





    transformer = StandardScaler()
    data_x = transformer.fit_transform(data_x)

    transformer_labels = MinMaxScaler()
    data_y = transformer_labels.fit_transform(data_y.reshape(-1, 1))
    data_y = data_y.reshape(-1)



    MSE = MSELoss(reduction='mean')

    def mse_loss(y_prediction, y_true):
        y_prediction = np.asarray([y_prediction])
        y_prediction = y_prediction.reshape((len(y_prediction), 1))
        y_true = np.asarray([y_true])
        y_true = y_true.reshape((len(y_true))).astype(int)
        return MSE(pred=y_prediction, target=y_true)

    model = AdaptiveRandomForestRegressor()
    model_function = RiverToPredictionFunction(model, feature_names, classification=False).predict
    metric = MAE()
    metric_rolling = Rolling(metric, window_size=200)

    loss_function = mse_loss

    plotter = FeatureImportancePlotter(
        feature_names=feature_names
    )

    SMOOTHING_ALPHA = 0.001

    model_performance = []

    # Instantiating objects
    storage = GeometricReservoirStorage(store_targets=False, size=100)
    # storage = SequenceStorage(store_targets=True)
    imputer = MarginalImputer(model_function, 'product', storage)
    # imputer = DefaultImputer(model, values={'N_1': 5, 'N_2': 3, 'N_10': 2})
    incremental_explainer = IncrementalSageExplainer(
        model_function=model_function,
        feature_names=feature_names,
        storage=storage,
        imputer=imputer,
        n_inner_samples=2,
        loss_function=loss_function,
        smoothing_alpha=SMOOTHING_ALPHA,
        dynamic_setting=True
    )

    for n, (x_i, y_i) in enumerate(iter_array(data_x, data_y, feature_names=feature_names), start=1):
        y_i_pred = model.predict_one(x_i)
        metric_rolling.update(y_true=y_i, y_pred=y_i_pred)
        model_performance.append({'mae': metric.get()})
        inc_fi_values = incremental_explainer.explain_one(x_i, y_i)
        plotter.update(importance_values=inc_fi_values, facet_name='inc')
        model.learn_one(x_i, y_i)
        if n % 1000 == 0:
            print(f"{n}: performance {metric_rolling.get()}\n"
                  f"{n}: inc-SAGE {incremental_explainer.importance_values}\n")
        if n >= n_samples:
            break


    plotter.plot(
        figsize=(5, 10),
        model_performance={'mae': model_performance},
        title=f'Electricity Data Set',
        y_label='SAGE values',
        x_label='Samples',
        names_to_highlight=None,
        h_lines=[{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
        line_styles={'inc': '-', 'int': '--'},
        legend_style=None,
        legend=None,
    )

