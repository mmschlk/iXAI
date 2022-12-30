import pandas as pd

from river.stream import iter_pandas
from river.metrics import MSE
from river.utils import Rolling

from sklearn.datasets import fetch_california_housing
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from ixai.explainer import IncrementalPFI
from ixai.explainer.sage import IncrementalSage, IntervalSage
from ixai.imputer import MarginalImputer
from ixai.storage import GeometricReservoirStorage
from ixai.utils.wrappers import SklearnWrapper

if __name__ == "__main__":

    data = fetch_california_housing()
    data_x = pd.DataFrame(data['data'], columns=data['feature_names'])
    data_y = pd.Series(data['target'], name='target')
    data_y = data_y - data_y.mean() / (data_y.max() - data_y.min())
    data_x, data_y = shuffle(data_x, data_y)  # of course don't shuffle real stream data

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.25)

    feature_names = data['feature_names']

    # model
    model = RandomForestRegressor(n_estimators=15)
    model.fit(train_x.values, train_y)
    print(model.score(test_x.values, test_y))

    model_function = SklearnWrapper(model.predict)

    # performance metric and loss
    loss_metric = MSE()  # real metric can be everything from river.metrics used for explanations
    training_metric = Rolling(MSE(), window_size=1000)  # metric wrapper to summarize individual prediction losses

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
        n_inner_samples=1
    )

    interval_sage = IntervalSage(
        model_function=model_function,
        loss_function=loss_metric,
        feature_names=feature_names,
        interval_length=2000,
        n_inner_samples=1
    )

    incremental_pfi = IncrementalPFI(
        model_function=model_function,
        loss_function=loss_metric,
        imputer=imputer,
        storage=storage,
        feature_names=feature_names,
        smoothing_alpha=0.001,
        n_inner_samples=1
    )

    # main training loop data_x and data_y needs to be pd.DataFrames and / or pd.Series
    for n, (x_i, y_i) in enumerate(iter_pandas(data_x, data_y), start=1):
        y_i_pred = model_function(x_i)['output']
        training_metric.update(y_true=y_i, y_pred=y_i_pred)

        # sage inc
        _ = incremental_sage.explain_one(x_i, y_i)
        _ = interval_sage.explain_one(x_i, y_i)
        _ = incremental_pfi.explain_one(x_i, y_i, update_storage=False)

        if n % 1000 == 0:
            print(f"{n}: perf                {training_metric.get()}\n"
                  f"{n}: x_i, y_i            {x_i, y_i}\n"
                  f"{n}: inc-sage            {incremental_sage.importance_values}\n"
                  f"{n}: int-sage            {interval_sage.importance_values}\n"
                  f"{n}: pfi                 {incremental_pfi.importance_values}\n"
                  f"{n}: diff                {incremental_sage.marginal_loss - incremental_sage.model_loss}\n"
                  f"{n}: marginal-loss       {incremental_sage.marginal_loss}\n"
                  f"{n}: model-loss          {incremental_sage.model_loss}\n"
                  f"{n}: sum-sage            {sum(list(incremental_sage.importance_values.values()))}\n")
