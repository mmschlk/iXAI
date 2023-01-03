import river.metrics
from river.utils import Rolling
from river.ensemble import AdaptiveRandomForestClassifier

from river.datasets.synth import Agrawal

from ixai.explainer import IncrementalPFI
from ixai.explainer.sage import IncrementalSage, IntervalSage
from ixai.imputer import MarginalImputer
from ixai.storage import GeometricReservoirStorage
from ixai.utils.wrappers.river import RiverWrapper

N_SAMPLES = 10_000

if __name__ == "__main__":

    # Get Data -------------------------------------------------------------------------------------
    stream = Agrawal(classification_function=1, seed=42)
    feature_names = list([x_0 for x_0, _ in stream.take(1)][0].keys())

    loss_metric = river.metrics.Accuracy()
    training_metric = Rolling(river.metrics.Accuracy(), window_size=1000)

    model = AdaptiveRandomForestClassifier(n_models=15, max_depth=10, leaf_prediction='mc')
    model_function = RiverWrapper(model.predict_one)

    # Get imputer and explainers -------------------------------------------------------------------
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
        n_inner_samples=1,
        loss_bigger_is_better=True
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

    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        # predicting
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)

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
