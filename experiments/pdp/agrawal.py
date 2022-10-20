import copy

from river.ensemble import AdaptiveRandomForestClassifier
from river import metrics
from river.utils import Rolling
from river.datasets.synth import Agrawal

from data.batch import AGRAWAL_FEATURE_NAMES

from increment_explain.storage import UniformReservoirStorage
from increment_explain.utils.converters import RiverToPredictionFunction
from increment_explain.explainer.pdp import PartialDependenceExplainer


if __name__ == "__main__":
    RANDOM_SEED = 1
    CLASSIFICATION_FUNCTIONS = (1, 2)

    STREAM_LENGTH = 10000
    STREAM_POSITION = int(STREAM_LENGTH * 0.5)
    N_STREAM_1 = STREAM_POSITION
    N_STREAM_2 = STREAM_LENGTH - N_STREAM_1


    for i in range(1):

        # Setup Data ---------------------------------------------------------------------------------------------------
        feature_names = list(AGRAWAL_FEATURE_NAMES)
        stream_1 = Agrawal(classification_function=CLASSIFICATION_FUNCTIONS[0], seed=RANDOM_SEED)
        stream_2 = Agrawal(classification_function=CLASSIFICATION_FUNCTIONS[1], seed=RANDOM_SEED)

        # Model and training setup
        model = AdaptiveRandomForestClassifier(seed=RANDOM_SEED)
        model_function = RiverToPredictionFunction(model, feature_names, classification=True).predict
        metric = metrics.Accuracy()
        metric = Rolling(metric, window_size=500)

        # Setup explainers ---------------------------------------------------------------------------------------------

        # Instantiating objects
        storage = UniformReservoirStorage(store_targets=False, size=1000)
        # storage = SequenceStorage(store_targets=True)
        # imputer = DefaultImputer(model, values={'N_1': 5, 'N_2': 3, 'N_10': 2})
        incremental_explainer = PartialDependenceExplainer(
            model_function=model_function,
            feature_names=feature_names,
            xlim=(0, 150_000),
            feature_name='salary'
        )

        # warm-up because of errors ------------------------------------------------------------------------------------
        for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
            model.learn_one(x_i, y_i)
            x_explain = copy.deepcopy(x_i)
            if n > 10:
                break

        # Concept 1 ----------------------------------------------------------------------------------------------------

        for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
            y_i_pred = model.predict_one(x_i)
            metric.update(y_true=y_i, y_pred=y_i_pred)
            model.learn_one(x_i, y_i)
            storage.update(x=x_i)
            incremental_explainer.explain_one(
                x_i,
                storage=storage,
                sample_frequency=25,
            )
            if n % 1000 == 0:
                print(f"{n}: performance {metric.get()}\n")
                incremental_explainer.plot_pdp(title=f"PDP after {n} samples")
            if n >= N_STREAM_1:
                break

        # Concept 2 ----------------------------------------------------------------------------------------------------

        for (n, (x_i, y_i)) in enumerate(stream_2, start=1):
            y_i_pred = model.predict_one(x_i)
            metric.update(y_true=y_i, y_pred=y_i_pred)
            model.learn_one(x_i, y_i)
            storage.update(x=x_i)
            incremental_explainer.explain_one(
                x_i,
                storage=storage,
                sample_frequency=25,
            )
            if n % 1000 == 0:
                print(f"{n}: performance {metric.get()}\n")
                incremental_explainer.plot_pdp(title=f"PDP after {n} samples")
            if n >= N_STREAM_2:
                break
