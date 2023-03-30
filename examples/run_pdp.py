import copy

from river.ensemble import AdaptiveRandomForestClassifier
from river import metrics
from river.utils import Rolling
from river.datasets.synth import Agrawal
from ixai.storage import UniformReservoirStorage
from ixai.utils.wrappers import RiverWrapper
from ixai.explainer.pdp import IncrementalPDP


if __name__ == "__main__":
    RANDOM_SEED = 1
    CLASSIFICATION_FUNCTIONS = (1, 2)

    STREAM_LENGTH = 10000
    STREAM_POSITION = int(STREAM_LENGTH * 0.5)
    N_STREAM_1 = STREAM_POSITION
    N_STREAM_2 = STREAM_LENGTH - N_STREAM_1

    for i in range(1):

        # Setup Data ---------------------------------------------------------------------------------------------------

        stream_1 = Agrawal(classification_function=CLASSIFICATION_FUNCTIONS[0], seed=RANDOM_SEED)
        feature_names = list([x_0 for x_0, _ in stream_1.take(1)][0].keys())
        stream_2 = Agrawal(classification_function=CLASSIFICATION_FUNCTIONS[1], seed=RANDOM_SEED)

        # Model and training setup
        model = AdaptiveRandomForestClassifier(seed=RANDOM_SEED)
        model_function = RiverWrapper(model.predict_proba_one)
        loss_metric = metrics.Accuracy()
        training_metric = Rolling(metrics.Accuracy(), window_size=1000)

        # Setup explainers ---------------------------------------------------------------------------------------------

        # Instantiating objects
        storage = UniformReservoirStorage(store_targets=False, size=1000)
        # storage = SequenceStorage(store_targets=True)
        # imputer = DefaultImputer(model, values={'N_1': 5, 'N_2': 3, 'N_10': 2})
        incremental_explainer = IncrementalPDP(
            model_function=model_function,
            feature_names=feature_names,
            xlim={'salary':(0, 150_000)},
            gridsize=8,
            pdp_feature_list=['salary'],
            storage=storage,
            storage_size=50
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
            training_metric.update(y_true=y_i, y_pred=y_i_pred)
            model.learn_one(x_i, y_i)
            incremental_explainer.explain_one(
                x_i
            )
            if n % 1000 == 0:
                print(f"{n}: performance {training_metric.get()}\n")
                incremental_explainer.plot_pdp(title=f"PDP after {n} samples - 1", feature_name='salary')
            if n >= N_STREAM_1:
                break

        # Concept 2 ----------------------------------------------------------------------------------------------------

        for (n, (x_i, y_i)) in enumerate(stream_2, start=1):
            y_i_pred = model.predict_one(x_i)
            training_metric.update(y_true=y_i, y_pred=y_i_pred)
            model.learn_one(x_i, y_i)
            incremental_explainer.explain_one(
                x_i
            )
            if n % 1000 == 0:
                print(f"{n}: performance {training_metric.get()}\n")
                incremental_explainer.plot_pdp(title=f"PDP after {n} samples -2", feature_name='salary')
            if n >= N_STREAM_2:
                break
