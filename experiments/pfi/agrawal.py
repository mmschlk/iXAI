import numpy as np

from river.ensemble import AdaptiveRandomForestClassifier
from river import metrics
from river.utils import Rolling

from data.stream.synth import Agrawal

from increment_explain.imputer import MarginalImputer
from increment_explain.storage import UniformReservoirStorage
from increment_explain.utils.converters import RiverToPredictionFunction
from increment_explain.explainer.pfi import IncrementalPFI
from increment_explain.visualization import FeatureImportancePlotter

from sage.utils import CrossEntropyLoss


def cross_entropy_loss(y_prediction, y_true):
    y_prediction = np.asarray([y_prediction])
    y_prediction = y_prediction.reshape((len(y_prediction), 1))
    y_true = np.asarray([y_true])
    y_true = y_true.reshape((len(y_true))).astype(int)
    return CROSS_ENTROPY(pred=y_prediction, target=y_true)


if __name__ == "__main__":
    CROSS_ENTROPY = CrossEntropyLoss(reduction='mean')
    RANDOM_SEED = 1
    CLASSIFICATION_FUNCTIONS = (1, 2)

    STREAM_LENGTH = 10000
    STREAM_POSITION = int(STREAM_LENGTH * 0.5)
    N_STREAM_1 = STREAM_POSITION
    N_STREAM_2 = STREAM_LENGTH - N_STREAM_1

    EXPLAINER_NAME = "inc-PFI"

    SMOOTHING_ALPHA = 0.001

    for i in range(1):

        # Setup Data ---------------------------------------------------------------------------------------------------
        dataset_1 = Agrawal(classification_function=CLASSIFICATION_FUNCTIONS[0], random_seed=RANDOM_SEED, n_samples=N_STREAM_1)
        dataset_2 = Agrawal(classification_function=CLASSIFICATION_FUNCTIONS[1], random_seed=RANDOM_SEED, n_samples=N_STREAM_1)
        stream_1 = dataset_1.stream
        stream_2 = dataset_2.stream

        feature_names = list(dataset_1.feature_names)

        # Model and training setup
        model = AdaptiveRandomForestClassifier(seed=RANDOM_SEED)
        model_function = RiverToPredictionFunction(model, feature_names).predict
        metric = metrics.Accuracy()
        metric = Rolling(metric, window_size=500)

        loss_function = cross_entropy_loss

        plotter = FeatureImportancePlotter(
            feature_names=feature_names
        )

        model_performance = []

        # Setup explainers ---------------------------------------------------------------------------------------------

        # Instantiating objects
        storage = UniformReservoirStorage(store_targets=False, size=100)
        # storage = SequenceStorage(store_targets=True)
        imputer = MarginalImputer(model_function, 'product', storage)
        # imputer = DefaultImputer(model, values={'N_1': 5, 'N_2': 3, 'N_10': 2})
        incremental_explainer = IncrementalPFI(
            model_function=model_function,
            feature_names=feature_names,
            storage=storage,
            imputer=imputer,
            n_samples=2,
            loss_function=loss_function,
            smoothing_alpha=SMOOTHING_ALPHA,
            dynamic_setting=True
        )

        # warm-up because of errors ------------------------------------------------------------------------------------
        for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
            model.learn_one(x_i, y_i)
            if n > 10:
                break

        # Concept 1 ----------------------------------------------------------------------------------------------------

        for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
            y_i_pred = model.predict_one(x_i)
            metric.update(y_true=y_i, y_pred=y_i_pred)
            model_performance.append({'accuracy': metric.get()})
            inc_fi_values = incremental_explainer.explain_one(x_i, y_i)
            plotter.update(importance_values=inc_fi_values, facet_name='inc')
            model.learn_one(x_i, y_i)
            if n % 1000 == 0:
                print(f"{n}: performance {metric.get()}\n"
                      f"{n}: {EXPLAINER_NAME} {incremental_explainer.importance_values}")
            if n >= N_STREAM_1:
                break

        # Concept 2 ----------------------------------------------------------------------------------------------------

        for (n, (x_i, y_i)) in enumerate(stream_2, start=1):
            y_i_pred = model.predict_one(x_i)
            metric.update(y_true=y_i, y_pred=y_i_pred)
            model_performance.append({'accuracy': metric.get()})
            inc_fi_values = incremental_explainer.explain_one(x_i, y_i)
            plotter.update(importance_values=inc_fi_values, facet_name='inc')
            model.learn_one(x_i, y_i)
            if n % 1000 == 0:
                print(f"{n}: performance {metric.get()}\n"
                      f"{n}: {EXPLAINER_NAME} {incremental_explainer.importance_values}")
            if n >= N_STREAM_2:
                break

        # Plotting -----------------------------------------------------------------------------------------------------
        plotter.plot(
            figsize=(5, 10),
            model_performance={'accuracy': model_performance},
            title=f'Agrawal Stream (fn. {CLASSIFICATION_FUNCTIONS[0]} to fn. {CLASSIFICATION_FUNCTIONS[1]})',
            y_label='SAGE values',
            x_label='Samples',
            names_to_highlight=['salary', 'commission', 'age', 'elevel'],
            v_lines=[{'x': N_STREAM_1, 'ls': '--', 'c': 'black', 'linewidth': 1}],
            h_lines=[{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
            line_styles={'inc': '-', 'int': '--'},
            legend_style={}
        )
