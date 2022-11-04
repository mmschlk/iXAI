from data.stream.synth import Agrawal
from experiments.setup.data import get_dataset
from experiments.setup.explainer import get_incremental_sage_explainer, \
    get_batch_sage_explainer, get_interval_sage_explainer
from experiments.setup.loss import get_loss_function, get_training_metric
from experiments.setup.model import get_model
from increment_explain.explainer.sage import IncrementalSageExplainer
from increment_explain.imputer import MarginalImputer
from increment_explain.imputer.tree_imputer import TreeImputer
from increment_explain.storage import GeometricReservoirStorage
from increment_explain.storage.tree_storage import TreeStorage
from increment_explain.visualization import FeatureImportancePlotter

DATASET_NAME = 'agrawal 1'
DATASET_RANDOM_SEED = 1
MODEL_NAME = 'ARF'

N_INNER_SAMPLES = 1

TRAINING_METRIC_WINDOW = 100

MODEL_PARAMS = {
    'ARF': {
        'n_models': 25,
    }
}

SAMPLE_STRATEGY = 'joint'

if __name__ == "__main__":

    # Get Data ---------------------------------------------------------------------------------------------------------
    dataset = Agrawal(classification_function=1, random_seed=42, n_samples=20_000)
    feature_names = dataset.feature_names
    cat_feature_names = dataset.cat_feature_names
    num_feature_names = dataset.num_feature_names
    n_samples = dataset.n_samples
    stream = dataset.stream

    # Get Model / Loss and Training Procedure --------------------------------------------------------------------------
    task = dataset.task
    loss_function = get_loss_function(task=task)
    training_metric, _ = get_training_metric(task=task, rolling_window=TRAINING_METRIC_WINDOW)
    model, model_function = get_model(
        model_name=MODEL_NAME, task=task, feature_names=feature_names, **MODEL_PARAMS[MODEL_NAME])

    # Get explainers ---------------------------------------------------------------------------------------------------
    # Instantiating objects
    storage = TreeStorage(cat_feature_names=cat_feature_names, num_feature_names=num_feature_names)
    imputer = TreeImputer(model_function, storage_object=storage, direct_predict_numeric=False, use_storage=True)
    incremental_explainer = IncrementalSageExplainer(
        model_function=model_function,
        feature_names=feature_names,
        storage=storage,
        imputer=imputer,
        n_inner_samples=1,
        loss_function=loss_function,
        smoothing_alpha=0.0005,
        dynamic_setting=True
    )

    # Instantiating objects
    storage_marginal = GeometricReservoirStorage(store_targets=False, size=100)
    imputer_marginal = MarginalImputer(model_function, 'product', storage_marginal)
    incremental_explainer_marginal = IncrementalSageExplainer(
        model_function=model_function,
        feature_names=feature_names,
        storage=storage_marginal,
        imputer=imputer_marginal,
        n_inner_samples=1,
        loss_function=loss_function,
        smoothing_alpha=0.0005,
        dynamic_setting=True
    )

    # Plotter ----------------------------------------------------------------------------------------------------------
    plotter = FeatureImportancePlotter(
        feature_names=feature_names
    )

    # warm-up because of errors ------------------------------------------------------------------------------------
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        model.learn_one(x_i, y_i)
        if n > 10:
            break

    # Concept 1 ----------------------------------------------------------------------------------------------------

    model_performance = []

    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model_performance.append({'performance': training_metric.get()})
        inc_fi_values = incremental_explainer.explain_one(x_i, y_i)
        plotter.update(importance_values=inc_fi_values, facet_name='cond.')
        inc_fi_values_marginal = incremental_explainer_marginal.explain_one(x_i, y_i)
        plotter.update(importance_values=inc_fi_values_marginal, facet_name='marg.')

        model.learn_one(x_i, y_i)
        if n % 1000 == 0:
            print(f"{n}: performance {training_metric.get()}\n"
                  f"{n}: inc-sage    {incremental_explainer.importance_values}\n"
                  f"{n}: inc-sage-ma {incremental_explainer_marginal.importance_values}")
        if n >= n_samples:
            break

    # Plotting -----------------------------------------------------------------------------------------------------
    plotter.plot(
        figsize=(5, 10),
        model_performance={'performance': model_performance},
        title=f'Agrawal Stream',
        y_label='SAGE values',
        x_label='Samples',
        names_to_highlight=['salary', 'commission', 'age'],
        h_lines=[{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
        line_styles={'cond.': '-', 'marg.': '-'},
        legend_style={},
        facet_not_to_highlight=['marg.']
    )