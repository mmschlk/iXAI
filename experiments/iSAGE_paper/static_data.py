from experiments.setup.data import get_dataset
from experiments.setup.explainer import get_incremental_sage_explainer, \
    get_batch_sage_explainer, get_interval_sage_explainer
from experiments.setup.loss import get_loss_function, get_training_metric
from experiments.setup.model import get_model

DATASET_NAME = 'agrawal'
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

    dataset = get_dataset(DATASET_NAME, random_seed=DATASET_RANDOM_SEED)
    feature_names = dataset.feature_names
    n_samples = dataset.n_samples

    stream = dataset.stream
    task = dataset.task

    loss_function = get_loss_function(task=task)
    training_metric = get_training_metric(task=task, rolling_window=TRAINING_METRIC_WINDOW)

    model, model_function = get_model(
        model_name=MODEL_NAME, task=task, feature_names=feature_names, **MODEL_PARAMS[MODEL_NAME])

    incremental_explainer = get_incremental_sage_explainer(
        feature_names=feature_names,
        model_function=model_function,
        loss_function=loss_function,
        n_inner_samples=N_INNER_SAMPLES,
        sample_strategy=SAMPLE_STRATEGY
    )

    batch_explainer = get_batch_sage_explainer(
        feature_names=feature_names,
        model_function=model_function,
        loss_function=loss_function,
        n_inner_samples=N_INNER_SAMPLES
    )

    interval_explainer = get_interval_sage_explainer(
        feature_names=feature_names,
        model_function=model_function,
        loss_function=loss_function,
        n_inner_samples=N_INNER_SAMPLES
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
        model.learn_one(x_i, y_i)
        if n % 1000 == 0:
            print(f"{n}: performance {training_metric.get()}\n"
                  f"{n}: inc-sage    {incremental_explainer.importance_values}")
        if n >= n_samples:
            break
