import time

from experiments.setup.data import get_dataset, get_concept_drift_dataset
from experiments.setup.explainer import get_incremental_sage_explainer, \
    get_batch_sage_explainer, get_interval_sage_explainer, get_incremental_pfi_explainer
from experiments.setup.loss import get_loss_function, get_training_metric
from experiments.setup.model import get_model

DATASET_1_NAME = 'agrawal 1'
DATASET_2_NAME = 'agrawal 2'
DATASET_RANDOM_SEED = 1
SHUFFLE_DATA = False
N_SAMPLES_1 = 100
N_SAMPLES_2 = 100

CONCEPT_DRIFT_POSITION = 0.05
CONCEPT_DRIFT_WIDTH = 0.05
CONCEPT_DRIFT_SWITCHING_FEATURES = 'salary_age'
CONCEPT_DRIFT_SUDDEN = True

MODEL_NAME = 'ARF'

N_INNER_SAMPLES = 1
N_INTERVAL_LENGTH = 2000

TRAINING_METRIC_WINDOW = 100

MODEL_PARAMS = {
    'ARF': {
        'n_models': 25,
    }
}

if __name__ == "__main__":

    dataset_1 = get_dataset(dataset_name=DATASET_1_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES_1)
    dataset_2 = get_dataset(dataset_name=DATASET_2_NAME, random_seed=DATASET_RANDOM_SEED, shuffle=SHUFFLE_DATA, n_samples=N_SAMPLES_2)

    dataset = get_concept_drift_dataset(
        dataset_1_name=DATASET_1_NAME, dataset_2_name=DATASET_2_NAME,
        dataset_1=dataset_1, dataset_2=dataset_2,
        position=CONCEPT_DRIFT_POSITION, width=CONCEPT_DRIFT_WIDTH,
        features_to_switch=CONCEPT_DRIFT_SWITCHING_FEATURES, sudden_drift=CONCEPT_DRIFT_SUDDEN,
    )

    feature_names = dataset.feature_names
    n_samples = dataset.n_samples
    print(n_samples)
    stream = dataset.stream

    task = dataset.task

    loss_function = get_loss_function(task=task)
    rolling_training_metric, training_metric = get_training_metric(task=task, rolling_window=TRAINING_METRIC_WINDOW)

    model, model_function = get_model(
        model_name=MODEL_NAME, task=task, feature_names=feature_names, **MODEL_PARAMS[MODEL_NAME])

    incremental_sage = get_incremental_sage_explainer(
        feature_names=feature_names,
        model_function=model_function,
        loss_function=loss_function,
        n_inner_samples=N_INNER_SAMPLES
    )

    incremental_pfi = get_incremental_pfi_explainer(
        feature_names=feature_names,
        model_function=model_function,
        loss_function=loss_function,
        n_inner_samples=N_INNER_SAMPLES
    )

    interval_explainer = get_interval_sage_explainer(
        feature_names=feature_names,
        model_function=model_function,
        loss_function=loss_function,
        n_inner_samples=N_INNER_SAMPLES,
        interval_length=N_INTERVAL_LENGTH
    )

    # warm-up because of errors ------------------------------------------------------------------------------------
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        model.learn_one(x_i, y_i)
        if n > 10:
            break

    # Concept 1 ----------------------------------------------------------------------------------------------------

    model_performance_raw = []
    model_performance = []
    time_pfi = 0.
    time_sage = 0.
    time_interval = 0.
    time_learning = 0.

    sage_fi_values = []
    interval_fi_values = []
    pfi_fi_values = []

    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        # predicting
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model_performance_raw.append(training_metric.get())
        rolling_training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model_performance.append(rolling_training_metric.get())

        # sage inc
        start_time = time.time()
        fi_values = incremental_sage.explain_one(x_i, y_i)
        time_sage += time.time() - start_time
        sage_fi_values.append(fi_values)

        # pfi inc
        start_time = time.time()
        fi_values = incremental_pfi.explain_one(x_i, y_i)
        time_pfi += time.time() - start_time
        pfi_fi_values.append(fi_values)

        # sage int
        start_time = time.time()
        fi_values = interval_explainer.explain_one(x_i, y_i)
        time_interval += time.time() - start_time
        interval_fi_values.append(fi_values)

        # learning
        start_time = time.time()
        model.learn_one(x_i, y_i)
        time_learning += time.time() - start_time

        if n % 100 == 0:
            print(f"{n}: x_i         {x_i}\n"
                  f"{n}: performance {rolling_training_metric.get()}\n"
                  f"{n}: inc-sage    {incremental_sage.importance_values}")
        if n >= n_samples:
            break
