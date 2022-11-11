from river.datasets.base import REG

from experiments.setup.data import get_dataset
from experiments.setup.explainer import get_incremental_sage_explainer, \
    get_batch_sage_explainer, get_interval_sage_explainer, get_imputer_and_storage
from experiments.setup.loss import get_loss_function, get_training_metric
from experiments.setup.model import get_model
from increment_explain.explainer.sage import IncrementalSageExplainer, BatchSageExplainer, IntervalSageExplainer

DEBUG = True

DATASET_NAME = 'adult'
DATASET_RANDOM_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DATASET_SHUFFLE = True
N_SAMPLES = 20000

MODEL_NAME = 'ARF'

N_INNER_SAMPLES: int = 1
N_INTERVAL_LENGTH: int = 2000
SMOOTHING_ALPHA: float = 0.001
FEATURE_REMOVAL_DISTRIBUTION: str = 'marginal joint'  # ['marginal joint', 'marginal product']
RESERVOIR_LENGTH: int = 100
RESERVOIR_KIND: str = 'geometric'  # ['geometric', 'uniform']

TRAINING_METRIC_WINDOW = 100

MODEL_PARAMS = {
    'ARF': {
        'n_models': 25,
    }
}

if __name__ == "__main__":

    inc_final_values = []
    bat_final_values = []

    for DATASET_RANDOM_SEED in DATASET_RANDOM_SEEDS:

        # Get Training Data --------------------------------------------------------------------------------------------
        dataset = get_dataset(DATASET_NAME, random_seed=DATASET_RANDOM_SEED, n_samples=N_SAMPLES, shuffle=DATASET_SHUFFLE)
        feature_names = dataset.feature_names
        n_samples = dataset.n_samples
        stream = dataset.stream

        # Get Model / Loss and Training Procedure ----------------------------------------------------------------------
        task = dataset.task
        loss_function = get_loss_function(task=task)
        rolling_training_metric, training_metric = get_training_metric(task=task, rolling_window=TRAINING_METRIC_WINDOW)
        model, model_function = get_model(
            model_name=MODEL_NAME, task=task, feature_names=feature_names, **MODEL_PARAMS[MODEL_NAME])

        # Get explainers -----------------------------------------------------------------------------------------------
        imputer, storage = get_imputer_and_storage(
            model_function=model_function,
            feature_removal_distribution=FEATURE_REMOVAL_DISTRIBUTION,
            reservoir_kind=RESERVOIR_KIND,
            reservoir_length=RESERVOIR_LENGTH,
            cat_feature_names=None,
            num_feature_names=None
        )

        incremental_explainer = IncrementalSageExplainer(
            model_function=model_function,
            loss_function=loss_function,
            imputer=imputer,
            storage=storage,
            feature_names=feature_names,
            smoothing_alpha=SMOOTHING_ALPHA,
            n_inner_samples=N_INNER_SAMPLES
        )

        batch_explainer = BatchSageExplainer(
            model_function=model_function,
            loss_function=loss_function,
            feature_names=feature_names,
            n_inner_samples=N_INNER_SAMPLES
        )

        interval_explainer = IntervalSageExplainer(
            model_function=model_function,
            loss_function=loss_function,
            feature_names=feature_names,
            n_inner_samples=N_INNER_SAMPLES,
            interval_length=N_INTERVAL_LENGTH
        )

        model_performance = []

        # Training Phase -----------------------------------------------------------------------------------------------
        if DEBUG:
            print(f"Starting Training for {n_samples}")
        for (n, (x_i, y_i)) in enumerate(stream, start=1):
            if task == REG:
                y_i_pred = model.predict_one(x_i)
            else:
                y_i_pred = model.predict_proba_one(x_i)
            rolling_training_metric.update(y_true=y_i, y_pred=y_i_pred)
            model.learn_one(x_i, y_i)
            if DEBUG and n % 1000 == 0:
                print(f"{n}: performance {rolling_training_metric.get()}\n")
            if n > n_samples:
                break

        # Get Explanation Data -----------------------------------------------------------------------------------------
        dataset = get_dataset(DATASET_NAME, random_seed=DATASET_RANDOM_SEED, n_samples=N_SAMPLES, shuffle=DATASET_SHUFFLE)
        n_samples = dataset.n_samples
        stream = dataset.stream

        # Explanation-Phase --------------------------------------------------------------------------------------------
        if DEBUG:
            print(f"Starting Explanation for {n_samples}")
        for (n, (x_i, y_i)) in enumerate(stream, start=1):
            inc_fi_values = incremental_explainer.explain_one(x_i, y_i)
            int_fi_values = interval_explainer.explain_one(x_i, y_i)
            batch_explainer.update_storage(x_i, y_i)
            bat_fi_values = batch_explainer.importance_values
            if DEBUG and n % 1000 == 0:
                print(f"{n}: inc-sage {incremental_explainer.importance_values}")
                print(f"{n}: int-sage {interval_explainer.importance_values}")
                print(f"{n}: bat-sage {batch_explainer.importance_values}\n")
            if n >= n_samples:
                bat_fi_values = batch_explainer.explain_one(x_i, y_i)
                int_fi_values = interval_explainer.explain_one(x_i, y_i, force_explain=True)
                break

        if DEBUG:
            print(f"final values: inc-sage {incremental_explainer.importance_values}")
            print(f"final values: int-sage {interval_explainer.importance_values}")
            print(f"final values: bat-sage {batch_explainer.importance_values}\n")

        inc_final_values.append(incremental_explainer.importance_values)
        bat_final_values.append(batch_explainer.importance_values)

    if DEBUG:
        print("all inc values:", inc_final_values)
        print("all bat values:", bat_final_values)
