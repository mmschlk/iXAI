from experiments.get_data import get_dataset
# from experiments.get_explainer import get_explainer TODO write
# from experiments.get_model import get_incremental_model TODO write

DATASET_NAME = 'bank'


dataset = get_dataset(DATASET_NAME)
feature_names = dataset.feature_names
n_samples = dataset.n_samples

stream = dataset.stream

stream.task

# warm-up because of errors ------------------------------------------------------------------------------------
for (n, (x_i, y_i)) in enumerate(stream, start=1):
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
