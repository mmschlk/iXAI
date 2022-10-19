import numpy as np
from river.utils import Rolling
from increment_explain.explainer.sage import IncrementalSageExplainer, BatchSageExplainer

from river.ensemble import AdaptiveRandomForestClassifier
from river import metrics
from increment_explain.utils import RiverToPredictionFunction
from river.datasets.synth import Agrawal
from data.batch import AGRAWAL_FEATURE_NAMES


from sage.utils import CrossEntropyLoss
from increment_explain.utils import ExponentialSmoothingTracker, WelfordTracker


def cross_entropy_loss(y_prediction, y_true):
    y_prediction = np.asarray([y_prediction])
    y_prediction = y_prediction.reshape((len(y_prediction), 1))
    y_true = np.asarray([y_true])
    y_true = y_true.reshape((len(y_true))).astype(int)
    return CROSS_ENTROPY(pred=y_prediction, target=y_true)


if __name__ == "__main__":
    diffs_inc = []
    sums_inc = []
    diffs_batch = []
    sums_batch = []

    CROSS_ENTROPY = CrossEntropyLoss(reduction='mean')
    RANDOM_SEED = 1
    CLASSIFICATION_FUNCTION = 1
    STREAM_LENGTH = 5000
    SMOOTHING_ALPHA = 0.001
    DYNAMIC_SETTING = True
    STATIC_MODEL = False

    for i in range(5):

        # Setup Data ---------------------------------------------------------------------------------------------------
        feature_names = list(AGRAWAL_FEATURE_NAMES)
        stream_1 = Agrawal(classification_function=CLASSIFICATION_FUNCTION)

        # Model and training setup
        model = AdaptiveRandomForestClassifier(seed=RANDOM_SEED)
        model_function = RiverToPredictionFunction(model, feature_names).predict
        metric = metrics.Accuracy()
        metric = Rolling(metric, window_size=500)

        loss_function = cross_entropy_loss

        # Concept 1 ----------------------------------------------------------------------------------------------------

        if STATIC_MODEL:
            for (n, (x_i, y_i)) in enumerate(stream_1):
                n += 1
                model.learn_one(x_i, y_i)
                y_i_pred = model.predict_one(x_i)
                metric.update(y_true=y_i, y_pred=y_i_pred)
                model.learn_one(x_i, y_i)
                if n >= STREAM_LENGTH:
                    break
                if n % 1000 == 0:
                    print(metric.get())

        incremental_explainer = IncrementalSageExplainer(
            model_function=model_function,
            feature_names=feature_names,
            loss_function=loss_function,
            n_inner_samples=1,
            smoothing_alpha=SMOOTHING_ALPHA,
            dynamic_setting=DYNAMIC_SETTING
        )

        if DYNAMIC_SETTING:
            full_loss = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
            marginal_prediction = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
            marginal_loss_dyn = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
            diff_dyn = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
        else:
            full_loss = WelfordTracker()
            marginal_prediction = WelfordTracker()
            marginal_loss_dyn = WelfordTracker()
            diff_dyn = WelfordTracker()
        diff_dyn_list = []

        x_data = []
        y_data = []
        for (n, (x_i, y_i)) in enumerate(stream_1):
            x_data.append(x_i)
            y_data.append(y_i)
            n += 1

            if not STATIC_MODEL:
                if n < 10:
                    model.learn_one(x_i, y_i)
                    continue
            if not STATIC_MODEL:
                model.learn_one(x_i, y_i)

            y_i_pred = model.predict_one(x_i)
            metric.update(y_true=y_i, y_pred=y_i_pred)
            y_i_pred = model_function(x_i)[0]
            marginal_prediction.update(y_i_pred)
            incremental_explainer.explain_one(x_i, y_i)

            y_i_arr = np.asarray([y_i])
            loss_i = loss_function(y_true=y_i_arr, y_prediction=y_i_pred)
            full_loss.update(loss_i)
            marginal_loss_i = loss_function(y_true=y_i_arr, y_prediction=np.asarray([marginal_prediction()]))
            marginal_loss_dyn.update(marginal_loss_i)
            diff_i = marginal_loss_i - loss_i
            diff_dyn.update(diff_i)
            diff_dyn_list.append(diff_i)

            if n >= STREAM_LENGTH:
                break
            if n % 1000 == 0:
                print(metric.get())

        marginal_loss = ExponentialSmoothingTracker(alpha=SMOOTHING_ALPHA)
        for y_i in y_data:
            marginal_loss_i = loss_function(y_true=y_i, y_prediction=np.asarray([marginal_prediction()]))
            marginal_loss.update(marginal_loss_i)

        diff = marginal_loss() - full_loss()

        incremental_sage_values = incremental_explainer.SAGE_values
        incremental_sage_values = list(incremental_sage_values.values())
        sum_incremental_sage_values = sum(incremental_sage_values)

        sums_inc.append(sum_incremental_sage_values)
        diffs_inc.append(diff)

        print()
        print("incremental SAGE")
        print("marginal prediction:", marginal_prediction())
        print("marginal loss:      ", marginal_loss())
        print("marginal loss dyn   ", marginal_loss_dyn())
        print("full loss:          ", full_loss())
        print("diff:               ", diff)
        print("diff dyn:           ", diff_dyn())
        print("diff dyn list mean  ", np.mean(diff_dyn_list))
        print("sum sage values:    ", sum_incremental_sage_values)

        batch_sage_explainer = BatchSageExplainer(
            model_fn=model_function,
            loss_function=loss_function,
            feature_names=feature_names,
            n_inner_samples=1,
            normalization_mode=None
        )
        batch_sage_values = batch_sage_explainer.explain_many(x_data=x_data, y_data=y_data, sub_sample_size=1)
        batch_sage_values = list(batch_sage_values.values())
        sum_batch_sage_values = sum(batch_sage_values)
        sums_batch.append(sum_batch_sage_values)

        full_loss = []
        y_trues = []
        predictions = []
        marginal_loss = []
        for n, (x_i, y_i) in enumerate(zip(x_data, y_data)):
            y_i_pred = model_function(x_i)
            y_i = y_i
            y_trues.append(y_i)
            predictions.append(y_i_pred)
            y_i = np.asarray([y_i])
            loss_i = loss_function(y_true=y_i, y_prediction=y_i_pred)
            full_loss.append(loss_i)
        marginal_prediction = np.mean(predictions, axis=0)
        for y_i_pred, y_i in zip(predictions, y_trues):
            y_i = np.asarray([y_i])
            loss_i = loss_function(y_prediction=marginal_prediction, y_true=y_i)
            marginal_loss.append(loss_i)
        marginal_loss = np.mean(marginal_loss)
        full_loss = np.mean(full_loss)
        diff = marginal_loss - full_loss

        print()
        print("batch SAGE")
        print("marginal prediction:", marginal_prediction)
        print("marginal loss:      ", marginal_loss)
        print("full loss:          ", full_loss)
        print("diff:               ", diff)
        print("sum sage values:    ", sum_batch_sage_values)

        diffs_batch.append(diff)

    print(diffs_inc)
    print(sums_inc)
    print(diffs_batch)
    print(sums_batch)
