import math
import random

import river.datasets.synth.concept_drift_stream
from river.metrics import Accuracy
from river.utils import Rolling
from river.ensemble import AdaptiveRandomForestClassifier
from river.tree import HoeffdingTreeClassifier
from river.linear_model import LogisticRegression
from river import neural_net as nn
from river.neighbors import KNNClassifier
from matplotlib import pyplot as plt
from river.datasets.base import BINARY_CLF

from data.stream._base import BatchStream
from experiments.setup.explainer import get_imputer_and_storage
from experiments.setup.loss import get_loss_function
from increment_explain.explainer import IncrementalPFI
from increment_explain.explainer.sage import IncrementalSageExplainer, BatchSageExplainer
from increment_explain.utils.converters import RiverToPredictionFunction
from increment_explain.utils.trackers import ExponentialSmoothingTracker
from increment_explain.visualization import FeatureImportancePlotter

N_NODE_FEATURES = 1


def _get_label_function_1_old(node_1, node_2):
    if node_1['node_1_interest_1'] >= 0.5 and node_2['node_2_interest_1'] >= 0.5:
        return 1
    if node_1['node_1_interest_2'] >= 0.5 and node_2['node_2_interest_2'] >= 0.5:
        return 1
    return 0


def _get_label_function_2_old(node_1, node_2):
    if node_1['node_1_interest_2'] >= 0.6 and node_2['node_2_interest_2'] >= 0.6:
        return 1
    if node_1['node_1_interest_3'] >= 0.5 and node_2['node_2_interest_3'] >= 0.5:
        return 1
    return 0


def _get_label_function_1(node_1, node_2):
    alpha = 30
    beta = 25
    gamma = 10
    z_1 = alpha * (node_1['node_1_interest_1'] * node_2['node_2_interest_1'])
    z_2 = beta * (node_1['node_1_interest_2'] * node_2['node_2_interest_2'])
    z_3 = gamma * node_1['graph_connectivity']
    p = 1 / (1 + math.exp(-(z_1 + z_2 + z_3)))
    random_number = random.random()
    if random_number < p:
        return 1, p
    return 0, p


def _get_label_function_2(node_1, node_2):
    alpha = 30
    beta = 25
    gamma = 10
    z_1 = alpha * (node_1['node_1_interest_2'] * node_2['node_2_interest_2'])
    z_2 = beta * (node_1['node_1_interest_3'] * node_2['node_2_interest_3'])
    z_3 = gamma * node_1['graph_connectivity']
    p = 1 / (1 + math.exp(-(z_1 + z_2 + z_3)))
    random_number = random.random()
    #print(int(random_number < p), p, node_1['node_1_interest_1'], node_2['node_2_interest_1'])
    #print(int(random_number < p), p, node_1['node_1_interest_3'], node_2['node_2_interest_3'])
    if random_number < p:
        return 1, p
    return 0, p


def get_sample(classification_function: int = 1, n_interests=3):
    node_1 = {'node_1_interest_' + str(i): random.uniform(-1, 1) for i in range(1, n_interests + 1)}
    node_1['node_1_att_1'] = random.randint(1, 3)
    node_1['graph_connectivity'] = random.uniform(-1, 1)
    node_1['node_1_att_2'] = random.randint(1, 3)
    node_2 = {'node_2_interest_' + str(i): random.uniform(-1, 1) for i in range(1, n_interests + 1)}
    node_2['node_2_att_1'] = random.randint(1, 3)

    #node_1.update({'interest_1': node_1['node_1_interest_1'] * node_2['node_2_interest_1'],'interest_2': node_1['node_1_interest_2'] * node_2['node_2_interest_2'],'interest_3': node_1['node_1_interest_3'] * node_2['node_2_interest_3']})
    #node_1['interest_1_interaction'] = node_1['node_1_interest_1'] * node_2['node_2_interest_1']
    #node_1['interest_2_interaction'] = node_1['node_1_interest_2'] * node_2['node_2_interest_2']
    #node_1['interest_2_interaction'] = node_1['node_1_interest_3'] * node_2['node_2_interest_3']


    if classification_function == 1:
        y_i, p_i = _get_label_function_1(node_1, node_2)
    elif classification_function == 2:
        y_i, p_i = _get_label_function_2(node_1, node_2)
    else:
        raise NotImplementedError("Only classificaiton function 1 and 2 implemented.")

    x_i = {**node_1, **node_2}

    return x_i, y_i

def print_probas(n=10_000, n_interests=3):
    ys = []
    ps = []
    for i in range(n):
        node_1 = {'node_1_interest_' + str(i): random.uniform(-0.5, 0.5) for i in range(1, n_interests + 1)}
        node_1['node_1_att_1'] = random.randint(1, 3)
        node_2 = {'node_2_interest_' + str(i): random.uniform(-0.5, 0.5) for i in range(1, n_interests + 1)}
        node_2['node_2_att_1'] = random.randint(1, 3)
        y_i, p_i = _get_label_function_1(node_1, node_2)
        ys.append(y_i)
        ps.append(p_i)

    plt.hist(ys)
    plt.title("Target")
    plt.show()

    plt.hist(ps)
    plt.title("Probability")
    plt.show()


def easy_problem(n=10_000, alpha=5, beta=5):
    ys = []
    ps = []
    xs = []
    x_1 = []
    x_2 = []
    dataset = []
    alpha = alpha
    for i in range(n):
        node_1_interest_1 = random.uniform(-1, 1)
        node_2_interest_1 = random.uniform(-1, 1)
        interest_1 = node_1_interest_1 * node_2_interest_1
        node_1_interest_2 = random.uniform(-1, 1)
        node_2_interest_2 = random.uniform(-1, 1)
        interest_2 = node_1_interest_2 * node_2_interest_2
        p = 1 / (1 + math.exp(-(alpha * interest_1 + beta * interest_2)))
        xs.append(interest_1)
        x_1.append(node_1_interest_1)
        x_2.append(node_2_interest_1)
        ps.append(p)
        y = 0
        if random.random() <= p:
            y = 1
        ys.append(y)
        dataset.append({"node_1_interest_1": node_1_interest_1, "node_2_interest_1": node_2_interest_1, #"interest_1": interest_1,
                        "node_1_interest_2": node_1_interest_2, "node_2_interest_2": node_2_interest_2, #"interest_2": interest_2
                        })

    plt.scatter(xs, ys)
    plt.title(f"Label, alpha: {alpha}")
    plt.xlabel("Feature 1 * Feature 2")
    plt.show()

    plt.scatter(xs, ps)
    plt.title(f"Probability, alpha: {alpha}")
    plt.xlabel("Feature 1 * Feature 2")
    plt.show()

    plt.scatter(x_1, ys)
    plt.title(f"Label, alpha: {alpha}")
    plt.xlabel("Feature 1")
    plt.show()

    plt.scatter(x_1, ps)
    plt.title(f"Probability, alpha: {alpha}")
    plt.xlabel("Feature 1")
    plt.show()

    return dataset, ys



if __name__ == "__main__":
    import sys



    #print_probas(n=10_000)

    N_SAMPLES = 50_000

    #dataset, ys = easy_problem(n=N_SAMPLES, alpha=30, beta=25)
    #stream = zip(dataset, ys)
    #feature_names = list(dataset[0].keys())

    DRIFT_POSITION = int(N_SAMPLES * 0.5)
    DRIFT_WIDTH = int(N_SAMPLES * 0.2)

    x_i, _ = get_sample(classification_function=1)
    feature_names = list(x_i.keys())

    model = AdaptiveRandomForestClassifier(n_models=10, binary_split=True, max_depth=10)
    #model = HoeffdingTreeClassifier(max_depth=10)
    #model = LogisticRegression()


    model_function = RiverToPredictionFunction(
        model,
        classification=True,
        feature_names=feature_names
    ).predict

    loss_function = get_loss_function(task=BINARY_CLF)

    imputer, storage = get_imputer_and_storage(
        model_function=model_function,
        feature_removal_distribution='marginal joint',
        reservoir_kind='geometric',
        reservoir_length=1000,
        cat_feature_names=[],
        num_feature_names=feature_names
    )

    incremental_sage = IncrementalSageExplainer(
        model_function=model_function,
        loss_function=loss_function,
        imputer=imputer,
        storage=storage,
        feature_names=feature_names,
        smoothing_alpha=0.001,
        n_inner_samples=5
    )

    incremental_pfi = IncrementalPFI(
        model_function=model_function,
        loss_function=loss_function,
        imputer=imputer,
        storage=storage,
        feature_names=feature_names,
        smoothing_alpha=0.001,
        n_samples=5
    )

    metric = Accuracy()
    metric = Rolling(metric, window_size=1000)

    plotter = FeatureImportancePlotter(
        feature_names=feature_names
    )

    model_loss_tracker = ExponentialSmoothingTracker(alpha=0.001)
    marginal_loss_tracker = ExponentialSmoothingTracker(alpha=0.001)
    model_loss = []
    marginal_loss = []

    performance = []

    stream_1 = iter([get_sample(classification_function=1) for i in range(N_SAMPLES*2)])
    stream_1 = BatchStream(stream=stream_1, n_features=len(feature_names), task=BINARY_CLF)
    stream_2 = iter([get_sample(classification_function=2) for i in range(N_SAMPLES*2)])
    stream_2 = BatchStream(stream=stream_2, n_features=len(feature_names), task=BINARY_CLF)
    stream = river.datasets.synth.concept_drift_stream.ConceptDriftStream(
        stream=stream_1, drift_stream=stream_2, position=DRIFT_POSITION, width=DRIFT_WIDTH)

    y_all = []

    for n, (x_i, y_i) in enumerate(stream):
        y_all.append(y_i)

        y_pred = model.predict_one(x_i)
        model.learn_one(x_i, y_i)


        if n > 100:
            y_i_pred = model_function(x_i)
            loss_i = loss_function(y_true=y_i, y_prediction=y_i_pred)
            model_loss_tracker.update(loss_i)
            model_loss.append({"loss": model_loss_tracker.get()})
            loss_marginal_i = loss_function(y_true=y_i, y_prediction=incremental_sage._marginal_prediction_tracker.get())
            marginal_loss_tracker.update(loss_marginal_i)
            marginal_loss.append({"loss": marginal_loss_tracker.get()})

            fi_values = incremental_sage.explain_one(x_i, y_i)
            plotter.update(fi_values, facet_name='sage')

            fi_values = incremental_pfi.explain_one(x_i, y_i)
            plotter.update(fi_values, facet_name='pfi')

        metric.update(y_i, y_pred)
        performance.append({'performance': metric.get()})

        if n % 1000 == 0:
            print(f"{n} performance {metric.get()}")
            print(f"{n} iSAGE       {incremental_sage.importance_values}")
            print(f"{n} iPFI        {incremental_pfi.importance_values}")

        if n >= N_SAMPLES:
            break

    plt.hist(y_all)
    plt.show()

    performance_kw_sage = {
        "y_min": 0, "y_max": 1, "y_label": "loss", "color_list": ["red", "black"],
        "line_names": ["loss"], "names_to_highlight": ['loss'],
        "line_styles": {"model_loss": "solid", "marginal_loss": "dashed"},
        "markevery": {"model_loss": 100, "marginal_loss": 100},
        "secondary_legends": [
            {"legend_props": {"loc": "center right", "ncol": 1, "fontsize": "small"},
             "legend_items": [("marginal loss", "dashed", "red"), ("model loss", "solid", "red")]}],
        "fill_between_props": [{'facet_1': 'model_loss', 'facet_2': 'marginal_loss', 'line_name_1': 'loss',
                                'line_name_2': 'loss', 'hatch': '///', 'color': 'red', 'alpha': 0.1}]
    }

    plotter.plot(
        figsize=(5, 10),
        model_performances={"model_loss": model_loss, "marginal_loss": marginal_loss},
        performance_kw=performance_kw_sage,
        title=f'Online Link Prediction based on common Interests',
        y_label='SAGE values',
        x_label='Samples',
        names_to_highlight=['node_1_interest_1', 'node_2_interest_1', 'node_1_interest_2',
                            'node_2_interest_2', 'node_1_interest_3', 'node_2_interest_3',
                            'graph_connectivity'
                            ],
        h_lines=[{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
        line_styles={'sage': '-', 'pfi': ''},
        legend_style={"fontsize": "small", "ncol": 4, "loc": "upper center"},
        markevery={'sage': 100, 'pfi': 100},
        y_ticks=[0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12],
        y_min=-0.02,
        y_max=0.14,
        color_list=['#ef27a6', '#4ea5d9', '#7d53de', '#44001A', '#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#44001A'],
        save_name="sage_probabilistic_edge_gradual_long.png"
    )

    # PFI --------------------------------------------------------------------------------------------------------------

    performance_kw = {"y_min": 0, "y_max": 1, "y_label": "accuracy", "color_list": ["red"],
                      "line_names": ["performance"], "names_to_highlight": ['performance']}

    plotter.plot(
        figsize=(5, 10),
        model_performances={"accuracy": performance},
        performance_kw=performance_kw,
        title=f'Online Link Prediction based on common Interests',
        y_label='PFI values',
        x_label='Samples',
        names_to_highlight=['node_1_interest_1', 'node_2_interest_1', 'node_1_interest_2',
                            'node_2_interest_2', 'node_1_interest_3', 'node_2_interest_3',
                            'graph_connectivity'
                            ],
        h_lines=[{'y': 0., 'ls': '--', 'c': 'grey', 'linewidth': 1}],
        line_styles={'sage': '', 'pfi': '-'},
        legend_style={"fontsize": "small", "ncol": 4, "loc": "upper center"},
        markevery={'sage': 100, 'pfi': 100},
        y_ticks=[0, 0.1, 0.2],
        y_min=-0.04,
        y_max=0.24,
        color_list=['#ef27a6', '#4ea5d9', '#7d53de', '#44001A', '#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#44001A'],
        save_name="pfi_probabilistic_edge_gradual_long.png"
    )



