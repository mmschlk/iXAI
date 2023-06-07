import copy
import numpy as np
import pandas as pd

from river.ensemble import AdaptiveRandomForestClassifier
from river import metrics
from river.utils import Rolling
from ixai.storage import GeometricReservoirStorage
from ixai.utils.wrappers import RiverWrapper
from ixai.explainer.pdp import IncrementalPDP, BatchPDP

STREAM_LEN = 10000
RANDOM_SEED = 1
N_SAMPLES = 4000

### Creating streams

# STREAM1 variables
VAR1_MEAN = 100
VAR1_STD = VAR1_MEAN * 0.1
VAR2_MEAN = 200
VAR2_STD = VAR2_MEAN * 0.1
ERROR_MEAN = 10
ERROR_STD = ERROR_MEAN * 0.1

#STREAM2 Variables
VAR4_MEAN = 100
VAR4_STD = VAR4_MEAN * 0.1
VAR5_MEAN = 200
VAR5_STD = VAR5_MEAN * 0.1

var1 = np.random.normal(VAR1_MEAN, VAR1_STD, size=(STREAM_LEN,))
var2 = np.random.normal(VAR2_MEAN, VAR2_STD, size=(STREAM_LEN,))
error = np.random.normal(ERROR_MEAN, ERROR_STD, size=(STREAM_LEN,))
coeff_var1 = 1
coeff_var2 = -0.5
thresh = 0.1
var3 = coeff_var1 * var1 + coeff_var2 * var2 + error
y = 1 / (1 + np.exp(-var3))
y[y>thresh] = 1
y[y<=thresh] = 0

var4 = np.random.normal(VAR4_MEAN, VAR4_STD, size=(STREAM_LEN,))
var5 = np.random.normal(VAR5_MEAN, VAR5_STD, size=(STREAM_LEN,))
coeff_var4 = -1
coeff_var5 = 0.5
thresh = 0.1
var6 = coeff_var4 * var4 + coeff_var5 * var5 + error
y1 = 1 / (1 + np.exp(-var6))
y1[y1>thresh] = 1
y1[y1<=thresh] = 0

# Setup Data ---------------------------------------------------------------------------------------------------

stream1_df = pd.DataFrame(pd.Series(var1, name='col1'))
stream1_df['col2'] = var2
stream_1 = stream1_df.to_dict('records')
stream_1 = zip(stream_1, y)

stream2_df = pd.DataFrame(pd.Series(var4, name='col1'))
stream2_df['col2'] = var5
stream_2 = stream2_df.to_dict('records')
stream_2 = zip(stream_2, y1)

feature_names = ['col1', 'col2']

# Model and training setup
model = AdaptiveRandomForestClassifier(seed=RANDOM_SEED)
model_function = RiverWrapper(model.predict_proba_one)
loss_metric = metrics.Accuracy()
training_metric = Rolling(metrics.Accuracy(), window_size=100)

# Setup explainers ---------------------------------------------------------------------------------------------

# Instantiating objects
storage = GeometricReservoirStorage(store_targets=False, size=100, constant_probability=0.8)

incremental_explainer = IncrementalPDP(
    model_function=model_function,
    feature_names=feature_names,
    gridsize=8,
    dynamic_setting=True,
    smoothing_alpha=0.1,
    pdp_feature='col1',
    storage=storage,
    storage_size=100
)

batch_explainer = BatchPDP(pdp_feature='col1',
                           gridsize=8,
                           model_function=model_function)

batch_explainer1 = BatchPDP(pdp_feature='col1',
                           gridsize=8,
                           model_function=model_function)
# warm-up because of errors ------------------------------------------------------------------------------------
for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
    model.learn_one(x_i, y_i)
    x_explain = copy.deepcopy(x_i)
    if n > 100:
        break

# Concept 1 ----------------------------------------------------------------------------------------------------

for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
    y_i_pred = model.predict_one(x_i)
    training_metric.update(y_true=y_i, y_pred=y_i_pred)
    model.learn_one(x_i, y_i)
    incremental_explainer.explain_one(
        x_i
    )
    batch_explainer.update_storage(x_i)
    if n % 1000 == 0:
        print(f"{n}: performance {training_metric.get()}\n")
        #incremental_explainer.plot_pdp(title=f"PDP after {n} samples - 1")
    if n >= N_SAMPLES:
        batch_explainer.explain_one(x_i)
        batch_explainer.plot_pdp("batch stream 1")
        incremental_explainer.plot_pdp("incremental stream 1")
        break

# Concept 2 ----------------------------------------------------------------------------------------------------

for (n, (x_i, y_i)) in enumerate(stream_2, start=1):
    y_i_pred = model.predict_one(x_i)
    training_metric.update(y_true=y_i, y_pred=y_i_pred)
    model.learn_one(x_i, y_i)
    incremental_explainer.explain_one(
        x_i
    )
    batch_explainer1.update_storage(x_i)
    if n % 1000 == 0:
        print(f"{n}: performance {training_metric.get()}\n")
        #incremental_explainer.plot_pdp(title=f"PDP after {n} samples -2")
    if n >= N_SAMPLES:
        batch_explainer1.explain_one(x_i)
        batch_explainer1.plot_pdp("batch stream 2")
        incremental_explainer.plot_pdp("incremental stream 2")
        break
