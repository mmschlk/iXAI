import copy
import numpy as np
import pandas as pd

from river.ensemble import AdaptiveRandomForestClassifier
from river import metrics
from river.utils import Rolling
from river.datasets.synth import Agrawal
from ixai.storage import GeometricReservoirStorage
from ixai.utils.wrappers import RiverWrapper
from ixai.explainer.pdp import IncrementalPDP


RANDOM_SEED = 1
STREAM_LEN = 4000
if __name__ == "__main__":
    strm1_rand = np.random.normal(50,25,size = (STREAM_LEN,))
    strm1_rand1 = np.random.normal(100, 25, size=(STREAM_LEN,))
    y_stream1 = np.random.randint(0,2,(STREAM_LEN,))



    strm2_rand = np.random.normal(1000,25,size = (STREAM_LEN,))
    print(np.min(strm2_rand))
    strm2_rand1 = np.random.normal(150, 25, size=(STREAM_LEN,))
    y_stream2 = np.random.randint(0, 2, (STREAM_LEN,))



    for i in range(1):

        # Setup Data ---------------------------------------------------------------------------------------------------

        stream1_df = pd.DataFrame(pd.Series(strm1_rand, name='col1'))
        stream1_df['col2'] = strm1_rand1
        stream_1 = stream1_df.to_dict('records')
        feature_names = ['col1','col2']
        stream_1 = zip(stream_1, y_stream1)

        stream2_df = pd.DataFrame(pd.Series(strm2_rand, name='col1'))
        stream2_df['col2'] = strm2_rand1
        stream_2 = stream2_df.to_dict('records')
        stream_2 = zip(stream_2, y_stream1)

        # Model and training setup
        model = AdaptiveRandomForestClassifier(seed=RANDOM_SEED)
        model_function = RiverWrapper(model.predict_proba_one)
        loss_metric = metrics.Accuracy()
        training_metric = Rolling(metrics.Accuracy(), window_size=1000)

        # Setup explainers ---------------------------------------------------------------------------------------------

        # Instantiating objects
        storage = GeometricReservoirStorage(store_targets=False, size=100, constant_probability=0.8)
        # storage = SequenceStorage(store_targets=True)
        # imputer = DefaultImputer(model, values={'N_1': 5, 'N_2': 3, 'N_10': 2})
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
            if n % 1000 == 0:
                print(f"{n}: performance {training_metric.get()}\n")
                incremental_explainer.plot_pdp(title=f"PDP after {n} samples - 1")
            if n >= STREAM_LEN:
                break

        # Concept 2 ----------------------------------------------------------------------------------------------------

        for (n, (x_i, y_i)) in enumerate(stream_2, start=1):
            print(x_i)
            y_i_pred = model.predict_one(x_i)
            training_metric.update(y_true=y_i, y_pred=y_i_pred)
            model.learn_one(x_i, y_i)
            incremental_explainer.explain_one(
                x_i
            )
            if n % 1000 == 0:
                print(f"{n}: performance {training_metric.get()}\n")
                incremental_explainer.plot_pdp(title=f"PDP after {n} samples -2")
            if n >= STREAM_LEN:
                break
