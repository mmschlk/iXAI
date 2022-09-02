import lightgbm as lgb
from sklearn.model_selection import train_test_split

from data.data_sets import Adult
from explainer.shap import KernelSHAP

if __name__ == "__main__":
    # Load data
    dataset = Adult(random_seed=42, feature_encoder='ordinal', feature_scaler='standard')
    x_data = dataset.x_data
    y_data = dataset.y_data
    n_features = dataset.n_features
    feature_names = dataset.feature_names

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    d_train = lgb.Dataset(x_train, label=y_train)
    d_test = lgb.Dataset(x_test, label=y_test)

    # Train model
    params = {
        "max_bin": 512,
        "learning_rate": 0.05,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 10,
        "verbose": -1,
        "min_data": 100,
        "boost_from_average": True
    }
    model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)

    explainer = KernelSHAP(
        model=model,
        feature_names=feature_names,
        random_state=None
    )


    x_explain = x_test.loc[0]
    x_reference = [x_test.loc[1]]
    print(x_explain)

    explainer.explain_one(budget=1024, x=x_explain, reference_set=x_reference)


