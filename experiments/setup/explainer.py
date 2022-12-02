import typing

from increment_explain.storage import GeometricReservoirStorage, UniformReservoirStorage, TreeStorage
from increment_explain.imputer import MarginalImputer, TreeImputer
from increment_explain.explainer.sage import IncrementalSageExplainer, BatchSageExplainer, IntervalSageExplainer
from increment_explain.explainer.pfi import IncrementalPFI


def get_imputer_and_storage(
        model_function,
        feature_removal_distribution,
        reservoir_kind,
        reservoir_length,
        cat_feature_names=None,
        num_feature_names=None
):
    if feature_removal_distribution in {'marginal joint', 'marginal product'}:
        if reservoir_kind == 'geometric':
            storage = GeometricReservoirStorage(
                size=reservoir_length,
                store_targets=False
            )
        elif reservoir_kind == 'uniform':
            storage = UniformReservoirStorage(
                size=reservoir_length,
                store_targets=False
            )
        else:
            raise NotImplementedError(f"Only 'geometric', or 'uniform' storage implemented, not {reservoir_kind}.")

        imputer = MarginalImputer(
            model_function=model_function,
            storage_object=storage,
            sampling_strategy=feature_removal_distribution.split(" ")[1]
        )
    elif feature_removal_distribution == 'conditional':
        assert cat_feature_names is not None and num_feature_names is not None, \
            "cat. and num. feature names must be provided"

        storage = TreeStorage(
            cat_feature_names=cat_feature_names,
            num_feature_names=num_feature_names
        )

        imputer = TreeImputer(
            model_function=model_function,
            storage_object=storage,
            use_storage=True,
            direct_predict_numeric=True
        )
    else:
        raise NotImplementedError(f"Only 'marginal joint', 'marginal product', and 'conditional' feature removal "
                                  f"distributions are implemented. Not {feature_removal_distribution}")

    return imputer, storage


def get_incremental_sage_explainer(
        model_function,
        feature_names,
        loss_function,
        distribution_kind: str = 'marginal',
        reservoir_kind: str = 'geometric',
        reservoir_length: int = 1000,
        smoothing_alpha: float = 0.001,
        n_inner_samples: int = 1,
        cat_feature_names: typing.Optional[typing.List[str]] = None,
        num_feature_names: typing.Optional[typing.List[str]] = None

) -> IncrementalSageExplainer:

    imputer, storage = get_imputer_and_storage(
        model_function=model_function,
        feature_removal_distribution=distribution_kind,
        reservoir_kind=reservoir_kind,
        reservoir_length=reservoir_length,
        cat_feature_names=cat_feature_names,
        num_feature_names=num_feature_names
    )
    explainer = IncrementalSageExplainer(
        model_function=model_function,
        loss_function=loss_function,
        imputer=imputer,
        storage=storage,
        feature_names=feature_names,
        smoothing_alpha=smoothing_alpha,
        n_inner_samples=n_inner_samples
    )
    return explainer


def get_incremental_pfi_explainer(
        model_function,
        feature_names,
        loss_function,
        reservoir_kind: str = 'geometric',
        sample_strategy: str = 'joint',
        reservoir_length: int = 1000,
        smoothing_alpha: float = 0.001,
        n_inner_samples: int = 1,
) -> IncrementalPFI:

    imputer, storage = get_imputer_and_storage(model_function, 'marginal', reservoir_kind, reservoir_length, sample_strategy)
    explainer = IncrementalPFI(
        model_function=model_function,
        loss_function=loss_function,
        imputer=imputer,
        feature_names=feature_names,
        smoothing_alpha=smoothing_alpha,
        n_samples=n_inner_samples,
        storage=storage
    )
    return explainer


def get_batch_sage_explainer(
        model_function,
        loss_function,
        feature_names,
        n_inner_samples: int = 1
) -> BatchSageExplainer:

    explainer = BatchSageExplainer(
        model_function=model_function,
        loss_function=loss_function,
        feature_names=feature_names,
        n_inner_samples=n_inner_samples
    )
    return explainer


def get_interval_sage_explainer(
        model_function,
        loss_function,
        feature_names,
        n_inner_samples: int = 1,
        interval_length: int = 100
) -> IntervalSageExplainer:

    explainer = IntervalSageExplainer(
        model_function=model_function,
        loss_function=loss_function,
        feature_names=feature_names,
        n_inner_samples=n_inner_samples,
        interval_length=interval_length
    )
    return explainer
