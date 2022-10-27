from increment_explain.storage import GeometricReservoirStorage, UniformReservoirStorage
from increment_explain.imputer import MarginalImputer
from increment_explain.explainer.sage import IncrementalSageExplainer, BatchSageExplainer, IntervalSageExplainer
from increment_explain.explainer.pfi import IncrementalPFI


def _get_imputer_and_storage(
        model_function,
        reservoir_kind,
        reservoir_length,
        sample_strategy
):
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
        sampling_strategy=sample_strategy
    )
    return imputer, storage


def get_incremental_sage_explainer(
        model_function,
        feature_names,
        loss_function,
        reservoir_kind: str = 'geometric',
        sample_strategy: str = 'joint',
        reservoir_length: int = 1000,
        smoothing_alpha: float = 0.001,
        n_inner_samples: int = 1,
) -> IncrementalSageExplainer:

    imputer, storage = _get_imputer_and_storage(model_function, reservoir_kind, reservoir_length, sample_strategy)
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

    imputer, storage = _get_imputer_and_storage(model_function, reservoir_kind, reservoir_length, sample_strategy)
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
