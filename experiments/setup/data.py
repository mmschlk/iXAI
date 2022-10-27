from typing import Optional, Union

import data.batch
import data.stream
import data.stream.synth
from data.stream._base import get_concept_drift_stream
from data.stream._base import StreamDataset

DATASET_NAMES = ['bank', 'adult', 'bike', 'elec2', 'agrawal', 'stagger', 'planes', 'insects', 'credit_card']
BATCH_DATASETS = ['bank', 'adult', 'bike']
SYNTH_DATASETS = ['agrawal', 'stagger', 'planes']
STREAM_DATASETS = ['elec2', 'insects', 'credit_card', "stream_bike"]

__all__ = [
    "get_dataset",
    "get_concept_drift_dataset"
]


def _get_batch_dataset(dataset_name, shuffle_dataset=False, random_seed=None, n_samples=None):
    if dataset_name == 'bank':
        dataset = data.batch.BankMarketing(random_seed=random_seed, shuffle_dataset=shuffle_dataset, n_samples=n_samples)
    elif dataset_name == 'adult':
        dataset = data.batch.Adult(random_seed=random_seed, shuffle_dataset=shuffle_dataset, n_samples=n_samples)
    elif dataset_name == 'bike':
        dataset = data.batch.BikeSharing(random_seed=random_seed, shuffle_dataset=shuffle_dataset, n_samples=n_samples)
    else:
        raise NotImplementedError(f"The batch dataset with name {dataset_name} is not implemented.")
    return dataset


def get_dataset(dataset_name, shuffle=False, random_seed=None, n_samples=None):
    dataset_parameter = dataset_name.split(' ')[1] if len(dataset_name.split(' ')) > 1 else None
    dataset_name = dataset_name.split(' ')[0]
    if dataset_name in BATCH_DATASETS:
        dataset = _get_batch_dataset(dataset_name=dataset_name, shuffle_dataset=shuffle, random_seed=random_seed, n_samples=n_samples)
    elif dataset_name == 'elec2':
        dataset = data.stream.Elec2(n_samples=n_samples)
    elif dataset_name == 'stream_bike':
        dataset = data.stream.Bike(n_samples=n_samples)
    elif dataset_name == 'agrawal':
        if dataset_parameter is None:
            dataset_parameter = 1
        dataset = data.stream.synth.Agrawal(classification_function=int(dataset_parameter), random_seed=random_seed, n_samples=n_samples)
    elif dataset_name == 'stagger':
        if dataset_parameter is None:
            dataset_parameter = 1
        dataset = data.stream.synth.Stagger(classification_function=int(dataset_parameter), random_seed=random_seed, n_samples=n_samples)
    elif dataset_name == 'planes':
        dataset = data.stream.synth.Planes2D(random_seed=random_seed, n_samples=n_samples)
    elif dataset_name == 'insects':
        if dataset_parameter is None:
            dataset_parameter = "abrupt_balanced"
        dataset = data.stream.Insects(variant=dataset_parameter, n_samples=n_samples)
    elif dataset_name == 'credit_card':
        dataset = data.stream.CreditCard()
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not implemented. Implemented Datasets are {DATASET_NAMES}.")
    return dataset


def get_concept_drift_dataset(dataset_1_name: str, dataset_2_name: str,
                              dataset_1, dataset_2,
                              position: Union[float, int] = 0.5,
                              width: Optional[Union[float, int]] = 0.05, features_to_switch: Optional[str] = None,
                              sudden_drift: bool = False):
    dataset_name = dataset_1_name.split(' ')[0]
    stream_1 = dataset_1.stream
    if dataset_name in BATCH_DATASETS or dataset_name in STREAM_DATASETS:
        stream_2 = None
        n_samples = dataset_1.n_samples
    else:
        stream_2 = dataset_2.stream
        n_samples = dataset_1.n_samples + dataset_2.n_samples
    if position < 1:
        position = int(position * n_samples)
    if width is not None and width < 1:
        width = int(width * n_samples)
    concept_drift_stream = get_concept_drift_stream(
        stream=stream_1, drift_stream=stream_2,
        position=position, width=width, feature_remapping=features_to_switch, sudden_drift=sudden_drift)

    dataset = StreamDataset(
        stream=concept_drift_stream,
        n_samples=n_samples,
        feature_names=dataset_1.feature_names,
        cat_feature_names=dataset_1.cat_feature_names,
        num_feature_names=dataset_1.num_feature_names,
        task=dataset_1.task,
        n_features=dataset_1.n_features,
        n_outputs=dataset_1.n_outputs
    )

    return dataset
