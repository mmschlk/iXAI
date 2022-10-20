import data.batch
import data.stream


DATASET_NAMES = ['bank', 'adult', 'bike', 'elec2', 'agrawal', 'stagger', 'planes', 'insects', 'credit_card']


__all__ = [
    "get_dataset"
]


def _get_batch_dataset(dataset_name, shuffle_dataset=False, random_seed=None):
    if dataset_name == 'bank':
        dataset = data.batch.BankMarketing(random_seed=random_seed, shuffle_dataset=shuffle_dataset)
    elif dataset_name == 'adult':
        dataset = data.batch.Adult(random_seed=random_seed, shuffle_dataset=shuffle_dataset)
    elif dataset_name == 'bike':
        dataset = data.batch.BikeSharing(random_seed=random_seed, shuffle_dataset=shuffle_dataset)
    else:
        raise NotImplementedError(f"The batch dataset with name {dataset_name} is not implemented.")
    return dataset


def get_dataset(dataset_name, shuffle=False, random_seed=None):
    dataset_name = dataset_name.split(' ')[0]
    if dataset_name in ['bank', 'adult', 'bike']:
        dataset = _get_batch_dataset(dataset_name=dataset_name, shuffle_dataset=shuffle, random_seed=random_seed)
    elif dataset_name == 'elec2':
        dataset = data.stream.Elec2()
    elif dataset_name == 'agrawal':
        classification_function = 1
        if len(dataset_name.split(' ')) > 1:
            classification_function = int(dataset_name.split('_')[1])
        dataset = data.stream.Agrawal(classification_function=classification_function, random_seed=random_seed)
    elif dataset_name == 'stagger':
        classification_function = 1
        if len(dataset_name.split(' ')) > 1:
            classification_function = int(dataset_name.split('_')[1])
        dataset = data.stream.Stagger(classification_function=classification_function, random_seed=random_seed)
    elif dataset_name == 'planes':
        dataset = data.stream.Planes2D(random_seed=random_seed)
    elif dataset_name == 'insects':
        variant = "abrupt_balanced"
        if len(dataset_name.split(' ')) > 1:
            variant = dataset_name.split(' ')[1]
        dataset = data.stream.Insects(variant=variant)
    elif dataset_name == 'credit_card':
        dataset = data.stream.CreditCard()
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not implemented. Implemented Datasets are {DATASET_NAMES}.")
    return dataset



