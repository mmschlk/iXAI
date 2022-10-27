from typing import Optional
from river.datasets import Insects as RiverDataset
from data.stream._base import StreamDataset


class Insects(StreamDataset):

    def __init__(
            self,
            variant="abrupt_balanced",
            n_samples: Optional[int] = None
    ):
        stream = RiverDataset()
        feature_names = ['Att1', 'Att2', 'Att3', 'Att4', 'Att5', 'Att6', 'Att7', 'Att8', 'Att9', 'Att10', 'Att11',
                              'Att12', 'Att13', 'Att14', 'Att15', 'Att16', 'Att17', 'Att18', 'Att19', 'Att20', 'Att21',
                              'Att22', 'Att23', 'Att24', 'Att25', 'Att26', 'Att27', 'Att28', 'Att29', 'Att30', 'Att31',
                              'Att32', 'Att33']
        cat_feature_names = []
        num_feature_names = ['Att1', 'Att2', 'Att3', 'Att4', 'Att5', 'Att6', 'Att7', 'Att8', 'Att9', 'Att10',
                                  'Att11', 'Att12', 'Att13', 'Att14', 'Att15', 'Att16', 'Att17', 'Att18', 'Att19',
                                  'Att20', 'Att21', 'Att22', 'Att23', 'Att24', 'Att25', 'Att26', 'Att27', 'Att28',
                                  'Att29', 'Att30', 'Att31', 'Att32', 'Att33']
        variant_sizes = {
            "abrupt_balanced": (52_848, 16_419_025),
            "abrupt_imbalanced": (355_275, 110_043_637),
            "gradual_balanced": (24_150, 7_503_750),
            "gradual_imbalanced": (143_323, 44_371_501),
            "incremental-abrupt_balanced": (79_986, 24_849_436),
            "incremental-abrupt_imbalanced": (452_044, 140_004_225),
            "incremental-reoccurring_balanced": (79_986, 24_849_092),
            "incremental-reoccurring_imbalanced": (452_044, 140_004_230),
            "incremental_balanced": (57_018, 17_713_574),
            "incremental_imbalanced": (452_044, 140_004_218),
            "out-of-control": (905_145, 277_777_854),
        }
        n_samples_variant = variant_sizes[variant][0]
        if n_samples is not None and n_samples > n_samples_variant:
            raise ValueError(f"The dataset with variant {variant} contains less samples, {n_samples_variant} "
                             f"than expected which was {n_samples}. Select less samples.")
        super().__init__(
            stream=stream,
            n_samples=n_samples,
            feature_names=feature_names,
            cat_feature_names=cat_feature_names,
            num_feature_names=num_feature_names,
            task=stream.task,
            n_features=len(feature_names),
            n_outputs=stream.n_outputs,
        )


if __name__ == "__main__":
    dataset = Insects()
    stream = dataset.stream
    print(stream.n_samples)
    for n, (x_i, y_i) in enumerate(stream):
        print(n, x_i, y_i)
        if n > 3:
            print("\n", dataset.feature_names, "\n", list(x_i.keys()))
            break
