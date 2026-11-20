"""P×K sampler for metric learning batches containing P identities with K images each."""

import random
import typing

import torch

import src.individual_dataset


class PKSampler(torch.utils.data.Sampler):
    """Samples batches of P identities with K images per identity, oversampling when needed."""
    def __init__(self, dataset: src.individual_dataset.IndividualDataset,
                 p: int, k: int):
        """Initialize PKSampler with target identity count P and per-identity sample count K."""
        super().__init__()
        self.dataset = dataset
        self.p = p
        self.k = k
        self.batch_size = p * k

        # Build label-to-indices mapping, using get_labels() to avoid eager data loading.
        self.label_to_indices: typing.Dict[int, typing.List[int]] = {}
        if hasattr(dataset, "get_labels"):
            labels_tensor = dataset.get_labels()
            for index, label_val in enumerate(labels_tensor.tolist()):
                self.label_to_indices.setdefault(label_val, []).append(index)
        else:
            # Preserve bounded index access for datasets without iterator support.
            # pylint: disable-next=consider-using-enumerate
            for index in range(len(dataset)):
                _, label = dataset[index]
                self.label_to_indices.setdefault(label.item(), []).append(index)

        self.labels = list(self.label_to_indices.keys())

        if len(self.labels) < self.p:
            raise ValueError(
                f"PKSampler requires at least P={self.p} distinct identities, "
                f"but dataset has only {len(self.labels)}."
            )

    def __len__(self) -> int:
        """Returns the total number of indices yielded per epoch across complete P-sized batches."""
        num_batches = len(self.labels) // self.p
        return num_batches * self.batch_size

    def __iter__(self) -> typing.Iterator[int]:
        """Yields dataset indices in batches of P identities with K samples each."""
        # Shuffle identity order for this epoch.
        shuffled_labels = self.labels.copy()
        random.shuffle(shuffled_labels)

        # Process identities in groups of P.
        for batch_start in range(0, len(shuffled_labels) - self.p + 1, self.p):
            batch_labels = shuffled_labels[batch_start:batch_start + self.p]

            for label in batch_labels:
                indices = self.label_to_indices[label]

                if len(indices) >= self.k:
                    # Sample K without replacement from available images.
                    sampled = random.sample(indices, self.k)
                else:
                    # Oversample with replacement to reach K.
                    sampled = random.choices(indices, k=self.k)

                yield from sampled
