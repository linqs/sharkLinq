"""Thin dataset wrapper pairing frozen image embeddings with engineered metadata features."""

import os
import typing

import torch
import torch.utils.data

import src.csv_store.media
import src.multimodal.metadata_features as metadata_features
from src.core import utils


class FusionDataset(torch.utils.data.Dataset):
    """Pairs frozen image embeddings with engineered metadata features for an individual ReID split."""

    def __init__(
            self,
            stored_dataset,
            image_embeddings: torch.Tensor,
            spec: metadata_features.MetadataFeatureSpec,
            split: typing.Optional[str] = None,
            norm_stats: typing.Optional[metadata_features.MetadataNormStats] = None,
            fit_split: str = "train",
    ):
        if fit_split not in ("train", "all"):
            raise ValueError(
                f"fit_split must be 'train' or 'all', got {fit_split!r}.")
        csv_dir = getattr(stored_dataset, "csv_dir", None)
        if csv_dir and not os.path.isabs(csv_dir):
            csv_dir = utils.resolve_relative_csv_dir(csv_dir)

        full_ids = list(stored_dataset.get_ids())
        media_records = src.csv_store.media.load_medias_with_individuals(
            csv_dir=csv_dir, ids=full_ids)
        media_by_id = {m.id: m for m in media_records}

        # Fit normalization stats on the specified split, defaulting to train to prevent leakage.
        if norm_stats is None:
            if fit_split == "train":
                fit_id_set = set(stored_dataset.get_train_ids())
                fit_medias = [
                    media_by_id[i] for i in full_ids
                    if i in media_by_id and i in fit_id_set
                ]
            else:  # "all"
                fit_medias = [media_by_id[i] for i in full_ids if i in media_by_id]
            norm_stats = metadata_features.fit_norm_stats(fit_medias, spec)
        self.spec = spec
        self.norm_stats = norm_stats
        self.fit_split = fit_split

        # Resolve the split's IDs in stored_dataset order.
        split_ids = self._resolve_split_ids(stored_dataset, split)
        full_ids_index = {mid: i for i, mid in enumerate(full_ids)}
        kept_indices = [
            full_ids_index[mid] for mid in split_ids
            if mid in full_ids_index and mid in media_by_id
        ]

        self.media_ids: typing.List[int] = [full_ids[i] for i in kept_indices]
        self.filenames: typing.List[str] = [media_by_id[mid].filename for mid in self.media_ids]
        self.image_embeddings: torch.Tensor = image_embeddings[kept_indices]
        self.labels: torch.Tensor = stored_dataset.get_labels()[kept_indices]
        self.metadata: torch.Tensor = torch.stack([
            metadata_features.engineer_features(media_by_id[mid], spec, norm_stats)
            for mid in self.media_ids
        ]) if self.media_ids else torch.empty(
            (0, metadata_features.feature_dim(spec, norm_stats)),
            dtype=torch.float32,
        )

        # Keep original train/test split IDs for downstream evaluation and retrieval tooling.
        self._train_ids_source = set(stored_dataset.get_train_ids())
        self._test_ids_source = set(stored_dataset.get_test_ids())

    def __len__(self) -> int:
        return len(self.media_ids)

    def __getitem__(self, index: int):
        return (
            self.image_embeddings[index],
            self.metadata[index],
            self.labels[index],
        )

    def get_ids(self) -> typing.List[int]:
        return list(self.media_ids)

    def get_filenames(self) -> typing.List[str]:
        return list(self.filenames)

    def get_labels(self) -> torch.Tensor:
        return self.labels

    def get_train_ids(self) -> typing.List[int]:
        return [mid for mid in self.media_ids if mid in self._train_ids_source]

    def get_test_ids(self) -> typing.List[int]:
        return [mid for mid in self.media_ids if mid in self._test_ids_source]

    def get_filename_by_id(self, individual_id: int) -> typing.Optional[str]:
        if individual_id not in self.media_ids:
            return None
        return self.filenames[self.media_ids.index(individual_id)]

    def get_label_by_id(self, individual_id: int):
        if individual_id not in self.media_ids:
            return None
        return self.labels[self.media_ids.index(individual_id)].item()

    def get_id_by_filename(self, filename: str) -> typing.Optional[int]:
        if filename not in self.filenames:
            return None
        return self.media_ids[self.filenames.index(filename)]

    def metadata_dim(self) -> int:
        return self.metadata.shape[1] if self.metadata.numel() else \
            metadata_features.feature_dim(self.spec, self.norm_stats)

    def image_dim(self) -> int:
        return self.image_embeddings.shape[1]

    @staticmethod
    def _resolve_split_ids(stored_dataset, split: typing.Optional[str]) -> typing.List[int]:
        if split is None:
            return list(stored_dataset.get_ids())
        if split == "train":
            return list(stored_dataset.get_train_ids())
        if split in ("test", "valid"):
            return list(stored_dataset.get_test_ids())
        raise ValueError(f"Unknown split '{split}'. Expected 'train', 'test', 'valid', or None.")
