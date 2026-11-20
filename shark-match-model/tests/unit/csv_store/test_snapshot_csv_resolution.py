"""Tests for resolving snapshot CSV directory paths during deserialization and ingestion."""

import os
import pickle
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.core import constants
from src.csv_store import _loader
from src.multimodal.fusion_dataset import FusionDataset
from src.multimodal.metadata_features import MetadataFeatureSpec
from src.individual_dataset import IndividualDataset
from src.core import utils


@pytest.fixture
def snapshot_csv_dir(tmp_path):
    """Create a temporary snapshot directory with media.csv and individuals.csv."""
    snap_dir = tmp_path / "inference-snapshots" / "test-snapshot-uuid"
    snap_dir.mkdir(parents=True)
    media_file = snap_dir / "media.csv"
    media_file.write_text(
        "id,hash,name,filename,source,metadata,location,time,type,quality\n"
        "1,h1,name1,file1.jpg,1,,loc1,100,left,1\n"
    )
    ind_file = snap_dir / "individuals.csv"
    ind_file.write_text(
        "id,name,sex,length,media,source,reference\n"
        "1,shark1,M,3.5,1,1,ref1\n"
    )
    return snap_dir


def test_resolve_relative_csv_dir_variants(snapshot_csv_dir, tmp_path):
    """resolve_relative_csv_dir correctly expands absolute, DB-relative, and snapshot-relative paths."""
    rel_snapshot = os.path.relpath(snapshot_csv_dir, tmp_path)
    with patch.object(constants, "BASE_DATA_DIR", str(tmp_path)):
        resolved = utils.resolve_relative_csv_dir(rel_snapshot)
        assert resolved == str(snapshot_csv_dir)

    assert utils.resolve_relative_csv_dir(None) is None
    assert utils.resolve_relative_csv_dir(str(snapshot_csv_dir)) == str(snapshot_csv_dir)


def test_individual_dataset_unpickle_resolves_snapshot_csv_dir(snapshot_csv_dir, tmp_path):
    """Unpickling IndividualDataset resolves relative snapshot csv_dir against BASE_DATA_DIR."""
    ds = IndividualDataset.__new__(IndividualDataset)
    ds.images_path = os.path.join(str(tmp_path), "images")
    ds.csv_dir = str(snapshot_csv_dir)
    ds.annotations_path = os.path.join(str(tmp_path), "annotations.tsv")

    with patch.object(constants, "BASE_DATA_DIR", str(tmp_path)):
        state = ds.__getstate__()
        assert state["csv_dir"] == os.path.relpath(snapshot_csv_dir, tmp_path)

        restored = pickle.loads(pickle.dumps(ds))
        assert restored.csv_dir == str(snapshot_csv_dir)
        assert restored.get_resolved_csv_dir() == str(snapshot_csv_dir)


def test_loader_set_db_dir_resolves_relative_snapshot(snapshot_csv_dir, tmp_path):
    """_loader.set_db_dir resolves relative snapshot directories and reads CSV records."""
    rel_snapshot = os.path.relpath(snapshot_csv_dir, tmp_path)
    with patch.object(constants, "BASE_DATA_DIR", str(tmp_path)):
        _loader.set_db_dir(rel_snapshot)
        try:
            media = _loader.load_media()
            assert len(media) == 1
            assert media[0].id == 1
            assert media[0].filename == "file1.jpg"

            individuals = _loader.load_individuals()
            assert len(individuals) == 1
            assert individuals[0].id == 1
            assert individuals[0].name == "shark1"
        finally:
            _loader.set_db_dir(None)


def test_fusion_dataset_with_relative_snapshot_csv_dir(snapshot_csv_dir, tmp_path):
    """FusionDataset initializes and loads media from a dataset with a relative snapshot csv_dir."""
    rel_snapshot = os.path.relpath(snapshot_csv_dir, tmp_path)

    mock_ds = MagicMock()
    mock_ds.csv_dir = rel_snapshot
    mock_ds.get_ids.return_value = [1]
    mock_ds.get_train_ids.return_value = [1]
    mock_ds.get_labels.return_value = torch.tensor([0], dtype=torch.long)

    spec = MetadataFeatureSpec()
    embeddings = torch.zeros((1, 16), dtype=torch.float32)

    with patch.object(constants, "BASE_DATA_DIR", str(tmp_path)):
        fusion_ds = FusionDataset(
            stored_dataset=mock_ds,
            image_embeddings=embeddings,
            spec=spec,
            fit_split="all",
        )
        assert fusion_ds.media_ids == [1]
        assert fusion_ds.filenames == ["file1.jpg"]


def test_resolve_relative_csv_dir_with_external_snapshot_root(tmp_path):
    """resolve_relative_csv_dir expands snapshot path when INFERENCE_SNAPSHOT_ROOT is external."""
    external_root = tmp_path / "external-snapshots"
    snap_dir = external_root / "custom-snap-uuid"
    snap_dir.mkdir(parents=True)

    with patch.object(constants, "INFERENCE_SNAPSHOT_ROOT", str(external_root)):
        resolved = utils.resolve_relative_csv_dir("inference-snapshots/custom-snap-uuid")
        assert resolved == str(snap_dir)

        resolved_direct = utils.resolve_relative_csv_dir("custom-snap-uuid")
        assert resolved_direct == str(snap_dir)


def test_individual_dataset_pickle_with_external_snapshot_root(tmp_path):
    """Unpickling IndividualDataset resolves relative snapshot against external snapshot root."""
    external_root = tmp_path / "external-snapshots"
    snap_dir = external_root / "custom-snap-uuid"
    snap_dir.mkdir(parents=True)

    ds = IndividualDataset.__new__(IndividualDataset)
    ds.images_path = os.path.join(str(tmp_path), "images")
    ds.csv_dir = str(snap_dir)
    ds.annotations_path = os.path.join(str(tmp_path), "annotations.tsv")

    with patch.object(constants, "INFERENCE_SNAPSHOT_ROOT", str(external_root)):
        state = ds.__getstate__()
        assert state["csv_dir"] == "inference-snapshots/custom-snap-uuid"

        restored = pickle.loads(pickle.dumps(ds))
        assert restored.csv_dir == str(snap_dir)



def test_relative_to_data_root_contains_only_descendants(tmp_path):
    """relative_to_data_root yields forward-slash descendants and rejects siblings and parents."""
    root = tmp_path / "root"
    assert utils.relative_to_data_root(root / "a" / "b", root) == "a/b"
    assert utils.relative_to_data_root(root, root) == "."
    assert utils.relative_to_data_root(tmp_path / "other", root) is None
    assert utils.relative_to_data_root(tmp_path, root) is None
    assert utils.relative_to_data_root(root / "..hidden", root) == "..hidden"


def test_portable_csv_dir_prefers_database_format_then_data_then_snapshots(tmp_path):
    """portable_csv_dir picks the first containing root and only knows snapshots when asked."""
    data_root = tmp_path / "data"
    db_dir = data_root / "database-format"
    snapshot_root = tmp_path / "elsewhere" / "snapshots"
    with patch.object(constants, "BASE_DATA_DIR", str(data_root)), \
            patch.object(constants, "DB_DATA_DIR", str(db_dir)), \
            patch.object(constants, "INFERENCE_SNAPSHOT_ROOT", str(snapshot_root)):
        assert utils.portable_csv_dir(db_dir / "thesis") == "thesis"
        assert utils.portable_csv_dir(data_root / "custom" / "csv") == "custom/csv"
        assert utils.portable_csv_dir(snapshot_root / "uuid") is None
        assert utils.portable_csv_dir(
            snapshot_root / "uuid", include_snapshots=True) == "inference-snapshots/uuid"
        assert utils.portable_csv_dir(tmp_path.parent / "outside", include_snapshots=True) is None


def test_individual_dataset_state_relativizes_every_stored_path(tmp_path):
    """Pickled state stores data-root paths relatively and leaves outside paths absolute."""
    outside = tmp_path.parent / "outside-data"
    ds = IndividualDataset.__new__(IndividualDataset)
    ds.csv_dir = str(outside / "csv")
    ds.images_path = str(tmp_path / "images" / "images")
    ds.annotations_path = str(outside / "annotations.tsv")

    with patch.object(constants, "BASE_DATA_DIR", str(tmp_path)), \
            patch.object(constants, "DB_DATA_DIR", str(tmp_path / "database-format")):
        state = ds.__getstate__()

    assert state["csv_dir"] == str(outside / "csv")
    assert state["images_path"] == "images/images"
    assert state["annotations_path"] == str(outside / "annotations.tsv")
