"""Contract tests for atomic V1 embedding generations.

These tests exercise the shared filesystem boundary used for resolving,
publishing, retaining, and rolling back immutable embedding generations.
"""

import hashlib
import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


GENERATION_ID = "22222222-2222-4222-8222-222222222222"
OTHER_GENERATION_ID = "33333333-3333-4333-8333-333333333333"
SNAPSHOT_ID = "44444444-4444-4444-8444-444444444444"
CREATED_AT = "2026-08-31T12:00:00Z"
SHA256 = "a" * 64


class _StoredDataset:
    """Small pickle-compatible dataset used by generation contract tests."""

    def __init__(self, marker):
        self.marker = marker
        self._filenames = [f"{marker}.jpg"]
        self._ids = [1]
        self._labels = torch.tensor([1])

    def get_filenames(self):
        """Return filenames in embedding-row order."""
        return self._filenames

    def get_ids(self):
        """Return media identifiers in embedding-row order."""
        return self._ids

    def get_labels(self):
        """Return labels in embedding-row order."""
        return self._labels


def _generation_store():
    """Import the proposed shared generation-storage implementation."""
    return importlib.import_module("src.retrieval.generation_store")


def _embedding_pipeline():
    """Import the existing embedding coordinator used by CLI and server calls."""
    return importlib.import_module("src.retrieval.embeddings")


def _build_baseline_generation(marker):
    """Return a builder that writes one internally consistent baseline set."""
    def build(staging_dir, _previous_dir):
        """Write the artifacts normally produced by the existing pipeline."""
        staging_dir = Path(staging_dir)
        dataset = _StoredDataset(marker)
        torch.save(dataset, staging_dir / "stored_dataset.pt")
        torch.save(torch.tensor([[1.0, 0.0]]), staging_dir / "stored_embeddings.pt")
        torch.save(
            torch.tensor([[1.0, 0.0]]),
            staging_dir / "stored_centroid_embeddings.pt",
        )
        torch.save(torch.tensor([1]), staging_dir / "stored_centroid_labels.pt")
        torch.save({f"{marker}.jpg": []}, staging_dir / "stored_image_neighbors.pt")
        torch.save(
            {f"{marker}.jpg": []},
            staging_dir / "stored_centroid_neighbors.pt",
        )
        return 1

    return build


def _publish(store, embedding_set_root, marker):
    """Publish a minimal offline generation through the proposed shared API."""
    return store.publish_generation(
        embedding_set_root=embedding_set_root,
        embedding_set="test_embeddings",
        model_name="test_model",
        source="offline_annotations",
        snapshot_id=None,
        start_time=1_000.0,
        retrieval_mode="baseline",
        build=_build_baseline_generation(marker),
    )


def _write_resolvable_generation(
        embedding_set_root, generation_id=GENERATION_ID,
        manifest_generation_id=None):
    """Write a valid pointer and minimal manifest for resolver tests."""
    embedding_set_root.mkdir()
    generation_dir = embedding_set_root / "generations" / generation_id
    generation_dir.mkdir(parents=True)
    (generation_dir / "embeddings_metadata.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generation_id": manifest_generation_id or generation_id,
        "created_at": CREATED_AT,
        "embedding_set": "test_embeddings",
        "model_name": "test_model",
        "source": "offline_annotations",
        "snapshot_id": None,
        "files": {
            "embeddings_metadata.json": {"size_bytes": 2, "sha256": SHA256},
        },
    }
    (generation_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    pointer = {
        "schema_version": 1,
        "generation_id": generation_id,
        "published_at": CREATED_AT,
    }
    (embedding_set_root / "current.json").write_text(
        json.dumps(pointer), encoding="utf-8"
    )
    return generation_dir


def test_publish_writes_metadata_manifest_and_atomic_pointer(tmp_path):
    """A completed generation contains matching metadata, manifest, and pointer."""
    store = _generation_store()
    published = _publish(store, tmp_path / "test_embeddings", "new")

    pointer = json.loads((tmp_path / "test_embeddings" / "current.json").read_text())
    manifest = json.loads((published.directory / "manifest.json").read_text())
    metadata = json.loads(
        (published.directory / "embeddings_metadata.json").read_text()
    )

    assert pointer["generation_id"] == published.generation_id
    assert manifest["generation_id"] == published.generation_id
    assert manifest["embedding_set"] == "test_embeddings"
    assert manifest["model_name"] == "test_model"
    assert metadata["generation_id"] == published.generation_id
    assert metadata["snapshot_id"] is None
    assert metadata["num_images"] == 1
    assert set(manifest["files"]) == {
        path.name for path in published.directory.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    for filename, artifact in manifest["files"].items():
        contents = (published.directory / filename).read_bytes()
        assert artifact["size_bytes"] == len(contents)
        assert artifact["sha256"] == hashlib.sha256(contents).hexdigest()


@pytest.mark.parametrize("update_embeddings", [False, True])
def test_pipeline_publishes_resolvable_full_and_incremental_generations(
        tmp_path, monkeypatch, update_embeddings):
    """The real coordinator and publisher complete both primary build modes."""
    store = _generation_store()
    pipeline = _embedding_pipeline()
    embedding_set_root = tmp_path / "embedding-set"
    previous = None
    if update_embeddings:
        previous = _publish(store, embedding_set_root, "old")

    monkeypatch.setattr(pipeline.constants, "EMBEDDINGS_DIR", str(tmp_path))
    monkeypatch.setattr(
        pipeline.utils, "resolve_csv_dir", lambda _args: "/annotations/csv"
    )

    def build_artifacts(
            _model_name, _embeddings_name, _args, staging_dir,
            initial_csv_dir, retrieval_mode):
        """Write representative artifacts inside real publisher staging."""
        staging_dir = Path(staging_dir)
        assert retrieval_mode == "baseline"
        if update_embeddings:
            assert initial_csv_dir == "/annotations/csv"
            assert torch.load(
                staging_dir / "stored_dataset.pt", weights_only=False
            ).marker == "old"
        else:
            assert initial_csv_dir == "/annotations/csv"
            assert not any(staging_dir.iterdir())
        return _build_baseline_generation("new")(staging_dir, previous)

    monkeypatch.setattr(
        pipeline, "_build_generation_artifacts", build_artifacts
    )
    args = SimpleNamespace(
        update_embeddings=update_embeddings,
        annotations_version="annotations-v1",
        use_psl=False,
        use_multimodal=False,
        stacked=False,
    )

    published = pipeline.run_pipeline(
        "trained-model", "embedding-set", args, 2_000.0
    )
    resolved = store.resolve_active_generation(embedding_set_root)

    assert resolved.generation_id == published.generation_id
    assert torch.load(
        resolved.directory / "stored_dataset.pt", weights_only=False
    ).marker == "new"
    assert resolved.manifest.model_name == "trained-model"
    if previous is not None:
        assert torch.load(
            previous.directory / "stored_dataset.pt", weights_only=False
        ).marker == "old"


def test_pipeline_records_validated_snapshot_provenance(tmp_path, monkeypatch):
    """Server input identity reaches existing metadata and manifest writers."""
    pipeline = _embedding_pipeline()
    snapshot_dir = tmp_path / "snapshot-input"
    snapshot_dir.mkdir()

    monkeypatch.setattr(pipeline.constants, "EMBEDDINGS_DIR", str(tmp_path))

    def build_artifacts(
            _model_name, _embeddings_name, _args, staging_dir,
            initial_csv_dir, retrieval_mode):
        """Write representative artifacts after checking snapshot input."""
        assert initial_csv_dir == str(snapshot_dir)
        assert retrieval_mode == "baseline"
        return _build_baseline_generation("snapshot")(staging_dir, None)

    monkeypatch.setattr(
        pipeline, "_build_generation_artifacts", build_artifacts
    )
    args = SimpleNamespace(
        update_embeddings=True,
        annotations_version=None,
        use_psl=False,
        use_multimodal=False,
        stacked=False,
    )
    source_snapshot = SimpleNamespace(
        directory=snapshot_dir,
        snapshot_id=SNAPSHOT_ID,
    )

    published = pipeline.run_pipeline(
        "trained-model",
        "snapshot-embeddings",
        args,
        3_000.0,
        source_snapshot=source_snapshot,
    )

    manifest = published.manifest
    metadata = json.loads(
        (published.directory / "embeddings_metadata.json").read_text(
            encoding="utf-8")
    )
    assert manifest.source == "snapshot"
    assert manifest.snapshot_id == SNAPSHOT_ID
    assert metadata["snapshot_id"] == SNAPSHOT_ID


def test_pipeline_publishes_dataset_maintenance_as_a_new_generation(
        tmp_path, monkeypatch):
    """Dataset maintenance updates copied staging and preserves provenance."""
    store = _generation_store()
    pipeline = _embedding_pipeline()
    embedding_set_root = tmp_path / "embedding-set"
    original = _publish(store, embedding_set_root, "old")

    monkeypatch.setattr(pipeline.constants, "EMBEDDINGS_DIR", str(tmp_path))
    monkeypatch.setattr(
        pipeline.utils, "resolve_csv_dir", lambda _args: "/annotations/csv"
    )

    def regenerate(_model_name, _embeddings_name, _args, model_dir, csv_dir):
        """Represent the existing dataset refresh against copied staging."""
        assert csv_dir == "/annotations/csv"
        dataset_path = Path(model_dir) / "stored_dataset.pt"
        dataset = torch.load(dataset_path, weights_only=False)
        dataset.marker = "dataset-refreshed"
        torch.save(dataset, dataset_path)
        return len(dataset.get_filenames())

    monkeypatch.setattr(pipeline, "regenerate_dataset_only", regenerate)
    args = SimpleNamespace(
        update_embeddings=False,
        annotations_version="annotations-v2",
        use_psl=False,
        use_multimodal=False,
        stacked=False,
    )

    maintained = pipeline.run_pipeline(
        "replacement-model",
        "embedding-set",
        args,
        3_000.0,
        maintenance_mode="dataset",
    )

    assert maintained.generation_id != original.generation_id
    assert maintained.manifest.model_name == original.manifest.model_name
    assert torch.load(
        maintained.directory / "stored_dataset.pt", weights_only=False
    ).marker == "dataset-refreshed"
    assert torch.load(
        original.directory / "stored_dataset.pt", weights_only=False
    ).marker == "old"


def test_pipeline_publishes_required_suggestions_as_a_new_generation(
        tmp_path, monkeypatch):
    """Suggestion maintenance adds its cache without mutating active artifacts."""
    store = _generation_store()
    pipeline = _embedding_pipeline()
    embedding_set_root = tmp_path / "embedding-set"
    original = _publish(store, embedding_set_root, "old")

    monkeypatch.setattr(pipeline.constants, "EMBEDDINGS_DIR", str(tmp_path))

    def store_suggestions(_embeddings_name, model_dir, retrieval_mode,
                          raise_on_error):
        """Represent required suggestion generation in copied staging."""
        assert retrieval_mode == "baseline"
        assert raise_on_error
        torch.save({"review": []}, Path(model_dir) / "suggestions.pt")

    monkeypatch.setattr(
        pipeline, "generate_and_store_suggestions", store_suggestions
    )
    args = SimpleNamespace(
        update_embeddings=False,
        annotations_version=None,
        use_psl=False,
        use_multimodal=False,
        stacked=False,
    )

    maintained = pipeline.run_pipeline(
        "replacement-model",
        "embedding-set",
        args,
        4_000.0,
        maintenance_mode="suggestions",
    )

    assert maintained.generation_id != original.generation_id
    assert maintained.manifest.model_name == original.manifest.model_name
    assert (maintained.directory / "suggestions.pt").is_file()
    assert not (original.directory / "suggestions.pt").exists()


def test_required_suggestion_failure_preserves_active_pipeline_generation(
        tmp_path, monkeypatch):
    """A maintenance failure cannot publish its copied staging generation."""
    store = _generation_store()
    pipeline = _embedding_pipeline()
    embedding_set_root = tmp_path / "embedding-set"
    original = _publish(store, embedding_set_root, "old")
    original_pointer = (embedding_set_root / "current.json").read_bytes()

    monkeypatch.setattr(pipeline.constants, "EMBEDDINGS_DIR", str(tmp_path))

    def fail_suggestions(*_args, **_kwargs):
        """Inject the required suggestion cache failure."""
        raise RuntimeError("injected suggestion failure")

    monkeypatch.setattr(
        pipeline, "generate_and_store_suggestions", fail_suggestions
    )
    args = SimpleNamespace(
        update_embeddings=False,
        annotations_version=None,
        use_psl=False,
        use_multimodal=False,
        stacked=False,
    )

    with pytest.raises(RuntimeError, match="injected suggestion failure"):
        pipeline.run_pipeline(
            "trained-model",
            "embedding-set",
            args,
            5_000.0,
            maintenance_mode="suggestions",
        )

    assert (embedding_set_root / "current.json").read_bytes() == original_pointer
    assert store.resolve_active_generation(
        embedding_set_root
    ).generation_id == original.generation_id
    assert not any(
        path.name.startswith(".staging-")
        for path in (embedding_set_root / "generations").iterdir()
    )


@pytest.mark.parametrize(
    "stage_function",
    [
        "_write_embeddings_metadata",
        "_validate_staged_generation",
        "_write_generation_manifest",
        "_finalize_generation_directory",
        "_replace_current_pointer",
    ],
)
def test_failure_before_pointer_publication_preserves_active_generation(
        tmp_path, monkeypatch, stage_function):
    """Failure at any pre-publication stage leaves the active pointer unchanged."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    original = _publish(store, embedding_set_root, "old")
    original_pointer = (embedding_set_root / "current.json").read_bytes()

    def fail_stage(*_args, **_kwargs):
        """Inject a deterministic publish-stage failure."""
        raise OSError(f"injected failure in {stage_function}")

    monkeypatch.setattr(store, stage_function, fail_stage)
    with pytest.raises(OSError, match="injected failure"):
        _publish(store, embedding_set_root, "new")

    assert (embedding_set_root / "current.json").read_bytes() == original_pointer
    resolved = store.resolve_active_generation(embedding_set_root)
    assert resolved.generation_id == original.generation_id
    assert torch.load(
        resolved.directory / "stored_dataset.pt", weights_only=False
    ).marker == "old"


def test_builder_failure_preserves_active_generation_and_cleans_staging(tmp_path):
    """Artifact-construction failure cannot change active state or leak staging."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    original = _publish(store, embedding_set_root, "old")

    def fail_build(_staging_dir, _previous_dir):
        """Represent a model or artifact writer failing during construction."""
        raise RuntimeError("injected build failure")

    with pytest.raises(RuntimeError, match="injected build failure"):
        store.publish_generation(
            embedding_set_root=embedding_set_root,
            embedding_set="test_embeddings",
            model_name="test_model",
            source="offline_annotations",
            snapshot_id=None,
            start_time=1_000.0,
            retrieval_mode="baseline",
            build=fail_build,
        )

    assert store.resolve_active_generation(
        embedding_set_root
    ).generation_id == original.generation_id
    generations_dir = embedding_set_root / "generations"
    assert not any(path.name.startswith(".staging-") for path in generations_dir.iterdir())


def test_full_generation_does_not_seed_prior_artifacts(tmp_path):
    """A full rebuild receives empty staging while retaining prior identity."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    original = _publish(store, embedding_set_root, "old")

    def replacement_build(staging_dir, previous_dir):
        """Verify the full-build boundary before creating replacement files."""
        assert previous_dir == original.directory
        assert not any(staging_dir.iterdir())
        return _build_baseline_generation("new")(staging_dir, previous_dir)

    published = store.publish_generation(
        embedding_set_root=embedding_set_root,
        embedding_set="test_embeddings",
        model_name="test_model",
        source="offline_annotations",
        snapshot_id=None,
        start_time=1_000.0,
        retrieval_mode="baseline",
        build=replacement_build,
        seed_previous=False,
    )

    assert torch.load(
        published.directory / "stored_dataset.pt", weights_only=False
    ).marker == "new"
    assert torch.load(
        original.directory / "stored_dataset.pt", weights_only=False
    ).marker == "old"


def test_maintenance_publication_inherits_active_manifest_identity(tmp_path):
    """A selected-artifact update retains its active generation provenance."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    original = store.publish_generation(
        embedding_set_root=embedding_set_root,
        embedding_set="test_embeddings",
        model_name="snapshot-model",
        source="snapshot",
        snapshot_id=SNAPSHOT_ID,
        start_time=1_000.0,
        retrieval_mode="baseline",
        build=_build_baseline_generation("old"),
    )

    def maintenance_build(staging_dir, previous_dir):
        """Confirm maintenance starts from the active immutable generation."""
        assert previous_dir == original.directory
        assert torch.load(
            staging_dir / "stored_dataset.pt", weights_only=False
        ).marker == "old"
        return 1

    maintained = store.publish_generation(
        embedding_set_root=embedding_set_root,
        embedding_set="replacement-name",
        model_name="replacement-model",
        source="offline_annotations",
        snapshot_id=None,
        start_time=2_000.0,
        retrieval_mode="baseline",
        build=maintenance_build,
        inherit_previous_identity=True,
    )

    assert maintained.manifest.embedding_set == "test_embeddings"
    assert maintained.manifest.model_name == "snapshot-model"
    assert maintained.manifest.source == "snapshot"
    assert maintained.manifest.snapshot_id == SNAPSHOT_ID
    metadata = json.loads(
        (maintained.directory / "embeddings_metadata.json").read_text()
    )
    assert metadata["snapshot_id"] == SNAPSHOT_ID


def test_real_row_count_validation_failure_preserves_active_generation(tmp_path):
    """Inconsistent dataset and embedding counts cannot replace active state."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    original = _publish(store, embedding_set_root, "old")
    original_pointer = (embedding_set_root / "current.json").read_bytes()

    def inconsistent_build(staging_dir, previous_dir):
        """Write one image while reporting a two-image generation."""
        _build_baseline_generation("invalid")(staging_dir, previous_dir)
        return 2

    with pytest.raises(store.GenerationValidationError, match="rows do not match"):
        store.publish_generation(
            embedding_set_root=embedding_set_root,
            embedding_set="test_embeddings",
            model_name="test_model",
            source="offline_annotations",
            snapshot_id=None,
            start_time=1_000.0,
            retrieval_mode="baseline",
            build=inconsistent_build,
        )

    assert (embedding_set_root / "current.json").read_bytes() == original_pointer
    assert store.resolve_active_generation(
        embedding_set_root
    ).generation_id == original.generation_id


def test_retention_failure_does_not_invalidate_newly_published_generation(
        tmp_path, monkeypatch):
    """Post-publication cleanup failure leaves the new complete pointer active."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    _publish(store, embedding_set_root, "old")

    def fail_retention(*_args, **_kwargs):
        """Inject failure after the pointer has already been replaced."""
        raise OSError("injected retention failure")

    monkeypatch.setattr(store, "_prune_generations", fail_retention)
    published = _publish(store, embedding_set_root, "new")

    resolved = store.resolve_active_generation(embedding_set_root)
    assert resolved.generation_id == published.generation_id
    assert torch.load(
        resolved.directory / "stored_dataset.pt", weights_only=False
    ).marker == "new"


def test_retention_keeps_active_and_one_prior_generation(tmp_path):
    """A third publication removes only the generation older than its prior."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    first = _publish(store, embedding_set_root, "first")
    second = _publish(store, embedding_set_root, "second")
    third = _publish(store, embedding_set_root, "third")

    assert not first.directory.exists()
    assert second.directory.is_dir()
    assert third.directory.is_dir()
    retained_ids = {
        path.name
        for path in (embedding_set_root / "generations").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    assert retained_ids == {second.generation_id, third.generation_id}


def test_reader_holds_one_resolved_generation_during_concurrent_publish(tmp_path):
    """A reader that resolved the old generation never mixes in new artifacts."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    old = _publish(store, embedding_set_root, "old")
    reader_resolved = threading.Event()
    publish_complete = threading.Event()
    observed = {}

    def read_generation():
        """Resolve once, wait for publication, then read only the held directory."""
        resolved = store.resolve_active_generation(embedding_set_root)
        reader_resolved.set()
        assert publish_complete.wait(timeout=5)
        observed["generation_id"] = resolved.generation_id
        observed["marker"] = torch.load(
            resolved.directory / "stored_dataset.pt", weights_only=False
        ).marker

    reader = threading.Thread(target=read_generation)
    reader.start()
    assert reader_resolved.wait(timeout=5)
    new = _publish(store, embedding_set_root, "new")
    publish_complete.set()
    reader.join(timeout=5)

    assert not reader.is_alive()
    assert observed == {"generation_id": old.generation_id, "marker": "old"}
    assert store.resolve_active_generation(embedding_set_root).generation_id == new.generation_id


def test_reader_resolves_old_generation_while_replacement_is_still_staged(
        tmp_path):
    """Unpublished staging cannot become visible through the active resolver."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    old = _publish(store, embedding_set_root, "old")
    original_pointer = (embedding_set_root / "current.json").read_bytes()
    build_entered = threading.Event()
    release_build = threading.Event()
    published = []
    errors = []

    def blocking_build(staging_dir, previous_dir):
        """Hold a complete replacement in staging before publication."""
        try:
            result = _build_baseline_generation("new")(
                staging_dir, previous_dir
            )
            build_entered.set()
            assert release_build.wait(timeout=5)
            return result
        except Exception as error:  # pylint: disable=broad-exception-caught
            errors.append(error)
            raise

    def publish_replacement():
        """Publish in a thread so the active resolver can run mid-build."""
        try:
            published.append(store.publish_generation(
                embedding_set_root=embedding_set_root,
                embedding_set="test_embeddings",
                model_name="test_model",
                source="offline_annotations",
                snapshot_id=None,
                start_time=1_000.0,
                retrieval_mode="baseline",
                build=blocking_build,
                seed_previous=False,
            ))
        except Exception as error:  # pylint: disable=broad-exception-caught
            errors.append(error)

    publisher = threading.Thread(target=publish_replacement)
    publisher.start()
    assert build_entered.wait(timeout=5)

    while_staged = store.resolve_active_generation(embedding_set_root)
    assert while_staged.generation_id == old.generation_id
    assert torch.load(
        while_staged.directory / "stored_dataset.pt", weights_only=False
    ).marker == "old"
    assert (embedding_set_root / "current.json").read_bytes() == original_pointer

    release_build.set()
    publisher.join(timeout=5)

    assert not publisher.is_alive()
    assert not errors
    assert len(published) == 1
    assert store.resolve_active_generation(
        embedding_set_root
    ).generation_id == published[0].generation_id


def test_publishers_serialize_build_callbacks_with_file_lock(tmp_path):
    """Two publishers cannot construct generations concurrently for one set."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    errors = []

    def blocking_build(staging_dir, previous_dir):
        """Hold the first publication inside its build callback."""
        first_entered.set()
        assert release_first.wait(timeout=5)
        return _build_baseline_generation("first")(staging_dir, previous_dir)

    def second_build(staging_dir, previous_dir):
        """Record when the second publication passes the shared lock."""
        second_entered.set()
        return _build_baseline_generation("second")(staging_dir, previous_dir)

    def publish(build):
        """Run one publisher thread and retain unexpected exceptions."""
        try:
            store.publish_generation(
                embedding_set_root=embedding_set_root,
                embedding_set="test_embeddings",
                model_name="test_model",
                source="offline_annotations",
                snapshot_id=None,
                start_time=1_000.0,
                retrieval_mode="baseline",
                build=build,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            errors.append(error)

    first_thread = threading.Thread(target=publish, args=(blocking_build,))
    second_thread = threading.Thread(target=publish, args=(second_build,))
    first_thread.start()
    assert first_entered.wait(timeout=5)
    second_thread.start()
    assert not second_entered.wait(timeout=0.2)
    release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not errors
    assert second_entered.is_set()


def test_directory_sync_failure_after_generation_rename_preserves_active(
        tmp_path, monkeypatch):
    """A failed generation-directory sync removes only the unpublished output."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    original = _publish(store, embedding_set_root, "old")
    original_pointer = (embedding_set_root / "current.json").read_bytes()
    real_sync = store._sync_directory  # pylint: disable=protected-access

    def fail_generations_sync(directory):
        """Fail only when syncing the finalized generations directory."""
        if Path(directory).name == "generations":
            raise OSError("injected generations directory sync failure")
        return real_sync(directory)

    monkeypatch.setattr(store, "_sync_directory", fail_generations_sync)
    with pytest.raises(OSError, match="injected generations"):
        _publish(store, embedding_set_root, "new")

    assert (embedding_set_root / "current.json").read_bytes() == original_pointer
    completed = [
        path for path in (embedding_set_root / "generations").iterdir()
        if not path.name.startswith(".staging-")
    ]
    assert completed == [original.directory]


def test_root_sync_failure_after_pointer_replace_keeps_new_generation(
        tmp_path, monkeypatch):
    """Once the pointer is visible, a later sync error cannot delete its target."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    _publish(store, embedding_set_root, "old")
    real_sync = store._sync_directory  # pylint: disable=protected-access

    def fail_root_sync(directory):
        """Fail only after current.json has been atomically replaced."""
        if Path(directory) == embedding_set_root:
            raise OSError("injected pointer directory sync failure")
        return real_sync(directory)

    monkeypatch.setattr(store, "_sync_directory", fail_root_sync)
    published = _publish(store, embedding_set_root, "new")

    resolved = store.resolve_active_generation(embedding_set_root)
    assert resolved.generation_id == published.generation_id
    assert published.directory.is_dir()


def test_legacy_flat_layout_is_used_only_when_pointer_is_absent(tmp_path):
    """Flat artifacts are a fallback for absence, not for an invalid pointer."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    embedding_set_root.mkdir()
    (embedding_set_root / "stored_dataset.pt").write_bytes(b"legacy")

    legacy = store.resolve_active_generation(embedding_set_root)
    assert legacy.directory == embedding_set_root
    assert legacy.generation_id is None

    (embedding_set_root / "current.json").write_text("{}", encoding="utf-8")
    with pytest.raises(store.GenerationValidationError):
        store.resolve_active_generation(embedding_set_root)


def test_publication_migrates_legacy_flat_artifacts_through_shared_staging(
        tmp_path):
    """A legacy flat set can seed its first immutable active generation."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    embedding_set_root.mkdir()
    _build_baseline_generation("legacy")(embedding_set_root, None)

    legacy = store.resolve_active_generation(embedding_set_root)
    assert legacy.is_legacy

    def migrate(staging_dir, previous_dir):
        """Confirm the existing seeding path copied every legacy artifact."""
        assert Path(previous_dir) == embedding_set_root
        dataset = torch.load(
            Path(staging_dir) / "stored_dataset.pt", weights_only=False
        )
        assert dataset.marker == "legacy"
        return len(dataset.get_filenames())

    migrated = store.publish_generation(
        embedding_set_root=embedding_set_root,
        embedding_set="test_embeddings",
        model_name="test_model",
        source="offline_annotations",
        snapshot_id=None,
        start_time=1_000.0,
        retrieval_mode="baseline",
        build=migrate,
    )

    resolved = store.resolve_active_generation(embedding_set_root)
    assert not resolved.is_legacy
    assert resolved.generation_id == migrated.generation_id
    assert torch.load(
        resolved.directory / "stored_dataset.pt", weights_only=False
    ).marker == "legacy"


def test_active_pointer_resolves_one_valid_generation(tmp_path):
    """A valid pointer and matching manifest resolve the immutable directory."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    generation_dir = _write_resolvable_generation(embedding_set_root)

    resolved = store.resolve_active_generation(embedding_set_root)

    assert resolved.directory == generation_dir
    assert resolved.generation_id == GENERATION_ID
    assert resolved.manifest.generation_id == GENERATION_ID
    assert resolved.pointer.generation_id == GENERATION_ID
    assert not resolved.is_legacy


def test_pointer_to_missing_generation_fails_without_legacy_fallback(tmp_path):
    """An existing pointer cannot fall back when its generation is missing."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    _write_resolvable_generation(embedding_set_root)
    generation_dir = embedding_set_root / "generations" / GENERATION_ID
    generation_dir.rename(embedding_set_root / "unreferenced-generation")

    with pytest.raises(store.GenerationValidationError, match="directory is missing"):
        store.resolve_active_generation(embedding_set_root)


def test_manifest_identifier_must_match_pointer(tmp_path):
    """Resolver rejects a structurally valid manifest for another generation."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    _write_resolvable_generation(
        embedding_set_root,
        manifest_generation_id=OTHER_GENERATION_ID,
    )

    with pytest.raises(store.GenerationValidationError, match="does not match"):
        store.resolve_active_generation(embedding_set_root)


def test_symlinked_pointer_is_invalid_not_absent(tmp_path):
    """A symlink at the pointer path is rejected instead of treated as absent."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    embedding_set_root.mkdir()
    pointer_target = tmp_path / "pointer.json"
    pointer_target.write_text("{}", encoding="utf-8")
    (embedding_set_root / "current.json").symlink_to(pointer_target)

    with pytest.raises(store.GenerationValidationError, match="nonsymlinked"):
        store.resolve_active_generation(embedding_set_root)


def test_rollback_repoints_to_valid_prior_generation_without_copying(tmp_path):
    """Rollback validates and republishes the retained prior generation pointer."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    old = _publish(store, embedding_set_root, "old")
    new = _publish(store, embedding_set_root, "new")
    generation_paths_before = set(
        (embedding_set_root / "generations").iterdir()
    )

    rolled_back = store.rollback_generation(embedding_set_root)

    assert rolled_back.generation_id == old.generation_id
    assert rolled_back.directory == old.directory
    assert new.directory.is_dir()
    assert set(
        (embedding_set_root / "generations").iterdir()
    ) == generation_paths_before
    assert torch.load(
        rolled_back.directory / "stored_dataset.pt", weights_only=False
    ).marker == "old"


def test_rollback_rejects_corrupt_prior_and_preserves_active_pointer(tmp_path):
    """Hash-invalid retained artifacts cannot become active through rollback."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    old = _publish(store, embedding_set_root, "old")
    new = _publish(store, embedding_set_root, "new")
    original_pointer = (embedding_set_root / "current.json").read_bytes()
    (old.directory / "stored_dataset.pt").write_bytes(b"corrupt")

    with pytest.raises(
            store.GenerationValidationError, match="no valid prior"):
        store.rollback_generation(embedding_set_root)

    assert (embedding_set_root / "current.json").read_bytes() == original_pointer
    assert store.resolve_active_generation(
        embedding_set_root
    ).generation_id == new.generation_id


def test_rollback_requires_a_retained_prior_generation(tmp_path):
    """A first publication cannot roll back to the legacy flat root."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    current = _publish(store, embedding_set_root, "only")

    with pytest.raises(
            store.GenerationValidationError, match="no valid prior"):
        store.rollback_generation(embedding_set_root)

    assert store.resolve_active_generation(
        embedding_set_root
    ).generation_id == current.generation_id


def test_incremental_update_transitions_fat_generation_to_slim_layout(tmp_path):
    """Updating a pre-migration generation with seed_neighbors=False drops neighbor files."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    pre_migration = _publish(store, embedding_set_root, "legacy_fat")
    assert (pre_migration.directory / "stored_image_neighbors.pt").exists()
    assert (pre_migration.directory / "stored_centroid_neighbors.pt").exists()

    def build_slim_incremental(staging_dir, _previous_dir):
        """Write an updated dataset without writing or updating neighbor files."""
        staging_dir = Path(staging_dir)
        dataset = _StoredDataset("slim_updated")
        torch.save(dataset, staging_dir / "stored_dataset.pt")
        torch.save(
            torch.tensor([[0.0, 1.0]]),
            staging_dir / "stored_embeddings.pt",
        )
        torch.save(
            torch.tensor([[0.0, 1.0]]),
            staging_dir / "stored_centroid_embeddings.pt",
        )
        torch.save(
            torch.tensor([1]),
            staging_dir / "stored_centroid_labels.pt",
        )
        return 1

    slim = store.publish_generation(
        embedding_set_root=embedding_set_root,
        embedding_set="test_embeddings",
        model_name="test_model",
        source="snapshot",
        snapshot_id=SNAPSHOT_ID,
        start_time=2_000.0,
        retrieval_mode="baseline",
        build=build_slim_incremental,
        seed_previous=True,
        seed_neighbors=False,
    )

    assert not (slim.directory / "stored_image_neighbors.pt").exists()
    assert not (slim.directory / "stored_centroid_neighbors.pt").exists()
    assert "stored_image_neighbors.pt" not in slim.manifest.files
    assert "stored_centroid_neighbors.pt" not in slim.manifest.files


def test_broken_pointer_errors_name_the_repair_step(tmp_path):
    """Resolving and publishing over a missing generation tell the operator how to repair it."""
    store = _generation_store()
    embedding_set_root = tmp_path / "test_embeddings"
    _write_resolvable_generation(embedding_set_root)
    (embedding_set_root / "generations" / GENERATION_ID).rename(
        embedding_set_root / "unreferenced-generation")

    with pytest.raises(store.GenerationValidationError, match="delete current.json"):
        store.resolve_active_generation(embedding_set_root)
    with pytest.raises(store.GenerationValidationError, match="delete current.json"):
        _publish(store, embedding_set_root, "after")
