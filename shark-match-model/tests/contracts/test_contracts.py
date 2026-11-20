"""Tests for V1 boundary definitions."""

import json
import os
import subprocess
import sys

from pydantic import ValidationError
import pytest

from src.core import constants
from src.contracts import (
    GenerationManifestV1,
    GenerationPointerV1,
    ModelDiagnostic,
    ModelSpec,
    ModelState,
    ModelStatus,
    SnapshotManifestV1,
)
from src.retrieval.retrieval import validate_model_checkpoint

SNAPSHOT_ID = "11111111-1111-4111-8111-111111111111"
GENERATION_ID = "22222222-2222-4222-8222-222222222222"
CREATED_AT = "2026-08-31T12:00:00Z"
SHA_A = "a" * 64
SHA_B = "b" * 64


def test_snapshot_manifest_accepts_the_v1_contract():
    """The frozen version-1 snapshot wire shape is accepted."""
    manifest = SnapshotManifestV1.model_validate(_snapshot_manifest())
    assert manifest.snapshot_id == SNAPSHOT_ID
    assert set(manifest.files) == {"individuals.csv", "media.csv"}


@pytest.mark.parametrize(
    "invalid_id",
    ["../snapshot", "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", "not-a-uuid"],
)
def test_snapshot_manifest_rejects_noncanonical_identifiers(invalid_id):
    """Paths, uppercase UUIDs, and malformed IDs are rejected."""
    payload = _snapshot_manifest()
    payload["snapshot_id"] = invalid_id
    with pytest.raises(ValidationError):
        SnapshotManifestV1.model_validate(payload)


def test_snapshot_manifest_rejects_missing_or_unknown_files():
    """A snapshot must contain exactly its two public CSV files."""
    payload = _snapshot_manifest()
    payload["files"].pop("media.csv")
    payload["files"]["users.csv"] = {"rows": 1, "sha256": SHA_B}
    with pytest.raises(ValidationError):
        SnapshotManifestV1.model_validate(payload)


def test_snapshot_manifest_rejects_unknown_top_level_fields():
    """Unknown fields cannot silently extend a versioned manifest."""
    payload = _snapshot_manifest()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        SnapshotManifestV1.model_validate(payload)


def test_generation_manifest_enforces_snapshot_source_pairing():
    """Snapshot IDs are present only for snapshot-sourced generations."""
    assert GenerationManifestV1.model_validate(_generation_manifest()).snapshot_id == SNAPSHOT_ID
    with pytest.raises(ValidationError):
        GenerationManifestV1.model_validate(_generation_manifest(snapshot_id=None))
    with pytest.raises(ValidationError):
        GenerationManifestV1.model_validate(
            _generation_manifest(source="offline_annotations", snapshot_id=SNAPSHOT_ID)
        )


def test_generation_manifest_rejects_nested_or_missing_metadata_artifacts():
    """Generation artifact names are flat and metadata is mandatory."""
    nested = _generation_manifest()
    nested["files"] = {"nested/file.pt": {"size_bytes": 1, "sha256": SHA_A}}
    with pytest.raises(ValidationError):
        GenerationManifestV1.model_validate(nested)


def test_generation_pointer_accepts_only_versioned_uuid_pointer():
    """A valid versioned generation pointer preserves its UUID."""
    pointer = GenerationPointerV1.model_validate({
        "schema_version": 1,
        "generation_id": GENERATION_ID,
        "published_at": CREATED_AT,
    })
    assert pointer.generation_id == GENERATION_ID


def test_failed_model_status_requires_safe_diagnostic():
    """Failed models require a structured safe diagnostic."""
    status = ModelStatus(
        display_name="MiewID Multimodal",
        embedding_set="miew_arcface",
        required=True,
        state=ModelState.FAILED,
        error=ModelDiagnostic(code="artifact_missing", message="Required model data is missing."),
    )
    assert status.state == ModelState.FAILED
    with pytest.raises(ValidationError):
        ModelStatus(
            display_name="MiewID Multimodal",
            embedding_set="miew_arcface",
            required=True,
            state=ModelState.FAILED,
        )


def test_snapshot_root_defaults_to_shared_data_directory(monkeypatch):
    """The default matcher snapshot root lives in shared data."""
    monkeypatch.delenv("SHARK_INFERENCE_SNAPSHOT_ROOT", raising=False)
    expected = os.path.abspath(os.path.join(constants.BASE_DATA_DIR, "inference-snapshots"))
    assert constants.resolve_inference_snapshot_root() == expected


def test_snapshot_root_honors_explicit_environment_path(monkeypatch, tmp_path):
    """The matcher honors an explicitly configured snapshot root."""
    configured = tmp_path / "snapshots"
    monkeypatch.setenv("SHARK_INFERENCE_SNAPSHOT_ROOT", str(configured))
    assert constants.resolve_inference_snapshot_root() == str(configured)


def test_data_root_defaults_to_sibling_sm_data(monkeypatch):
    """Without an override the shared data root stays the sibling sm-data directory."""
    monkeypatch.delenv("SHARK_DATA_ROOT", raising=False)
    assert constants.resolve_base_data_dir() == os.path.join(constants.BASE_DIR, "sm-data")


def test_data_root_honors_an_absolute_override(monkeypatch, tmp_path):
    """An absolute SHARK_DATA_ROOT replaces the sibling layout."""
    monkeypatch.setenv("SHARK_DATA_ROOT", str(tmp_path / "data"))
    assert constants.resolve_base_data_dir() == str(tmp_path / "data")


def test_data_root_rejects_a_relative_override(monkeypatch):
    """A relative SHARK_DATA_ROOT would resolve differently per working directory, so it is refused."""
    monkeypatch.setenv("SHARK_DATA_ROOT", os.path.join("relative", "data"))
    with pytest.raises(ValueError, match="absolute"):
        constants.resolve_base_data_dir()


def _derived_data_paths(data_root, names, extra_environment=None):
    """Return the named constants as imported in a fresh process rooted at data_root."""
    environment = {
        key: value for key, value in os.environ.items()
        if key != "SHARK_INFERENCE_SNAPSHOT_ROOT"
    }
    environment.update(extra_environment or {})
    environment["SHARK_DATA_ROOT"] = str(data_root)
    script = (
        "import json, sys; from src.core import constants; "
        "print(json.dumps({name: getattr(constants, name) for name in sys.argv[1:]}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *names],
        cwd=constants.THIS_DIR,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_data_root_override_moves_every_derived_data_directory(tmp_path):
    """Every data path derived at import time follows SHARK_DATA_ROOT."""
    data_root = tmp_path / "data"
    names = ["ANNOTATIONS_DIR", "DB_DATA_DIR", "EMBEDDINGS_DIR", "IMAGES_DIR", "INFERENCE_SNAPSHOT_ROOT", "OUT_DATA_DIR", "TEST_DATA_DIR", "TRAINED_MODELS_DIR"]
    paths = _derived_data_paths(data_root, names)
    assert all(path.startswith(str(data_root) + os.sep) for path in paths.values()), paths


def test_app_test_mode_variable_does_not_redirect_matcher_data(tmp_path):
    """The app's SHARK_TEST_MODE selects only its test database, so the matcher's data paths ignore it."""
    data_root = tmp_path / "data"
    names = ["ANNOTATIONS_DIR", "DB_DATA_DIR", "EMBEDDINGS_DIR", "TRAINED_MODELS_DIR", "BASE_IMAGES_DIR", "RAW_DATA_DIR"]
    paths = _derived_data_paths(data_root, names, {"SHARK_TEST_MODE": "true"})
    assert paths == _derived_data_paths(data_root, names, {"SHARK_TEST_MODE": "false"})
    assert not any("test-data" in path for path in paths.values()), paths


def test_inference_bind_address_defaults_to_all_interfaces_on_8000(monkeypatch):
    """Without overrides the inference server keeps its historical bind address."""
    monkeypatch.delenv("SHARK_INFERENCE_HOST", raising=False)
    monkeypatch.delenv("SHARK_INFERENCE_PORT", raising=False)
    assert constants.resolve_inference_bind_address() == ("0.0.0.0", 8000)


def test_inference_bind_address_honors_environment(monkeypatch):
    """The inference server binds to the configured host and port."""
    monkeypatch.setenv("SHARK_INFERENCE_HOST", "127.0.0.1")
    monkeypatch.setenv("SHARK_INFERENCE_PORT", "18000")
    assert constants.resolve_inference_bind_address() == ("127.0.0.1", 18000)


def test_model_spec_normalizes_legacy_fields():
    """Legacy dictionary fields are mapped to canonical ModelSpec attributes."""
    spec = ModelSpec.model_validate({
        "name": "Legacy Name",
        "dir": "legacy_dir",
        "model": "legacy_model",
        "retrieval_mode": "baseline",
        "rerank": "none",
        "required": False,
    })
    assert spec.display_name == "Legacy Name"
    assert spec.embedding_set == "legacy_dir"
    assert spec.checkpoint == "legacy_model"
    assert spec.dir == "legacy_dir"
    assert spec.model == "legacy_model"
    assert spec["dir"] == "legacy_dir"
    assert spec["model"] == "legacy_model"
    assert spec.get("dir") == "legacy_dir"
    assert not spec.required


def test_model_spec_rejects_empty_identifiers():
    """Empty model identifiers are rejected by the strict contract."""
    with pytest.raises(ValidationError):
        ModelSpec(display_name="", embedding_set="set", checkpoint="chk")
    with pytest.raises(ValidationError):
        ModelSpec(display_name="name", embedding_set="", checkpoint="chk")
    with pytest.raises(ValidationError):
        ModelSpec(display_name="name", embedding_set="set", checkpoint="")


def test_validate_model_checkpoint_accepts_precanned_models():
    """Pre-canned models bypass local checkpoint configuration checks."""
    spec = ModelSpec(display_name="miew", embedding_set="miew", checkpoint="miew")
    validate_model_checkpoint(spec)


def test_validate_model_checkpoint_accepts_existing_trained_model(tmp_path):
    """An existing trained-model configuration file passes validation."""
    model_dir = tmp_path / "custom_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    spec = ModelSpec(
        display_name="Custom",
        embedding_set="custom_dir",
        checkpoint="custom_model",
    )
    validate_model_checkpoint(spec, trained_models_dir=str(tmp_path))


def test_validate_model_checkpoint_fails_fast_on_missing_model(tmp_path):
    """Missing checkpoints raise an immediate descriptive error."""
    spec = ModelSpec(
        display_name="Missing",
        embedding_set="missing_dir",
        checkpoint="missing_model",
    )
    with pytest.raises(RuntimeError, match="Missing.*missing_model"):
        validate_model_checkpoint(spec, trained_models_dir=str(tmp_path))


def _snapshot_manifest():
    """Return a valid snapshot manifest payload for mutation in tests."""
    return {
        "schema_version": 1,
        "snapshot_id": SNAPSHOT_ID,
        "created_at": CREATED_AT,
        "files": {
            "individuals.csv": {"rows": 2, "sha256": SHA_A},
            "media.csv": {"rows": 1, "sha256": SHA_B},
        },
    }


def _generation_manifest(source="snapshot", snapshot_id=SNAPSHOT_ID):
    """Return a valid generation manifest payload for mutation in tests."""
    return {
        "schema_version": 1,
        "generation_id": GENERATION_ID,
        "created_at": CREATED_AT,
        "embedding_set": "miew_arcface",
        "model_name": "initial_miew_arcface_new",
        "source": source,
        "snapshot_id": snapshot_id,
        "files": {
            "embeddings_metadata.json": {"size_bytes": 256, "sha256": SHA_A},
        },
    }


def test_pairx_root_defaults_to_sibling_pairx(monkeypatch):
    """Without an override the optional pairx checkout is the normalized sibling directory."""
    monkeypatch.delenv("PAIRX_ROOT", raising=False)
    assert constants.resolve_pairx_dir() == os.path.abspath(os.path.join(constants.BASE_DIR, "pairx"))


def test_pairx_root_override_is_made_absolute(monkeypatch, tmp_path):
    """A relative or home-relative PAIRX_ROOT is expanded once so later directory changes cannot move it."""
    monkeypatch.setenv("PAIRX_ROOT", str(tmp_path / "pairx"))
    assert constants.resolve_pairx_dir() == str(tmp_path / "pairx")
    monkeypatch.setenv("PAIRX_ROOT", os.path.join("relative", "pairx"))
    assert os.path.isabs(constants.resolve_pairx_dir())
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PAIRX_ROOT", os.path.join("~", "pairx"))
    assert constants.resolve_pairx_dir() == os.path.join(str(tmp_path), "pairx")


def test_default_models_name_the_shipped_demo_bundle():
    """DEFAULT_MODELS must name the embedding set and checkpoint that the README documents for the demo bundle."""
    readme = open(os.path.join(constants.THIS_DIR, "README.md"), encoding="utf-8").read()
    assert constants.DEFAULT_MODELS
    for spec in constants.DEFAULT_MODELS:
        assert f"`{spec['dir']}`" in readme
        assert f"`{spec['model']}`" in readme
        ModelSpec(**spec)
