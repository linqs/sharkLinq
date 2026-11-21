"""Unit tests for src.retrieval.retrieval.Retriever covering neighbor lookups and leave-one-out centroid retrieval."""

import hashlib
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.contracts import (
    ArtifactFileV1,
    GenerationManifestV1,
    GenerationPointerV1,
    ModelDiagnostic,
    ModelState,
)
import src.retrieval.generation_store
from src.retrieval.retrieval import Retriever

# ---------------------------------------------------------------------------
# Helpers — fixture data construction
# ---------------------------------------------------------------------------

MODEL = "test_model"
GENERATION_ID = "22222222-2222-4222-8222-222222222222"
CREATED_AT = "2026-08-31T12:00:00Z"
IDS = [101, 102, 201]
LABELS = np.array([10, 10, 20])
FILENAMES = ["a.jpg", "b.jpg", "c.jpg"]
ID_TO_FILE = dict(zip(IDS, FILENAMES))


def _make_neighbor_entry(matches, distances, labels):
    return {
        "matches": matches,
        "match_distances": np.array(distances),
        "match_labels": np.array(labels),
    }


IMAGE_NEIGHBORS = {
    "a.jpg": _make_neighbor_entry([101, 102, 201], [0.0, 0.2, 0.8], [10, 10, 20]),
    "b.jpg": _make_neighbor_entry([102, 101, 201], [0.0, 0.3, 0.6], [10, 10, 20]),
    "c.jpg": _make_neighbor_entry([201, 101, 102], [0.0, 0.7, 0.9], [20, 10, 10]),
}

CENTROID_NEIGHBORS = {
    "a.jpg": _make_neighbor_entry([101, 201], [0.0, 0.5], [10, 20]),
}

# Unit embeddings along 2D unit circle angles (0, 30, and 90 degrees) for hand-verifiable LOO geometry.
EMBEDDING_DIM = 128
_ANGLES_DEG = {101: 0.0, 102: 30.0, 201: 90.0}


def _unit_embedding(angle_deg):
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    rad = np.deg2rad(angle_deg)
    vec[0] = np.cos(rad)
    vec[1] = np.sin(rad)
    return vec


EMBEDDINGS = np.stack([_unit_embedding(_ANGLES_DEG[i]) for i in IDS])


def _make_dataset(test_ids=None):
    """Return a minimal dataset mock."""
    ds = MagicMock()
    ds.get_filename_by_id.side_effect = lambda id_: ID_TO_FILE.get(id_)
    ds.get_ids.return_value = list(IDS)
    ds.get_filenames.return_value = list(FILENAMES)
    ds.get_labels.return_value = MagicMock(
        cpu=lambda: MagicMock(detach=lambda: MagicMock(numpy=lambda: LABELS.copy()))
    )
    ds.get_test_ids.return_value = list(IDS) if test_ids is None else test_ids
    return ds


class _StoredDataset:
    """Pickle-compatible dataset for focused on-disk reader tests."""

    def get_ids(self):
        """Return stored media identifiers."""
        return list(IDS)

    def get_filenames(self):
        """Return stored filenames."""
        return list(FILENAMES)

    def get_labels(self):
        """Return stored labels."""
        return torch.tensor(LABELS)


def _write_reader_artifacts(model_dir):
    """Write the baseline artifact set consumed by `_load_model_data`."""
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_StoredDataset(), model_dir / "stored_dataset.pt")
    torch.save(torch.tensor(EMBEDDINGS), model_dir / "stored_embeddings.pt")
    torch.save(IMAGE_NEIGHBORS, model_dir / "stored_image_neighbors.pt")
    torch.save(
        torch.zeros((2, EMBEDDING_DIM)),
        model_dir / "stored_centroid_embeddings.pt",
    )
    torch.save(CENTROID_NEIGHBORS, model_dir / "stored_centroid_neighbors.pt")
    torch.save(torch.tensor([10, 20]), model_dir / "stored_centroid_labels.pt")
    torch.save(
        {"unmatch_suggestions": [], "merge_suggestions": []},
        model_dir / "suggestions.pt",
    )
    (model_dir / "embeddings_metadata.json").write_text(
        json.dumps({"start_time": 1_000.0, "update_time": 5.0}),
        encoding="utf-8",
    )


def _write_versioned_reader_artifacts(embedding_set_root):
    """Write one versioned generation containing the reader artifact fixture."""
    generation_dir = embedding_set_root / "generations" / GENERATION_ID
    _write_reader_artifacts(generation_dir)
    files = {}
    for artifact_path in generation_dir.iterdir():
        contents = artifact_path.read_bytes()
        files[artifact_path.name] = ArtifactFileV1(
            size_bytes=len(contents),
            sha256=hashlib.sha256(contents).hexdigest(),
        )
    manifest = GenerationManifestV1(
        schema_version=1,
        generation_id=GENERATION_ID,
        created_at=CREATED_AT,
        embedding_set=MODEL,
        model_name=MODEL,
        source="offline_annotations",
        snapshot_id=None,
        files=files,
    )
    (generation_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    pointer = GenerationPointerV1(
        schema_version=1,
        generation_id=GENERATION_ID,
        published_at=CREATED_AT,
    )
    (embedding_set_root / "current.json").write_text(
        pointer.model_dump_json(indent=2), encoding="utf-8"
    )
    return generation_dir


def _make_model_data(centroid=True, test_ids=None):
    ds = _make_dataset(test_ids=test_ids)
    data = {
        "dataset": ds,
        "ids": list(IDS),
        "filenames": list(FILENAMES),
        "labels": LABELS.copy(),
        "embeddings": EMBEDDINGS.copy(),
        "image_neighbors": dict(IMAGE_NEIGHBORS),
        "centroid_neighbors": dict(CENTROID_NEIGHBORS) if centroid else None,
        "centroid_embeddings": np.zeros((2, 128)) if centroid else None,
        "centroid_labels": np.array([10, 20]) if centroid else None,
        "embeddings_metadata": {"start_time": None, "update_time": None},
    }
    return data


def _model_spec(display_name=MODEL, **overrides):
    """Return one normalized model configuration for Retriever tests."""
    spec = {
        "display_name": display_name,
        "dir": display_name,
        "model": display_name,
        "retrieval_mode": "baseline",
        "rerank": "none",
    }
    spec.update(overrides)
    return spec


def _load_disk_retriever(embeddings_root):
    """Load the baseline test model from an explicit embeddings root."""
    # Pin to CPU because MPS does not support float64 fixture tensors.
    with patch(
            "src.retrieval.retrieval.utils.get_torch_device",
            return_value=torch.device("cpu"),
    ):
        return Retriever(
            preload_models=[_model_spec()],
            embeddings_root=str(embeddings_root),
        )


def _model_statuses(retriever):
    """Return model statuses from the retriever."""
    return retriever.get_model_statuses()


def _mark_model_ready(retriever):
    """Mark the test model as ready."""
    retriever._set_model_status(MODEL, ModelState.READY)


def _mark_model_failed(retriever):
    """Mark the test model as failed with a missing-artifact error."""
    retriever._set_model_status(
        MODEL,
        ModelState.FAILED,
        ModelDiagnostic(
            code="artifact_missing",
            message="Required model data is missing.",
        ),
    )


@pytest.fixture
def retriever():
    """Retriever with one model loaded from in-memory fixture data."""
    with patch.object(Retriever, "_load_model_data"):
        r = Retriever(preload_models=[MODEL])
    r.loaded_models[MODEL] = _make_model_data()
    _mark_model_ready(r)
    return r


@pytest.fixture
def retriever_no_centroid():
    """Retriever where centroid files were not available."""
    with patch.object(Retriever, "_load_model_data"):
        r = Retriever(preload_models=[MODEL])
    r.loaded_models[MODEL] = _make_model_data(centroid=False)
    _mark_model_ready(r)
    return r


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class TestGenerationResolution:
    """Reader parity across legacy-flat and versioned generation layouts."""

    def test_legacy_layout_preserves_existing_model_load(self, tmp_path):
        """A pointerless embedding set loads with the existing flat behavior."""
        _write_reader_artifacts(tmp_path / MODEL)

        retriever = _load_disk_retriever(tmp_path)
        model_data = retriever.loaded_models[MODEL]

        assert model_data["generation_id"] is None
        assert model_data["snapshot_id"] is None
        assert model_data["filenames"] == FILENAMES
        assert np.array_equal(model_data["embeddings"], EMBEDDINGS)

    def test_versioned_layout_loads_only_resolved_generation(self, tmp_path):
        """All reader artifacts come from the generation selected by one pointer."""
        generation_dir = _write_versioned_reader_artifacts(tmp_path / MODEL)
        (tmp_path / MODEL / "stored_dataset.pt").write_bytes(b"stale flat data")

        with patch.object(
                src.retrieval.generation_store,
                "resolve_active_generation",
                wraps=src.retrieval.generation_store.resolve_active_generation,
        ) as resolve_active:
            retriever = _load_disk_retriever(tmp_path)
        model_data = retriever.loaded_models[MODEL]

        resolve_active.assert_called_once_with(str(tmp_path / MODEL))
        assert generation_dir.is_dir()
        assert model_data["generation_id"] == GENERATION_ID
        assert model_data["snapshot_id"] is None
        assert model_data["filenames"] == FILENAMES
        assert np.array_equal(model_data["embeddings"], EMBEDDINGS)

    def test_corrupt_pointer_marks_only_model_failed(self, tmp_path):
        """A malformed active pointer produces generation-invalid status."""
        model_root = tmp_path / MODEL
        _write_reader_artifacts(model_root)
        (model_root / "current.json").write_text(
            "{not-json", encoding="utf-8"
        )

        retriever = _load_disk_retriever(tmp_path)

        status = _model_statuses(retriever)[0]
        assert status.state == ModelState.FAILED
        assert status.error.code == "generation_invalid"
        assert MODEL not in retriever.loaded_models


class TestModelLifecycleStatus:
    """Tests for Retriever model status transitions."""

    def test_configured_model_starts_not_attempted(self):
        """Configuration creates status before the loader begins work."""
        with patch.object(Retriever, "_load_model_data"):
            retriever = Retriever(preload_models=[MODEL])

        statuses = _model_statuses(retriever)
        assert len(statuses) == 1
        assert statuses[0].state == ModelState.NOT_ATTEMPTED
        assert statuses[0].required is True

    def test_successful_load_finishes_ready(self, tmp_path):
        """A complete artifact load replaces loading state with ready state."""
        _write_reader_artifacts(tmp_path / MODEL)

        retriever = _load_disk_retriever(tmp_path)

        status = _model_statuses(retriever)[0]
        assert status.state == ModelState.READY
        assert status.error is None

    def test_missing_artifacts_finish_failed(self, tmp_path):
        """A missing model artifact set produces artifact-missing status."""
        retriever = _load_disk_retriever(tmp_path)

        status = _model_statuses(retriever)[0]
        assert status.state == ModelState.FAILED
        assert status.error.code == "artifact_missing"
        assert MODEL not in retriever.loaded_models

    def test_incompatible_stored_dataset_finishes_failed(self, tmp_path):
        """A valid serialization with the wrong interface is incompatible."""
        model_root = tmp_path / MODEL
        _write_reader_artifacts(model_root)
        torch.save(
            {"unexpected": "dataset shape"},
            model_root / "stored_dataset.pt",
        )

        retriever = _load_disk_retriever(tmp_path)

        status = _model_statuses(retriever)[0]
        assert status.state == ModelState.FAILED
        assert status.error.code == "artifact_incompatible"
        assert MODEL not in retriever.loaded_models

    @pytest.mark.parametrize(
        ("error", "expected_code", "expected_message"),
        [
            (
                FileNotFoundError("/private/model-store/secret/model.pt"),
                "artifact_missing",
                "Required model data is missing.",
            ),
            (
                src.retrieval.generation_store.GenerationValidationError(
                    "/private/model-store/secret/current.json"
                ),
                "generation_invalid",
                "The active model generation is invalid.",
            ),
            (
                EOFError("/private/model-store/secret/stored_dataset.pt"),
                "artifact_incompatible",
                "Required model data is incompatible.",
            ),
            (
                OSError("/private/model-store/secret/device"),
                "load_failed",
                "Required model data could not be loaded.",
            ),
        ],
    )
    def test_loading_transition_precedes_safe_persistent_failure(
            self, error, expected_code, expected_message, capsys):
        """Each core failure retains its fixed path-free diagnostic."""
        with patch.object(Retriever, "_load_model_data"):
            retriever = Retriever(preload_models=[MODEL])

        sensitive_path = str(error)

        def fail_after_observing_loading(_embedding_set_root):
            statuses = _model_statuses(retriever)
            assert statuses[0].state == ModelState.LOADING
            raise error

        with patch.object(
                src.retrieval.generation_store,
                "resolve_active_generation",
                side_effect=fail_after_observing_loading,
        ):
            retriever._load_model_data(MODEL)

        status = _model_statuses(retriever)[0]
        assert status.state == ModelState.FAILED
        assert status.error.code == expected_code
        assert status.error.message == expected_message
        assert sensitive_path not in status.model_dump_json()
        assert sensitive_path not in capsys.readouterr().out
        assert MODEL not in retriever.loaded_models

    @pytest.mark.parametrize(
        ("retrieval_mode", "expected_flag"),
        [("baseline", None), ("multimodal", "--use-multimodal")],
    )
    def test_failure_remedy_names_the_real_generation_command(
            self, tmp_path, retrieval_mode, expected_flag, capsys):
        """A missing embedding set prints a runnable, path-free embeddings command for that set."""
        with patch(
                "src.retrieval.retrieval.utils.get_torch_device",
                return_value=torch.device("cpu"),
        ):
            retriever = Retriever(
                preload_models=[_model_spec(
                    dir="missing_set",
                    model="trained_checkpoint",
                    retrieval_mode=retrieval_mode,
                )],
                embeddings_root=str(tmp_path),
            )

        out = capsys.readouterr().out
        assert _model_statuses(retriever)[0].state == ModelState.FAILED
        assert ("python3 -m src.retrieval.embeddings --model-name trained_checkpoint "
                "--embeddings-name missing_set") in out
        assert "SHARK_MODELS" in out
        assert "shark_identification" not in out
        assert str(tmp_path) not in out
        if expected_flag is None:
            assert "--use-multimodal" not in out
        else:
            assert expected_flag in out

    def test_explicitly_optional_model_is_disabled_without_loading(self):
        """An optional configured model is visible but skipped intentionally."""
        with patch.object(Retriever, "_load_model_data") as load_model:
            retriever = Retriever(
                preload_models=[_model_spec(required=False)]
            )

        load_model.assert_not_called()
        status = _model_statuses(retriever)[0]
        assert status.state == ModelState.DISABLED
        assert status.required is False

    def test_status_order_survives_out_of_order_transitions(self):
        """Status reads preserve configuration order rather than state order."""
        configured_names = ["zeta", "alpha", "middle"]
        with patch.object(Retriever, "_load_model_data"):
            retriever = Retriever(
                preload_models=[_model_spec(name) for name in configured_names]
            )

        retriever._set_model_status("middle", ModelState.LOADING)
        retriever._set_model_status("zeta", ModelState.READY)
        retriever._set_model_status("alpha", ModelState.LOADING)

        assert [
            status.display_name for status in _model_statuses(retriever)
        ] == configured_names


class TestModelResolution:
    def test_none_resolves_to_default(self, retriever):
        assert retriever._resolve_model(None) == MODEL

    def test_known_alias_resolves(self, retriever):
        assert retriever._resolve_model("ViT") == "vit_triplet"
        assert retriever._resolve_model("Miew") == "miew"

    def test_unknown_name_passes_through(self, retriever):
        assert retriever._resolve_model("custom_model") == "custom_model"


# ---------------------------------------------------------------------------
# _get_model_data gating
# ---------------------------------------------------------------------------

class TestGetModelData:
    def test_returns_data_for_loaded_model(self, retriever):
        data = retriever._get_model_data(MODEL)
        assert data is not None
        assert "image_neighbors" in data

    def test_none_model_returns_default(self, retriever):
        data = retriever._get_model_data(None)
        assert data is not None

    def test_failed_model_returns_none(self, retriever):
        _mark_model_failed(retriever)
        assert retriever._get_model_data(MODEL) is None

    def test_disallowed_model_returns_none(self, retriever):
        # "other_model" was never in preload_models, so it's not in allowed_models.
        assert retriever._get_model_data("other_model") is None


# ---------------------------------------------------------------------------
# get_image_neighbors — happy path
# ---------------------------------------------------------------------------

class TestGetImageNeighbors:
    def test_returns_correct_ids_order(self, retriever):
        result = retriever.get_image_neighbors(101)
        assert result["ids"] == [101, 102, 201]

    def test_distances_are_python_floats(self, retriever):
        result = retriever.get_image_neighbors(101)
        assert all(isinstance(d, float) for d in result["distances"])
        assert result["distances"] == pytest.approx([0.0, 0.2, 0.8])

    def test_ids_are_python_ints(self, retriever):
        result = retriever.get_image_neighbors(101)
        assert all(isinstance(i, int) for i in result["ids"])

    def test_labels_are_correct(self, retriever):
        result = retriever.get_image_neighbors(101)
        assert result["labels"] == [10, 10, 20]

    def test_different_query_returns_different_result(self, retriever):
        result = retriever.get_image_neighbors(201)
        assert result["ids"] == [201, 101, 102]
        assert result["distances"] == pytest.approx([0.0, 0.7, 0.9])

    def test_unknown_query_id_returns_empty(self, retriever):
        result = retriever.get_image_neighbors(999)
        assert result == {"ids": [], "distances": [], "labels": []}

    def test_failed_model_returns_empty(self, retriever):
        _mark_model_failed(retriever)
        result = retriever.get_image_neighbors(101)
        assert result == {"ids": [], "distances": [], "labels": []}


# Leave-one-out centroid retrieval recomputes the query individual's centroid excluding the query embedding.

class TestGetLooCentroidNeighbors:
    def test_returns_all_images_grouped_by_individual(self, retriever):
        # Individual 10 (101, 102) is nearer to query 101 than individual 20
        # (201), and every member of an individual shares its centroid distance.
        result = retriever.get_loo_centroid_neighbors(101)
        assert result["ids"] == [101, 102, 201]
        assert result["labels"] == [10, 10, 20]
        assert result["distances"] == pytest.approx([0.51764, 0.51764, 1.41421], abs=1e-5)

    def test_loo_excludes_query_from_centroid(self, retriever):
        """Verify the query image is excluded when computing its individual's centroid distance."""
        result = retriever.get_loo_centroid_neighbors(101)
        own_individual_distance = result["distances"][0]
        assert own_individual_distance == pytest.approx(0.51764, abs=1e-5)
        assert own_individual_distance != pytest.approx(0.26105, abs=1e-5)

    def test_single_member_individual_keeps_standard_centroid(self, retriever):
        """A single-image individual keeps itself as its centroid, giving distance 0."""
        result = retriever.get_loo_centroid_neighbors(201)
        assert result["ids"][0] == 201
        assert result["distances"][0] == pytest.approx(0.0, abs=1e-9)

    def test_unknown_query_id_returns_empty(self, retriever):
        result = retriever.get_loo_centroid_neighbors(999)
        assert result == {"ids": [], "distances": [], "labels": []}

    def test_does_not_require_precomputed_centroid_data(self, retriever_no_centroid):
        """LOO computes centroids from embeddings and works without precomputed centroid files."""
        result = retriever_no_centroid.get_loo_centroid_neighbors(101)
        assert result["ids"] == [101, 102, 201]

    def test_failed_model_returns_empty(self, retriever):
        _mark_model_failed(retriever)
        result = retriever.get_loo_centroid_neighbors(101)
        assert result == {"ids": [], "distances": [], "labels": []}


# ---------------------------------------------------------------------------
# get_image_neighbors_average
# ---------------------------------------------------------------------------

class TestGetImageNeighborsAverage:
    def test_individuals_ranked_by_average_distance(self, retriever):
        """Rank individual 10 (avg distance 0.2) before individual 20 (avg distance 0.8)."""
        result = retriever.get_image_neighbors_average(101)

        labels = result["labels"]
        # All individual-10 images before individual-20 images.
        last_10 = max(i for i, l in enumerate(labels) if l == 10)
        first_20 = min(i for i, l in enumerate(labels) if l == 20)
        assert last_10 < first_20

    def test_query_excluded_from_average_not_from_output(self, retriever):
        """Exclude the query image from its individual's average distance, but keep it in results."""
        result = retriever.get_image_neighbors_average(101)
        assert 101 in result["ids"]

    def test_output_covers_all_images(self, retriever):
        result = retriever.get_image_neighbors_average(101)
        assert set(result["ids"]) == {101, 102, 201}

    def test_empty_when_no_neighbors(self, retriever):
        result = retriever.get_image_neighbors_average(999)
        assert result == {"ids": [], "distances": [], "labels": []}


# ---------------------------------------------------------------------------
# get_neighbors dispatch (method 0–2)
# ---------------------------------------------------------------------------

class TestGetNeighborsDispatch:
    def test_method_0_uses_image_neighbors(self, retriever):
        result = retriever.get_neighbors(0, 101)
        assert result["ids"] == [101, 102, 201]

    def test_method_2_uses_loo_centroid_neighbors(self, retriever):
        result = retriever.get_neighbors(2, 101)
        assert result["ids"] == [101, 102, 201]
        assert result["distances"] == pytest.approx([0.51764, 0.51764, 1.41421], abs=1e-5)

    def test_model_kwarg_forwarded(self, retriever):
        """Passing model=None (default) still works."""
        result = retriever.get_neighbors(0, 101, model=None)
        assert result["ids"] == [101, 102, 201]


class TestPureNumpyPrimitives:
    """Verify pure-NumPy normalization, centroid computation, and cross L2 distances."""

    def test_l2_normalize_unit_rows_and_zero_row_safety(self):
        from src.retrieval.rerank import l2_normalize

        raw = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        normed = l2_normalize(raw)
        assert normed.shape == (2, 3)
        assert np.allclose(normed[0], [0.6, 0.8, 0.0], atol=1e-6)
        assert np.allclose(normed[1], [0.0, 0.0, 0.0], atol=1e-6)

    def test_compute_centroids_sorted_unique_labels(self):
        from src.retrieval.rerank import compute_centroids

        centroids, unique_labels = compute_centroids(EMBEDDINGS, LABELS)
        assert list(unique_labels) == [10, 20]
        assert centroids.shape == (2, EMBEDDING_DIM)
        norms = np.linalg.norm(centroids, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_pairwise_l2_cross_matrix_matches_symmetric_submatrix(self):
        from src.retrieval.rerank import _pairwise_l2

        full = _pairwise_l2(EMBEDDINGS)
        cross = _pairwise_l2(EMBEDDINGS[:2], EMBEDDINGS)
        assert cross.shape == (2, 3)
        assert np.allclose(cross, full[:2, :], atol=1e-12)


class TestOnTheFlyImageNeighbors:
    """Verify on-the-fly NumPy neighbor retrieval when image_neighbors is None."""

    def test_on_the_fly_method_0_and_method_1_match_precomputed_dict(self, retriever):
        from src.retrieval.rerank import _pairwise_l2

        dists = _pairwise_l2(EMBEDDINGS)
        precomputed = {}
        for i, fn in enumerate(FILENAMES):
            order = np.argsort(dists[i], kind="mergesort")
            precomputed[fn] = _make_neighbor_entry(
                [IDS[j] for j in order],
                [float(dists[i, j]) for j in order],
                [int(LABELS[j]) for j in order],
            )

        data = retriever.loaded_models[MODEL]
        data["image_neighbors"] = precomputed
        dict_m0 = retriever.get_image_neighbors(101)
        dict_m1 = retriever.get_image_neighbors_average(101)

        data["image_neighbors"] = None
        fly_m0 = retriever.get_image_neighbors(101)
        fly_m1 = retriever.get_image_neighbors_average(101)

        assert fly_m0["ids"] == dict_m0["ids"]
        assert fly_m0["labels"] == dict_m0["labels"]
        assert fly_m0["distances"] == pytest.approx(dict_m0["distances"], abs=1e-6)

        assert fly_m1["ids"] == dict_m1["ids"]
        assert fly_m1["labels"] == dict_m1["labels"]
        assert fly_m1["distances"] == pytest.approx(dict_m1["distances"], abs=1e-6)

    def test_load_model_data_succeeds_without_neighbor_files(
            self, tmp_path, monkeypatch):
        """Baseline generation without stored_image_neighbors.pt loads cleanly."""
        monkeypatch.setattr("src.core.constants.EMBEDDINGS_DIR", str(tmp_path))
        model_root = tmp_path / MODEL
        gen_dir = model_root / "generations" / GENERATION_ID
        gen_dir.mkdir(parents=True, exist_ok=True)
        torch.save(_StoredDataset(), gen_dir / "stored_dataset.pt")
        torch.save(torch.tensor(EMBEDDINGS), gen_dir / "stored_embeddings.pt")
        torch.save(
            torch.zeros((2, EMBEDDING_DIM)),
            gen_dir / "stored_centroid_embeddings.pt",
        )
        torch.save(torch.tensor([10, 20]), gen_dir / "stored_centroid_labels.pt")
        torch.save(
            {"unmatch_suggestions": [], "merge_suggestions": []},
            gen_dir / "suggestions.pt",
        )
        (gen_dir / "embeddings_metadata.json").write_text(
            json.dumps({"start_time": 1_000.0, "update_time": 5.0}),
            encoding="utf-8",
        )
        files = {
            path.name: ArtifactFileV1(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            )
            for path in sorted(gen_dir.iterdir())
        }
        manifest = GenerationManifestV1(
            schema_version=1,
            generation_id=GENERATION_ID,
            created_at=CREATED_AT,
            embedding_set=MODEL,
            model_name=MODEL,
            source="offline_annotations",
            snapshot_id=None,
            files=files,
        )
        (gen_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        pointer = GenerationPointerV1(
            schema_version=1,
            generation_id=GENERATION_ID,
            published_at=CREATED_AT,
        )
        (model_root / "current.json").write_text(
            pointer.model_dump_json(indent=2), encoding="utf-8"
        )

        r = Retriever(preload_models=[{
            "display_name": MODEL,
            "dir": MODEL,
            "model": MODEL,
            "retrieval_mode": "baseline",
            "rerank": "none",
            "required": True,
        }])
        assert r._model_statuses[MODEL].state == ModelState.READY
        assert r.loaded_models[MODEL]["image_neighbors"] is None
        assert r.loaded_models[MODEL]["centroid_neighbors"] is None
        res = r.get_image_neighbors(101)
        assert res["ids"] == [101, 102, 201]


@pytest.mark.parametrize(
    ("dir_name", "expect_warning"),
    [(".staging-abc", False), ("finished_set", True)],
)
def test_missing_metadata_only_warns_for_a_finished_generation(
        tmp_path, capsys, dir_name, expect_warning):
    """A generation still being staged has no metadata yet, so loading it stays quiet."""
    set_root = tmp_path / dir_name
    _write_reader_artifacts(set_root)
    (set_root / "embeddings_metadata.json").unlink()

    with patch("src.retrieval.retrieval.utils.get_torch_device",
               return_value=torch.device("cpu")):
        retriever = Retriever(preload_models=[_model_spec(dir=dir_name)],
                              embeddings_root=str(tmp_path))

    out = capsys.readouterr().out
    assert _model_statuses(retriever)[0].state == ModelState.READY
    assert ("Embeddings metadata not found" in out) is expect_warning
    assert "File not found" not in out
