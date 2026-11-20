"""Tools that read a named embedding set load its active generation, not the set root."""

import argparse
import contextlib
import importlib
import io
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.core import constants
from src.individual_dataset import IndividualDataset
import src.retrieval.evaluation as evaluation
import src.retrieval.fine_evaluation as fine_evaluation
from src.retrieval import generation_store
import src.visualization.visualization as visualization
import src.visualization.visualize_model as visualize_model

SET_NAME = "walkthrough-model/miewid_arcface"
PSL_MODULES = ("pslpython", "pslpython.model", "pslpython.partition",
               "pslpython.predicate", "pslpython.rule")


def import_train_psl():
    """Import train_psl with pslpython faked, because CI does not install the optional PSL stack."""
    with patch.dict(sys.modules, {name: MagicMock() for name in PSL_MODULES}):
        return importlib.import_module("src.psl.train_psl")


class StoredDataset:
    """Pickle-compatible dataset holding two images of two individuals."""

    def __len__(self):
        """Return the number of rows."""
        return 2

    def get_ids(self):
        """Return media identifiers in row order."""
        return [1, 2]

    def get_filenames(self):
        """Return filenames in row order."""
        return ["a.jpg", "b.jpg"]

    def get_labels(self):
        """Return labels in row order."""
        return torch.tensor([0, 1])

    def get_train_ids(self):
        """Return the training split."""
        return [1]

    def get_test_ids(self):
        """Return the test split."""
        return [2]


def _build(staging_dir, _previous_dir):
    staging_dir = Path(staging_dir)
    torch.save(StoredDataset(), staging_dir / "stored_dataset.pt")
    torch.save(torch.eye(2), staging_dir / "stored_embeddings.pt")
    torch.save(torch.eye(2), staging_dir / "stored_centroid_embeddings.pt")
    torch.save(torch.tensor([0, 1]), staging_dir / "stored_centroid_labels.pt")
    torch.save({"a.jpg": [], "b.jpg": []}, staging_dir / "stored_image_neighbors.pt")
    torch.save({"a.jpg": [], "b.jpg": []}, staging_dir / "stored_centroid_neighbors.pt")
    return 2


@pytest.fixture(name="published")
def published_fixture(tmp_path):
    """Publish one versioned generation under a patched embeddings root."""
    with patch.object(constants, "EMBEDDINGS_DIR", str(tmp_path)):
        generation = generation_store.publish_generation(
            embedding_set_root=os.path.join(str(tmp_path), SET_NAME),
            embedding_set=SET_NAME,
            model_name=SET_NAME,
            source="offline_annotations",
            snapshot_id=None,
            start_time=1_000.0,
            retrieval_mode="baseline",
            build=_build,
        )
        yield generation


class TestResolveActiveDirectory:
    def test_versioned_set_resolves_to_its_generation(self, published):
        assert generation_store.resolve_active_directory(SET_NAME) == published.directory
        assert "generations" in published.directory.parts

    def test_set_without_pointer_resolves_to_the_flat_root(self, tmp_path):
        with patch.object(constants, "EMBEDDINGS_DIR", str(tmp_path)):
            resolved = generation_store.resolve_active_directory("legacy")
        assert resolved == (tmp_path / "legacy").resolve()

    def test_broken_pointer_fails_instead_of_using_the_root(self, published):
        (published.directory.parent.parent / "current.json").write_text("{}", encoding="utf-8")
        with pytest.raises(generation_store.GenerationValidationError, match="to repair"):
            generation_store.resolve_active_directory(SET_NAME)


class TestReaders:
    def test_visualize_model_loads_the_active_generation(self, published):
        embeddings, labels = visualize_model.load_embeddings_and_labels(SET_NAME)
        assert embeddings.shape == (2, 2)
        assert labels.tolist() == [0, 1]

    def test_visualize_model_exits_nonzero_when_embeddings_are_missing(self, tmp_path):
        with patch.object(constants, "EMBEDDINGS_DIR", str(tmp_path)), \
                contextlib.redirect_stdout(io.StringIO()) as output, \
                pytest.raises(SystemExit) as exited:
            visualize_model.load_embeddings_and_labels("missing")
        assert exited.value.code == 1
        assert "stored_embeddings.pt not found" in output.getvalue()

    def test_visualize_model_creates_no_output_folder_when_embeddings_are_missing(self, tmp_path):
        visualizations = tmp_path / "visualizations"
        argv = ["visualize_model", "--model-name", "missing"]
        with patch.object(constants, "EMBEDDINGS_DIR", str(tmp_path)), \
                patch.object(constants, "VISUALIZATIONS_DIR", str(visualizations)), \
                patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()), \
                pytest.raises(SystemExit):
            visualize_model.main()
        assert not visualizations.exists()

    @pytest.mark.parametrize("centroid", [False, True])
    def test_visualization_loads_the_active_generation(self, published, centroid):
        with contextlib.redirect_stdout(io.StringIO()):
            embeddings, labels = visualization.load_embeddings_labels(SET_NAME, centroid=centroid)
        assert embeddings.shape == (2, 2)
        assert labels.tolist() == [0, 1]

    def test_print_stored_info_reads_the_active_generation(self, published):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            IndividualDataset.print_stored_info(SET_NAME)
        assert "Train Ids: 1" in output.getvalue()

    def test_fine_evaluation_consistency_check_finds_the_generation(self, published):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            fine_evaluation._show_dataset_embeddings_consistency(SET_NAME)
        assert "WARNING" not in output.getvalue()

    def test_train_psl_dry_run_finds_both_inputs(self, published):
        train_psl = import_train_psl()
        argv = ["train_psl", "--embeddings-name", SET_NAME, "--dry-run"]
        with patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            train_psl.main()
        assert "MISSING" not in output.getvalue()
        assert str(published.directory) in output.getvalue()

    def test_evaluation_stops_on_a_broken_pointer_instead_of_using_the_root(self, published):
        (published.directory.parent.parent / "current.json").write_text("{}", encoding="utf-8")
        args = argparse.Namespace(
            retrieval_method=0, rerank="none", use_psl=False, use_multimodal=False,
            stacked=False, min_instances=2, min_quality=None, verbose=False,
            random_guesses=False, mrr=False, map=False, cmc=False)
        with patch("src.retrieval.retrieval.Retriever") as retriever, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            evaluation.run_single_evaluation(SET_NAME, args, train_test=True)
        assert "ERROR:" in output.getvalue() and "to repair" in output.getvalue()
        retriever.assert_not_called()
