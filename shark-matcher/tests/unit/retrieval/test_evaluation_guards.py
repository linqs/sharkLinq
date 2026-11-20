"""Tests for embedding-set resolution and the empty-query guards in evaluation."""

import argparse
import contextlib
import io
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.core import constants
from src.core import utils
import src.retrieval.evaluation as evaluation

GENERATION_A = "0b9c4a4e-6a11-4d44-8b0a-1c0f0d3f1a11"
GENERATION_B = "7d2f1c88-3b6e-4a0f-9e55-2a7b6c4d8e22"


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").close()


@pytest.fixture(name="embeddings_root")
def embeddings_root_fixture(tmp_path):
    """An embeddings root with a versioned set (two generations) and a legacy flat set."""
    versioned = tmp_path / "parent" / "versioned"
    for generation in (GENERATION_A, GENERATION_B):
        _touch(str(versioned / "generations" / generation / "stored_embeddings.pt"))
    _touch(str(versioned / "current.json"))
    _touch(str(tmp_path / "parent" / "legacy" / "stored_embeddings.pt"))
    with patch.object(constants, "EMBEDDINGS_DIR", str(tmp_path)):
        yield tmp_path


class TestResolveEmbeddingDirs:
    def test_parent_resolves_each_set_once_without_its_generations(self, embeddings_root):
        resolved = utils.resolve_embedding_dirs("parent")
        assert resolved == [os.path.join("parent", "legacy"), os.path.join("parent", "versioned")]

    def test_versioned_set_resolves_to_itself(self, embeddings_root):
        assert utils.resolve_embedding_dirs("parent/versioned") == ["parent/versioned"]

    def test_legacy_flat_set_resolves_to_itself(self, embeddings_root):
        assert utils.resolve_embedding_dirs("parent/legacy") == ["parent/legacy"]


class _StubDataset:
    """Dataset with four labelled images and an empty test split."""

    def get_ids(self):
        return [1, 2, 3, 4]

    def get_filenames(self):
        return ["a", "b", "c", "d"]

    def get_labels(self):
        return torch.tensor([0, 0, 1, 1])

    def get_train_ids(self):
        return [1, 2, 3, 4]

    def get_test_ids(self):
        return []


class _StubRetriever:
    use_stacked = False
    use_multimodal = False
    use_psl = False

    def get_dataset(self):
        return _StubDataset()


class TestEmptyQueryGuards:
    def test_hits_at_k_reports_no_queries_instead_of_dividing_by_zero(self):
        with pytest.raises(evaluation.NoEvaluationQueriesError, match="split-strategy none"):
            evaluation.retrieval_hits_at_k(_StubRetriever(), train_test=True)

    def test_cmc_reports_no_queries_instead_of_dividing_by_zero(self):
        with pytest.raises(evaluation.NoEvaluationQueriesError):
            evaluation.retrieval_cmc(_StubRetriever(), train_test=True)

    def test_single_evaluation_skips_a_set_without_queries(self):
        args = argparse.Namespace(
            retrieval_method=0, min_instances=2, min_quality=None, verbose=False,
            random_guesses=False, mrr=False, map=False, cmc=False)
        output = io.StringIO()
        with patch("src.retrieval.retrieval.Retriever", return_value=_StubRetriever()), \
                contextlib.redirect_stdout(output):
            evaluation.run_single_evaluation("some/set", args, train_test=True)
        assert "Skipping 'some/set'" in output.getvalue()

    def test_queries_present_still_evaluates(self):
        stub = SimpleNamespace(get_dataset=lambda: _StubDataset())
        ids, _, _, _ = evaluation._get_test_instances(stub, train_test=False)
        assert len(ids) == 4


class TestRandomGuesses:
    """Only Hits@K is shuffled, so rank-based metrics are reported as n/a instead of real-ranking values."""

    def _run(self, random_guesses):
        args = argparse.Namespace(
            retrieval_method=0, min_instances=2, min_quality=None, verbose=False,
            random_guesses=random_guesses, mrr=True, map=True, cmc=True)
        output = io.StringIO()
        with patch("src.retrieval.retrieval.Retriever", return_value=_StubRetriever()), \
                patch.object(evaluation, "retrieval_hits_at_k",
                             return_value={"average_hit_at_1": 0.5}), \
                patch.object(evaluation, "retrieval_k_for_hit", return_value={"1": [7]}) as k_for_hit, \
                patch.object(evaluation, "retrieval_map", return_value=0.25) as map_metric, \
                patch.object(evaluation, "retrieval_cmc", return_value=0.75), \
                contextlib.redirect_stdout(output):
            evaluation.run_single_evaluation("some/set", args, train_test=False)
        return output.getvalue(), k_for_hit, map_metric

    def test_rank_metrics_are_not_applicable(self):
        output, k_for_hit, map_metric = self._run(random_guesses=True)
        for metric in ("Avg K", "MRR", "mAP", "AUC-CMC"):
            assert any(metric in line and evaluation.RANDOM_GUESSES_NOT_APPLICABLE in line
                       for line in output.splitlines()), metric
        k_for_hit.assert_not_called()
        map_metric.assert_not_called()

    def test_normal_run_still_reports_rank_metrics(self):
        output, _, _ = self._run(random_guesses=False)
        assert evaluation.RANDOM_GUESSES_NOT_APPLICABLE not in output
        assert "0.250" in output
