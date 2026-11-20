import pytest
from unittest.mock import MagicMock, patch

import src.retrieval.fine_evaluation
import src.retrieval.evaluation


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.get_dataset.return_value.get_ids.return_value = [101, 102, 103]
    return retriever


def _pin_to(retriever, csv_dir):
    """Configure retriever mock with a real dataset csv_dir for csv_store pinning."""
    dataset = MagicMock()
    dataset.csv_dir = csv_dir
    retriever._get_model_data.return_value = {'dataset': dataset}
    return retriever


@pytest.fixture
def mock_get_test_instances():
    with patch('src.retrieval.fine_evaluation.src.retrieval.evaluation._get_test_instances') as mock:
        # Returns: ids, filenames, labels, label_counts_dict.
        mock.return_value = ([101, 102, 103], ['f1', 'f2', 'f3'], [1, 1, 2], {1: 2, 2: 1})
        yield mock


def test_csv_sync_check_fail(mock_retriever, mock_get_test_instances, csv_fine_evaluation_partial):
    """When the CSV store is missing an ID that retrieval returned, an error is printed
    and empty lists are returned."""
    # csv_fine_evaluation_partial has only media 101 and 102; retrieval returns 101, 102, 103.
    _pin_to(mock_retriever, csv_fine_evaluation_partial)
    with patch('builtins.print'):
        ids, filenames, labels, counts = src.retrieval.fine_evaluation._get_filtered_test_instances(
            mock_retriever)

    assert ids == []
    assert filenames == []
    assert list(labels) == []
    assert counts == {}


def test_media_filtering(mock_retriever, mock_get_test_instances, csv_fine_evaluation_full):
    """Filtering by location returns only media whose location matches."""
    # csv_fine_evaluation_full: 101 (LocA), 102 (LocB), 103 (LocA).
    _pin_to(mock_retriever, csv_fine_evaluation_full)
    ids, filenames, labels, counts = src.retrieval.fine_evaluation._get_filtered_test_instances(
        mock_retriever, locations=["LocA"]
    )

    assert ids == [101, 103]
    assert filenames == ['f1', 'f3']
    assert list(labels) == [1, 2]


def test_individual_filtering(mock_retriever, mock_get_test_instances, csv_fine_evaluation_full):
    """Filtering by sex keeps only media linked to individuals of that sex."""
    # csv_fine_evaluation_full: individual 501 (F, media=101), 502 (M, media=102), 503 (F, media=103).
    _pin_to(mock_retriever, csv_fine_evaluation_full)
    ids, filenames, labels, counts = src.retrieval.fine_evaluation._get_filtered_test_instances(
        mock_retriever, sex="F"
    )

    # Only media belonging to female individuals (501, 503) should remain.
    assert set(ids) == {101, 103}


def test_compare_hits_dict_deltas():
    hits_dict_1 = {"1": [101], "5": [102], "10": [103]}
    hits_dict_2 = {"2": [101], "1": [102, 104], "10": [103]}

    sorted_deltas = src.retrieval.fine_evaluation.compare_hits_dict_by_delta(hits_dict_1, hits_dict_2)

    assert sorted_deltas[4] == [102]
    assert sorted_deltas[1] == [101]
    assert sorted_deltas[0] == [103]
    assert list(sorted_deltas.keys()) == [4, 1, 0]


def test_compute_average_k():
    """Average k across queries: (1*2 + 5*1 + 10*1) / 4 = 4.25."""
    hits_dict = {"1": [101, 102], "5": [103], "10": [104]}
    assert abs(src.retrieval.evaluation.compute_average_k(hits_dict) - 4.25) < 1e-9
    assert src.retrieval.evaluation.compute_average_k({}) == 0.0


def test_cross_model_failure_analysis(mock_get_test_instances):
    """cross_model_failure_analysis aggregates per-query k stats across models."""
    import torch

    call_count = [0]
    def mock_retrieval_k_for_hit(retriever, retrieval_method, ids, labels):
        if call_count[0] == 0:
            call_count[0] += 1
            return {"1": [101], "5": [102], "10": [103]}
        call_count[0] += 1
        return {"1": [101, 102], "8": [103]}

    with patch('src.retrieval.evaluation.retrieval_k_for_hit',
               side_effect=mock_retrieval_k_for_hit):
        with patch('src.retrieval.fine_evaluation.src.retrieval.retrieval.Retriever') as mock_ret_cls:
            mock_instance = MagicMock()
            mock_dataset = MagicMock()
            mock_dataset.get_labels.return_value = torch.tensor([1, 1, 2])
            mock_dataset.get_ids.return_value = [101, 102, 103]
            mock_instance.get_dataset.return_value = mock_dataset
            mock_ret_cls.return_value = mock_instance

            result = src.retrieval.fine_evaluation.cross_model_failure_analysis(
                ["parent/model_a", "parent/model_b"],
                retrieval_method=0, train_test=False, min_instances=2)

    assert result is not None
    assert len(result) == 3

    by_mean = sorted(result, key=lambda x: -x["mean_k"])
    assert by_mean[0]["id"] == 103
    assert by_mean[2]["id"] == 101

    by_std = sorted(result, key=lambda x: -x["std_k"])
    assert by_std[0]["id"] == 102
    assert by_std[2]["id"] == 101

    stats_by_id = {s["id"]: s for s in result}
    assert stats_by_id[101]["total_images"] == 2
    assert stats_by_id[102]["total_images"] == 2
    assert stats_by_id[103]["total_images"] == 1


def test_pin_csv_store_nonexistent_warns_and_raises(capsys, tmp_path):
    """When the dataset's csv_dir does not exist, warn and raise FileNotFoundError."""
    dataset = MagicMock()
    dataset.csv_dir = str(tmp_path / "foreign_machine" / "sm-data" / "thesis")
    with pytest.raises(FileNotFoundError, match="Use --csv-dir to provide the local path"):
        src.retrieval.fine_evaluation._pin_csv_store_from_dataset(dataset)
    captured = capsys.readouterr()
    assert "WARNING: The CSV directory recorded on this dataset was not found" in captured.err
    assert "--csv-dir" in captured.err


def test_pin_csv_store_with_csv_dir_override(csv_fine_evaluation_full, tmp_path):
    """csv_dir_override takes precedence over an invalid dataset.csv_dir."""
    dataset = MagicMock()
    dataset.csv_dir = str(tmp_path / "foreign_machine" / "sm-data" / "thesis")
    src.retrieval.fine_evaluation._pin_csv_store_from_dataset(
        dataset, csv_dir_override=csv_fine_evaluation_full)
    assert dataset.csv_dir == csv_fine_evaluation_full
    assert src.csv_store._loader._custom_db_dir == csv_fine_evaluation_full


def test_pin_csv_store_relative_path(csv_fine_evaluation_full):
    """Relative dataset.csv_dir resolves against constants.DB_DATA_DIR."""
    import os
    from src.core import constants
    dataset = MagicMock()
    dataset.csv_dir = os.path.basename(csv_fine_evaluation_full)
    with patch.object(constants, "DB_DATA_DIR", os.path.dirname(csv_fine_evaluation_full)):
        src.retrieval.fine_evaluation._pin_csv_store_from_dataset(dataset)
    assert dataset.csv_dir == csv_fine_evaluation_full


def test_individual_dataset_pickle_relative_paths(tmp_path):
    """IndividualDataset.__getstate__ serializes paths as relative for portability."""
    import os
    import pickle
    from src.core import constants
    from src.individual_dataset import IndividualDataset

    data_root = str(tmp_path / "sm-data")
    db_dir = os.path.join(data_root, "database-format")
    os.makedirs(os.path.join(db_dir, "thesis"))

    with patch.object(constants, "BASE_DATA_DIR", data_root), \
            patch.object(constants, "DB_DATA_DIR", db_dir):
        ds = IndividualDataset.__new__(IndividualDataset)
        ds.images_path = os.path.join(constants.BASE_DATA_DIR, "images", "images")
        ds.csv_dir = os.path.join(constants.DB_DATA_DIR, "thesis")
        ds.annotations_path = os.path.join(
            constants.BASE_DATA_DIR, "annotations", "baylearn", "full.tsv")

        state = ds.__getstate__()
        assert state["csv_dir"] == "thesis"
        assert state["images_path"] == "images/images"
        assert state["annotations_path"] == "annotations/baylearn/full.tsv"

        restored = pickle.loads(pickle.dumps(ds))
        assert restored.csv_dir == os.path.abspath(ds.csv_dir)


def test_get_test_instances_media_type_filtering(csv_fine_evaluation_full):
    """Filtering by media_type keeps only test instances matching that type."""
    import torch
    import src.retrieval.evaluation

    retriever = MagicMock()
    dataset = MagicMock()
    dataset.csv_dir = csv_fine_evaluation_full
    dataset.get_ids.return_value = [101, 102, 103]
    dataset.get_filenames.return_value = ['f1', 'f2', 'f3']
    dataset.get_labels.return_value = torch.tensor([1, 1, 1])
    retriever.get_dataset.return_value = dataset
    retriever._get_model_data.return_value = {'dataset': dataset}

    ids, _, _, _ = src.retrieval.evaluation._get_test_instances(
        retriever, train_test=False, min_instances=2, media_type="topshot"
    )
    assert ids == [101, 102, 103]

    ids_uw, _, _, _ = src.retrieval.evaluation._get_test_instances(
        retriever, train_test=False, min_instances=2, media_type="underwater"
    )
    assert ids_uw == []


