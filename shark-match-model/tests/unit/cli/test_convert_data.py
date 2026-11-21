import os
from unittest.mock import patch

import pytest

from src.core import constants
import src.cli.convert_data


def test_partition_strategy_per_image():
    """Verify random sampling per shark: one test image per individual."""
    ids = list(range(10))
    labels = [0]*5 + [1]*5  # Two individuals, 5 images each.

    train_idx, test_idx = src.cli.convert_data._partition_data(
        ids, labels, test_per_shark=1, full_clusters_removed=False, test_newest=False
    )

    assert len(test_idx) == 2
    test_labels = [labels[i] for i in test_idx]
    assert 0 in test_labels
    assert 1 in test_labels
    assert set(train_idx).isdisjoint(set(test_idx))


def test_partition_strategy_full_clusters():
    """Verify cluster-based split: entire individuals move to test as a unit."""
    ids = list(range(100))
    labels = []
    for i in range(10):
        labels.extend([i] * 10)  # 10 individuals x 10 images.

    train_idx, test_idx = src.cli.convert_data._partition_data(
        ids, labels, full_clusters_removed=True, test_split_ratio=0.2
    )

    # ratio 0.2 of 100 images = 2 individuals -> 20 test images.
    assert len(test_idx) == 20
    test_labels_set = {labels[i] for i in test_idx}
    train_labels_set = {labels[i] for i in train_idx}
    assert test_labels_set.isdisjoint(train_labels_set)


def test_partition_strategy_temporal(csv_convert_data):
    """Verify temporal split moves the newest individual (label 2, 5 images) to test."""
    ids = list(range(15))
    labels = [0]*5 + [1]*5 + [2]*5

    train_idx, test_idx = src.cli.convert_data._partition_data(
        ids, labels, test_newest=True, test_split_ratio=0.2
    )

    test_labels = {labels[i] for i in test_idx}
    assert 2 in test_labels   # Newest individual is in test.
    assert 0 not in test_labels
    assert len(test_idx) == 5  # All 5 images of the newest individual.


def test_source_csv_dir_sidecar_is_relative_inside_the_data_root(csv_convert_data, tmp_path):
    """The sidecar records a database-format-relative name so annotations move between machines."""
    fixture_dir = os.path.dirname(csv_convert_data)
    output_dir = tmp_path / "annotations"
    with patch.object(constants, "DB_DATA_DIR", fixture_dir):
        src.cli.convert_data.prepare_annotations(
            input_dir=csv_convert_data, output_directory=str(output_dir), seed=1)

    sidecar = output_dir / constants.SOURCE_CSV_DIR_SIDECAR
    assert sidecar.read_text(encoding="utf-8") == "convert_data"


def test_source_csv_dir_sidecar_is_absolute_outside_the_data_root(csv_convert_data, tmp_path):
    """A CSV directory outside every data root is recorded as an absolute path."""
    output_dir = tmp_path / "annotations"
    with patch.object(constants, "DB_DATA_DIR", str(tmp_path / "database-format")), \
            patch.object(constants, "BASE_DATA_DIR", str(tmp_path / "data")):
        src.cli.convert_data.prepare_annotations(
            input_dir=csv_convert_data, output_directory=str(output_dir), seed=1)

    sidecar = output_dir / constants.SOURCE_CSV_DIR_SIDECAR
    assert sidecar.read_text(encoding="utf-8") == os.path.abspath(csv_convert_data)


def _write_small_database(directory):
    """Write six media (ids 0-5) split over two source individuals, 100 and 101."""
    directory.mkdir()
    (directory / "media.csv").write_text(
        "".join(f"{i},hash{i},name{i}\n" for i in range(6)), encoding="utf-8")
    (directory / "individuals.csv").write_text(
        "".join(f"{i},{i},,,{i},{100 if i < 3 else 101},\n" for i in range(6)), encoding="utf-8")
    return str(directory)


def test_annotations_path_limits_the_prepared_media(tmp_path):
    """Only the media ids listed in the given annotations file are prepared."""
    input_dir = _write_small_database(tmp_path / "csvs")
    limit_file = tmp_path / "limit.tsv"
    limit_file.write_text("\n".join(f"{i}\tx.jpeg\t0" for i in (1, 2, 4)), encoding="utf-8")

    def prepared_ids(**options):
        output_dir = tmp_path / f"out{len(options)}"
        src.cli.convert_data.prepare_annotations(
            input_dir=input_dir, output_directory=str(output_dir), train_test=False,
            seed=1, **options)
        lines = (output_dir / constants.FULL_ANNOTATIONS).read_text(encoding="utf-8").splitlines()
        return sorted(int(line.split("\t")[0]) for line in lines)

    assert prepared_ids() == [0, 1, 2, 3, 4, 5]
    assert prepared_ids(annotations_file=str(limit_file)) == [1, 2, 4]


def test_command_line_passes_annotations_path_through(tmp_path):
    """--annotations-path reaches prepare_annotations instead of being dropped."""
    argv = ["convert_data", "--input-directory", "any", "--output-directory", "out",
            "--annotations-path", str(tmp_path / "limit.tsv")]
    with patch("sys.argv", argv), \
            patch("src.core.utils.print_script_banner"), \
            patch("src.core.utils.resolve_directory_path", side_effect=lambda path, _fallback: path), \
            patch.object(src.cli.convert_data, "prepare_annotations") as prepare:
        src.cli.convert_data.main()

    assert prepare.call_args.kwargs["annotations_file"] == str(tmp_path / "limit.tsv")


def test_missing_input_directory_exits_with_status_one(tmp_path, capsys):
    """A missing --input-directory prints one error line and exits 1 instead of a traceback."""
    argv = ["convert_data", "--input-directory", "does-not-exist", "--output-directory", "out"]
    with patch("sys.argv", argv), \
            patch.object(constants, "DB_DATA_DIR", str(tmp_path)), \
            patch.object(constants, "BASE_DATA_DIR", str(tmp_path)), \
            patch.object(constants, "ANNOTATIONS_DIR", str(tmp_path / "annotations")), \
            pytest.raises(SystemExit) as exited:
        src.cli.convert_data.main()
    assert exited.value.code == 1
    assert "Error: input directory not found" in capsys.readouterr().out
    assert not (tmp_path / "annotations").exists()


def test_unreadable_input_raises_before_creating_the_output(tmp_path):
    """prepare_annotations raises instead of returning silently, and leaves no output folder."""
    output_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="not found"):
        src.cli.convert_data.prepare_annotations(
            input_dir=str(tmp_path / "missing"), output_directory=str(output_dir))
    assert not output_dir.exists()


def test_conversion_error_in_main_exits_with_status_one(tmp_path):
    """A conversion error reaching main exits 1."""
    argv = ["convert_data", "--input-directory", "any", "--output-directory", "out"]
    with patch("sys.argv", argv), \
            patch("src.core.utils.print_script_banner"), \
            patch("src.core.utils.resolve_directory_path", side_effect=lambda path, _fallback: path), \
            patch.object(src.cli.convert_data, "prepare_annotations",
                         side_effect=ValueError("Could not read media.csv: bad")), \
            pytest.raises(SystemExit) as exited:
        src.cli.convert_data.main()
    assert exited.value.code == 1
