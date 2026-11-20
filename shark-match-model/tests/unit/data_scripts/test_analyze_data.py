
import importlib.util
import os
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "analyze_data",
    os.path.join(os.path.dirname(__file__), "../../../src/data_scripts/analyze_data.py")
)
analyze_data = importlib.util.module_from_spec(_spec)
sys.modules["analyze_data"] = analyze_data
_spec.loader.exec_module(analyze_data)


class TestAnalyzeData(unittest.TestCase):

    @patch('analyze_data.os.path.exists', return_value=True)
    @patch('analyze_data.os.listdir')
    @patch('src.core.utils.load_csv_file')
    @patch('src.core.utils.print_key_value_section')
    def test_show_annotations_data_integrity(self, mock_print_kv, mock_load_csv, mock_listdir, mock_exists):
        """
        Verify that show_annotations_data correctly identifies:
        - Annotations without images
        - Images without annotations
        - Correctly/Incorrectly named images
        """
        # Mock annotations with one matching image and one missing image.
        mock_load_csv.return_value = [
            ["1", "match.jpg", "10"],
            ["2", "missing.jpg", "20"]
        ]

        # Mock directory contents with matching, orphan, and misnamed files.
        mock_listdir.return_value = ["match.jpg", "orphan.jpg", "SHARK12345678.jpg", "bad_name.jpg"]

        # Execute image data analysis against default annotations path.
        analyze_data.show_image_data(annotations_path=None, verbose=False)

        # Verify report counts passed to print_key_value_section match expectations.
        args, _ = mock_print_kv.call_args
        report_dict = args[1]

        self.assertEqual(report_dict["Images in metadata"], 1)
        self.assertEqual(report_dict["Missing from disk"], 1)
        self.assertEqual(report_dict["Images with metadata"], 1)
        self.assertEqual(report_dict["Orphan images (no metadata)"], 3)
        self.assertEqual(report_dict["Correctly named images"], 1)
        self.assertEqual(report_dict["Incorrectly named images"], 3)

    @patch('src.core.utils.print_table')
    @patch('src.core.utils.print_key_value_section')
    @patch('analyze_data.csv_media.get_medias')
    @patch('src.core.utils.load_csv_file')
    def test_images_per_year_report_counts_utc_years_without_deprecation_warnings(
            self, mock_load_csv, mock_get_medias, mock_print_kv, mock_print_table):
        """Years come from UTC timestamps and the call raises no DeprecationWarning."""
        mock_load_csv.return_value = [["1"], ["2"], ["3"], ["4"]]
        new_year_eve_2019_utc = 1577836799  # 2019-12-31T23:59:59Z
        new_year_2020_utc = 1577836800  # 2020-01-01T00:00:00Z
        mock_get_medias.return_value = [
            SimpleNamespace(id=1, time=new_year_eve_2019_utc),
            SimpleNamespace(id=2, time=new_year_2020_utc),
            SimpleNamespace(id=3, time=new_year_2020_utc),
            SimpleNamespace(id=4, time=None),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            analyze_data.images_per_year_report(annotations_path="annotations.tsv")

        rows = mock_print_table.call_args.args[1]
        self.assertEqual(rows, [[2019, 1], [2020, 2]])
        summary = mock_print_kv.call_args.args[1]
        self.assertEqual(summary["Images without date"], 1)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.bar')
    @patch('src.core.utils.load_csv_file')
    def test_images_per_individual_report(self, mock_load_csv, mock_bar, mock_show):
        """
        Verify images_per_individual_report correctly aggregates counts.
        """
        # Mock annotations across three sharks with 2, 1, and 1 images respectively.
        mock_load_csv.return_value = [
            ["1", "i1.jpg", "1"],
            ["2", "i2.jpg", "1"],
            ["3", "i3.jpg", "2"],
            ["4", "i4.jpg", "3"]
        ]

        analyze_data.images_per_individual_report(plot=True)

        # Expected histogram: two sharks with 1 image and one shark with 2 images.
        args, _ = mock_bar.call_args
        x_data = list(args[0])
        y_data = list(args[1])

        self.assertIn(1, x_data)
        self.assertIn(2, x_data)
        self.assertEqual(y_data[x_data.index(1)], 2) # 2 sharks have 1 image.
        self.assertEqual(y_data[x_data.index(2)], 1) # 1 shark has 2 images.

        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
