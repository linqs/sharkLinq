"""Tests for the weekly backup schedule in db.scripts.backup_runner."""

import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import db.scripts.backup_runner as backup_runner

SATURDAY_2300 = datetime.datetime(2026, 11, 14, 23, 0)
SUNDAY_0020 = datetime.datetime(2026, 11, 15, 0, 20)
SUNDAY_1200 = datetime.datetime(2026, 11, 15, 12, 0)
MONDAY_0900 = datetime.datetime(2026, 11, 16, 9, 0)


def _hours_slept(now):
    """Return the hours sleep_until_next_sunday would sleep from now."""
    with patch.object(backup_runner.time, "sleep") as sleep:
        backup_runner.sleep_until_next_sunday(now)
    return sleep.call_args.args[0] / 3600


@pytest.mark.parametrize(("now", "expected_hours"), [
    (SATURDAY_2300, 1.0),
    (SUNDAY_0020, 7 * 24 - 20 / 60),
    (SUNDAY_1200, 6.5 * 24),
    (MONDAY_0900, 5 * 24 + 15),
])
def test_sleep_ends_at_the_next_sunday_midnight(now, expected_hours):
    assert _hours_slept(now) == pytest.approx(expected_hours)


def _run_one_pass(now):
    """Run the backup loop once at a fixed time; return the dump calls and the sleep."""
    fake_datetime = SimpleNamespace(
        datetime=SimpleNamespace(now=lambda: now), timedelta=datetime.timedelta)
    with patch.object(backup_runner, "datetime", fake_datetime), \
            patch.object(backup_runner.db.scripts.db_dump, "db_dump_tables",
                         return_value="tables-dir") as tables, \
            patch.object(backup_runner.db.scripts.db_dump, "db_dump_annotations",
                         return_value="annotations-dir") as annotations, \
            patch.object(backup_runner.time, "sleep", side_effect=KeyboardInterrupt) as sleep:
        with pytest.raises(KeyboardInterrupt):
            backup_runner.backup_runner()
    return tables, annotations, sleep


def test_backup_runs_in_the_sunday_midnight_hour_with_trusted_durable_export():
    tables, annotations, sleep = _run_one_pass(SUNDAY_0020)
    tables.assert_called_once_with(
        export_mode=backup_runner.db.scripts.db_dump.ExportMode.TRUSTED, durable=True)
    annotations.assert_called_once_with()
    assert sleep.call_args.args[0] / 3600 == pytest.approx(7 * 24 - 20 / 60)


@pytest.mark.parametrize("now", [SATURDAY_2300, SUNDAY_1200, MONDAY_0900])
def test_no_backup_outside_the_window(now):
    tables, annotations, _ = _run_one_pass(now)
    tables.assert_not_called()
    annotations.assert_not_called()
