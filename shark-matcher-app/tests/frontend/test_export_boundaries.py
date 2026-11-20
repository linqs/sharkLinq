"""Tests for trusted and browser-safe export boundaries."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from db.scripts import db_dump  # pylint: disable=import-error,no-name-in-module
from frontend import project  # pylint: disable=import-error


def _export_mode():
    """Return the explicit export-mode enumeration."""
    from db.scripts.db_dump import ExportMode  # pylint: disable=import-outside-toplevel
    return ExportMode


def test_browser_snapshot_export_forces_public_mode(tmp_path):
    """Request fields cannot promote a browser export to trusted mode."""
    export_mode = _export_mode()
    with patch.object(db_dump, "db_dump_tables", return_value=str(tmp_path)) as dump_tables:
        project._export_annotations(  # pylint: disable=protected-access
            format="snapshot",
            export_mode="trusted",
            public=False,
        )

    dump_tables.assert_called_once_with(export_mode=export_mode.PUBLIC)


def test_browser_annotations_export_forces_public_mode(tmp_path):
    """The table portion of a browser annotations export is always public."""
    export_mode = _export_mode()
    with patch.object(
        db_dump, "db_dump_annotations", return_value=str(tmp_path)
    ), patch.object(db_dump, "db_dump_tables") as dump_tables:
        project._export_annotations(  # pylint: disable=protected-access
            format="annotations",
            filtered_ids=[1, 2],
            export_mode="trusted",
        )

    dump_tables.assert_called_once_with(
        tables=[],
        output_directory=str(tmp_path),
        filtered_ids=[1, 2],
        export_mode=export_mode.PUBLIC,
    )


def test_database_dump_requires_an_explicit_mode():
    """No caller can receive sensitive fields by relying on a default."""
    with pytest.raises(TypeError):
        db_dump.db_dump_tables()  # pylint: disable=missing-kwoa


def test_database_dump_owns_one_consistent_read_transaction(tmp_path):
    """An ordinary dump shares one owned read transaction across its helpers."""
    connection = MagicMock()
    export_mode = _export_mode()
    with patch.object(
        db_dump.app_utils,
        "db_connect",
        return_value=connection,
    ), patch.object(
        db_dump.backend.db.user,
        "get_users",
        return_value=[],
    ) as get_users:
        db_dump.db_dump_tables(
            tables=["Users"],
            output_directory=tmp_path,
            export_mode=export_mode.TRUSTED,
        )

    connection.execute.assert_called_once_with("BEGIN")
    get_users.assert_called_once_with(ids=None, connection=connection)
    connection.rollback.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_public_mode_excludes_users_and_tokens(tmp_path):
    """Public mode omits credential and session tables even when requested."""
    export_mode = _export_mode()
    with patch.object(
        db_dump.backend.db.user,
        "get_users",
        return_value=[SimpleNamespace(
            id=1,
            name="Admin",
            email="admin@example.test",
            username="admin",
            role="admin",
            password="secret",
            salt="salt",
        )],
    ), patch.object(
        db_dump.backend.db.user,
        "get_tokens",
        return_value=[SimpleNamespace(
            id=1,
            token="token-secret",
            user=1,
            expires=2_000_604_800,
        )],
    ):
        db_dump.db_dump_tables(
            tables=["Users", "Tokens"],
            output_directory=str(tmp_path),
            export_mode=export_mode.PUBLIC,
        )

    assert not (tmp_path / "users.csv").exists()
    assert not (tmp_path / "tokens.csv").exists()


def test_trusted_mode_preserves_complete_server_side_dump(tmp_path):
    """Trusted administrative dumps retain credentials and session tokens."""
    export_mode = _export_mode()
    with patch.object(
        db_dump.backend.db.user,
        "get_users",
        return_value=[SimpleNamespace(
            id=1,
            name="Admin",
            email="admin@example.test",
            username="admin",
            role="admin",
            password="secret",
            salt="salt",
        )],
    ), patch.object(
        db_dump.backend.db.user,
        "get_tokens",
        return_value=[SimpleNamespace(
            id=1,
            token="token-secret",
            user=1,
            expires=2_000_604_800,
        )],
    ):
        db_dump.db_dump_tables(
            tables=["Users", "Tokens"],
            output_directory=str(tmp_path),
            export_mode=export_mode.TRUSTED,
        )

    assert "secret" in (tmp_path / "users.csv").read_text(encoding="utf-8")
    assert "token-secret" in (tmp_path / "tokens.csv").read_text(encoding="utf-8")


def test_trusted_mode_preserves_session_expiry(tmp_path):
    """Trusted administrative dumps retain the complete expiring session row."""
    export_mode = _export_mode()
    with patch.object(
        db_dump.backend.db.user,
        "get_tokens",
        return_value=[SimpleNamespace(
            id=1,
            token="token-digest",
            user=2,
            expires=2_000_604_800,
        )],
    ):
        db_dump.db_dump_tables(
            tables=["Tokens"],
            output_directory=str(tmp_path),
            export_mode=export_mode.TRUSTED,
        )

    token_row = (tmp_path / "tokens.csv").read_text(encoding="utf-8").strip()
    assert token_row == "1,token-digest,2,2000604800"
