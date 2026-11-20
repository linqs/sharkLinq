"""Smoke tests for the self-locating cron installer."""

import os
from pathlib import Path
import shutil
import subprocess


APP_ROOT = Path(__file__).resolve().parents[2]


def test_setup_cron_preserves_existing_jobs_and_installs_weekly_job(tmp_path):
    """Decline avoids installation; repeated acceptance preserves other jobs."""
    checkout, environment, crontab_state, crontab_marker = _cron_test_context(
        tmp_path, "relocated-app"
    )
    wrapper = checkout / "scripts" / "cron_wrapper.sh"
    stale_project_dir = Path.home() / "previous-owner" / "app"
    wrapper.write_text(f'PROJECT_DIR="{stale_project_dir}"\n', encoding="utf-8")

    unrelated_entry = "15 4 * * * maintenance-task"
    crontab_state.write_text(
        f"MAILTO=ops@example.org\n{unrelated_entry}\n"
        f'0 0 * * 0 "{wrapper}"\n',
        encoding="utf-8",
    )
    python_marker = tmp_path / "python-working-directory"
    _write_executable(
        tmp_path / "bin" / "python3",
        f'#!/bin/sh\nprintf "%s\\n" "$PWD" > "{python_marker}"\nexit 7\n',
    )

    result = _run_setup(checkout, tmp_path, environment, reply="n\n")

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Cron job installation cancelled" in result.stdout
    assert not crontab_marker.exists()
    wrapper_source = wrapper.read_text(encoding="utf-8")
    assert 'dirname "${BASH_SOURCE[0]}"' in wrapper_source
    assert str(stale_project_dir) not in wrapper_source
    temporary_path = Path(
        next(
            line for line in result.stdout.splitlines()
            if line.startswith("Created temporary cron file: ")
        ).split(": ", 1)[1]
    )
    assert not temporary_path.exists()

    wrapper_result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert wrapper_result.returncode == 7
    assert python_marker.read_text(encoding="utf-8").strip() == str(checkout)
    log_source = (checkout / "data-out/logs/cron_output.txt").read_text(
        encoding="utf-8"
    )
    assert "failed with exit code 7" in log_source

    accepted = _run_setup(checkout, tmp_path, environment)
    assert accepted.returncode == 0, accepted.stderr or accepted.stdout
    assert "Cron job installed successfully" in accepted.stdout
    assert "every Sunday at midnight" in accepted.stdout
    assert "Existing cron entries were preserved" in accepted.stdout
    assert "To remove only this snapshot job, run: crontab -e" in accepted.stdout
    assert "crontab -r" not in accepted.stdout
    assert crontab_marker.exists()
    cron_source = crontab_state.read_text(encoding="utf-8")
    assert "MAILTO=ops@example.org" in cron_source
    assert unrelated_entry in cron_source
    assert f'0 0 * * 0 "{wrapper}"' in cron_source
    assert cron_source.count(str(wrapper)) == 1

    repeated = _run_setup(checkout, tmp_path, environment)
    assert repeated.returncode == 0, repeated.stderr or repeated.stdout
    repeated_source = crontab_state.read_text(encoding="utf-8")
    assert repeated_source.count(str(wrapper)) == 1
    assert repeated_source.count("BEGIN shark-matcher-app snapshot") == 1


def test_setup_cron_pins_the_detected_interpreter_for_cron(tmp_path):
    """The managed block records the setup interpreter and the wrapper runs it under a bare PATH."""
    checkout, environment, crontab_state, _ = _cron_test_context(
        tmp_path, "pinned-interpreter-checkout"
    )
    pinned_marker = tmp_path / "pinned-interpreter-arguments"
    pinned_python = tmp_path / "venv" / "bin" / "python3"
    pinned_python.parent.mkdir(parents=True)
    _write_executable(
        pinned_python,
        f'#!/bin/sh\nprintf "%s\\n" "$*" > "{pinned_marker}"\nexit 0\n',
    )
    _write_executable(
        tmp_path / "bin" / "python3",
        f'#!/bin/sh\nprintf "%s\\n" "{pinned_python}"\n',
    )

    result = _run_setup(checkout, tmp_path, environment)

    assert result.returncode == 0, result.stderr or result.stdout
    assert f"Using Python interpreter: {pinned_python}" in result.stdout
    assert f"SHARK_PYTHON={pinned_python}\n" in crontab_state.read_text(encoding="utf-8")

    cron_environment = {"PATH": "/usr/bin:/bin", "SHARK_PYTHON": str(pinned_python)}
    wrapper_result = subprocess.run(
        ["bash", str(checkout / "scripts" / "cron_wrapper.sh")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=cron_environment,
    )
    assert wrapper_result.returncode == 0, wrapper_result.stderr
    assert pinned_marker.read_text(encoding="utf-8").strip() == "-m scripts.shark_identification -s"


def test_setup_cron_carries_a_configured_data_root_into_cron(tmp_path):
    """A data-root override active during setup is preserved for cron's bare environment."""
    checkout, environment, crontab_state, _ = _cron_test_context(
        tmp_path, "data-root-checkout"
    )
    data_root = tmp_path / "shared-data"
    environment["SHARK_DATA_ROOT"] = str(data_root)

    result = _run_setup(checkout, tmp_path, environment)

    assert result.returncode == 0, result.stderr or result.stdout
    assert f"SHARK_DATA_ROOT={data_root}\n" in crontab_state.read_text(encoding="utf-8")


def test_setup_cron_initializes_when_no_crontab_exists(tmp_path):
    """The platform's no-crontab response permits a first safe installation."""
    checkout, environment, crontab_state, install_marker = _cron_test_context(
        tmp_path, "new-checkout", read_error="no crontab for test-user"
    )

    result = _run_setup(checkout, tmp_path, environment)

    assert result.returncode == 0, result.stderr or result.stdout
    assert install_marker.exists()
    assert "0 0 * * 0" in crontab_state.read_text(encoding="utf-8")


def test_setup_cron_aborts_when_existing_crontab_cannot_be_read(tmp_path):
    """An ambiguous crontab read failure cannot lead to replacement."""
    checkout, environment, _, install_marker = _cron_test_context(
        tmp_path, "unreadable-crontab-checkout", read_error="permission denied"
    )

    result = _run_setup(checkout, tmp_path, environment)

    assert result.returncode == 1
    assert "unable to read the existing crontab" in result.stdout
    assert "permission denied" in result.stderr
    assert not install_marker.exists()


def test_setup_cron_aborts_on_malformed_managed_markers(tmp_path):
    """An incomplete managed block cannot discard later cron entries."""
    checkout, environment, crontab_state, install_marker = _cron_test_context(
        tmp_path, "malformed-crontab-checkout"
    )
    crontab_state.write_text(
        "# BEGIN shark-matcher-app snapshot\n"
        "15 4 * * * maintenance-task\n",
        encoding="utf-8",
    )

    result = _run_setup(checkout, tmp_path, environment)

    assert result.returncode == 1
    assert "cron markers are malformed" in result.stdout
    assert not install_marker.exists()
    assert "maintenance-task" in crontab_state.read_text(encoding="utf-8")


def test_cron_sources_contain_no_machine_specific_checkout_path():
    """The installer and guide contain no developer home-directory path."""
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (APP_ROOT / "scripts" / "setup_cron.sh", APP_ROOT / "docs" / "cron-setup.md")
    )
    assert str(Path.home()) not in sources


def _write_executable(path, source):
    """Write one executable shell-test fixture."""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_setup(checkout, cwd, environment, reply="y\n"):
    """Run the copied setup script with one confirmation response."""
    return subprocess.run(
        ["bash", str(checkout / "scripts" / "setup_cron.sh")],
        cwd=cwd,
        input=reply,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def _write_fake_crontab(path):
    """Write a stateful crontab fixture controlled through environment paths."""
    _write_executable(
        path,
        """#!/bin/sh
if [ "$1" = "-l" ]; then
    if [ -n "$FAKE_CRONTAB_READ_ERROR" ]; then
        printf "%s\\n" "$FAKE_CRONTAB_READ_ERROR" >&2
        exit 1
    fi
    cat "$FAKE_CRONTAB_STATE"
    exit 0
fi
cp "$1" "$FAKE_CRONTAB_STATE"
touch "$FAKE_CRONTAB_INSTALL_MARKER"
""",
    )


def _cron_test_context(tmp_path, checkout_name, read_error=""):
    """Create one relocated checkout and stateful fake-crontab environment."""
    checkout = tmp_path / checkout_name
    checkout.mkdir()
    (checkout / "scripts").mkdir()
    shutil.copy2(APP_ROOT / "scripts" / "setup_cron.sh", checkout / "scripts")
    (checkout / "scripts" / "shark_identification.py").write_text("", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_crontab(fake_bin / "crontab")
    crontab_state = tmp_path / "current-crontab"
    crontab_state.touch()
    install_marker = tmp_path / "crontab-was-called"
    environment = os.environ.copy()
    environment.pop("SHARK_PYTHON", None)
    environment.pop("SHARK_DATA_ROOT", None)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["FAKE_CRONTAB_STATE"] = str(crontab_state)
    environment["FAKE_CRONTAB_INSTALL_MARKER"] = str(install_marker)
    environment["FAKE_CRONTAB_READ_ERROR"] = read_error
    return checkout, environment, crontab_state, install_marker
