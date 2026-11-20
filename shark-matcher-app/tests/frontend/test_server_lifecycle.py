"""Tests for app server startup options, signal handling, and shutdown cleanup."""

import signal
import sys
from unittest.mock import MagicMock

import pytest

from core import app_constants
import frontend.cli.serve
import frontend.server
from scripts import shark_identification


def _orchestrated_commands(monkeypatch, tmp_path, argv):
    """Run the orchestrator with argv and return the commands it would execute."""
    calls = []
    monkeypatch.setattr(sys, "argv", ["shark_identification", *argv])
    monkeypatch.setattr(app_constants, "OUT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(shark_identification.app_utils, "print_script_banner", lambda *_args: None)
    monkeypatch.setattr(shark_identification, "run_step",
                        lambda command, *_args, **_kwargs: calls.append(command))
    shark_identification.main()
    return calls


def test_orchestrator_forwards_port_only_when_given(monkeypatch, tmp_path):
    """The server command gains --port only when the operator supplies one."""
    default_command = _orchestrated_commands(monkeypatch, tmp_path, ["--run"])[0]
    custom_command = _orchestrated_commands(monkeypatch, tmp_path, ["--run", "--port", "12399"])[0]

    assert "--port" not in default_command
    assert custom_command == default_command + ["--port", "12399"]


def test_orchestrator_forwards_host_only_when_given(monkeypatch, tmp_path):
    """The server command gains --host only when the operator supplies one."""
    default_command = _orchestrated_commands(monkeypatch, tmp_path, ["--run"])[0]
    custom_command = _orchestrated_commands(
        monkeypatch, tmp_path, ["--run", "--host", "127.0.0.1"])[0]

    assert "--host" not in default_command
    assert custom_command == default_command + ["--host", "127.0.0.1"]


@pytest.mark.parametrize(("argv", "expected_host"), [
    ([], app_constants.DEFAULT_HOST),
    (["--host", "127.0.0.1"], "127.0.0.1"),
])
def test_server_cli_binds_the_requested_host(monkeypatch, tmp_path, argv, expected_host):
    """The server CLI listens on every interface by default and on --host when given."""
    bound = {}

    class FakeServer:
        """Record the bind address instead of opening a socket."""

        def __init__(self, server_address, _handler):
            bound["address"] = server_address

        def serve_forever(self):
            """Return immediately, as if the server had been stopped."""

        def server_close(self):
            """Release nothing, since no socket was opened."""

    monkeypatch.setattr(sys, "argv", ["server", str(tmp_path), "--no-browser", "--port", "0", *argv])
    monkeypatch.setattr(frontend.cli.serve.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(frontend.cli.serve.db.scripts.backup_runner,
                        "start_backup_runner_process", MagicMock)
    monkeypatch.setattr(frontend.cli.serve.app_utils, "print_script_banner", lambda *_args: None)
    monkeypatch.setattr(frontend.server, "_check_database", lambda: None)
    monkeypatch.setattr(frontend.server._handler, "init", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(frontend.server._handler, "init_retriever", lambda: None)
    monkeypatch.setattr(frontend.server.http.server, "ThreadingHTTPServer", FakeServer)

    assert frontend.cli.serve.main() == 0
    assert bound["address"] == (expected_host, 0)


def test_run_step_forwards_sigterm_to_the_running_step(monkeypatch):
    """SIGTERM sent to the orchestrator reaches its child and the prior handler is restored."""
    installed = {}
    process = MagicMock()

    def fake_signal(signum, handler):
        previous = installed.get(signum, "original")
        installed[signum] = handler
        return previous

    def wait():
        installed[signal.SIGTERM](signal.SIGTERM, None)
        return 0

    process.wait.side_effect = wait
    monkeypatch.setattr(shark_identification.signal, "signal", fake_signal)
    monkeypatch.setattr(shark_identification.subprocess, "Popen", lambda *_args, **_kwargs: process)

    shark_identification.run_step(["child"], "Step")

    process.send_signal.assert_called_once_with(signal.SIGTERM)
    assert installed[signal.SIGTERM] == "original"


def _interrupted_process(*, stops_on_its_own):
    """Return a step whose first wait is interrupted by Ctrl-C and which may or may not exit by itself."""
    process = MagicMock()
    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        if len(waits) == 1:
            raise KeyboardInterrupt
        if timeout is not None and not stops_on_its_own:
            raise shark_identification.subprocess.TimeoutExpired("child", timeout)
        return 0

    process.wait.side_effect = wait
    return process


@pytest.mark.parametrize("stops_on_its_own", [True, False])
def test_run_step_forwards_ctrl_c_only_when_the_step_keeps_running(monkeypatch, stops_on_its_own):
    """Ctrl-C sent only to the orchestrator still stops the step instead of hanging."""
    process = _interrupted_process(stops_on_its_own=stops_on_its_own)
    monkeypatch.setattr(shark_identification.subprocess, "Popen", lambda *_args, **_kwargs: process)

    shark_identification.run_step(["child"], "Step")

    if stops_on_its_own:
        process.send_signal.assert_not_called()
    else:
        process.send_signal.assert_called_once_with(signal.SIGINT)


def test_run_step_exits_with_the_failed_step_status(monkeypatch):
    """A failing step still stops the orchestrator with the step's exit status."""
    process = MagicMock()
    process.wait.return_value = 7
    monkeypatch.setattr(shark_identification.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(shark_identification.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(SystemExit) as raised:
        shark_identification.run_step(["child"], "Step")

    assert raised.value.code == 7


def test_sigterm_handler_unwinds_cleanly_and_ignores_repeats(monkeypatch, tmp_path):
    """The server's SIGTERM handler exits successfully through finally blocks instead of deadlocking."""
    installed = {}
    monkeypatch.setattr(frontend.cli.serve.signal, "signal",
                        lambda signum, handler: installed.__setitem__(signum, handler))
    monkeypatch.setattr(frontend.cli.serve.db.scripts.backup_runner,
                        "start_backup_runner_process", MagicMock)
    monkeypatch.setattr(frontend.cli.serve.frontend.server, "run", lambda *_args, **_kwargs: None)

    frontend.cli.serve.run_cli(base_dir=str(tmp_path), no_browser=True)
    handler = installed[signal.SIGTERM]

    with pytest.raises(SystemExit) as raised:
        handler(signal.SIGTERM, None)

    assert raised.value.code == 0
    assert installed[signal.SIGTERM] == signal.SIG_IGN


def test_ctrl_c_shuts_down_cleanly_and_stops_the_backup_runner(monkeypatch, tmp_path):
    """Ctrl-C ends the server like SIGTERM: cleanup runs and the CLI reports success."""
    backup_process = MagicMock()
    backup_process.is_alive.return_value = True
    monkeypatch.setattr(frontend.cli.serve.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(frontend.cli.serve.db.scripts.backup_runner,
                        "start_backup_runner_process", lambda: backup_process)

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(frontend.cli.serve.frontend.server, "run", interrupted)

    try:
        result = frontend.cli.serve.run_cli(base_dir=str(tmp_path), no_browser=True)
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C escaped run_cli instead of shutting down cleanly.")

    assert result == 0
    backup_process.terminate.assert_called_once()


def test_run_closes_demo_session_when_startup_fails(monkeypatch, tmp_path):
    """A failure before serving still releases the disposable demo session."""
    demo_session = MagicMock()

    def init(_project_dir, demo_enabled=False):
        frontend.server._handler.demo_session = demo_session if demo_enabled else None

    def unreachable_retriever():
        raise RuntimeError("inference service unavailable")

    monkeypatch.setattr(frontend.server, "_check_database", lambda: None)
    monkeypatch.setattr(frontend.server._handler, "init", init)
    monkeypatch.setattr(frontend.server._handler, "init_retriever", unreachable_retriever)

    with pytest.raises(RuntimeError):
        frontend.server.run(str(tmp_path), port=0, demo_enabled=True)

    demo_session.close.assert_called_once()
    assert frontend.server._handler.demo_session is None
    assert frontend.server._server is None
