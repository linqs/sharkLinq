"""Optional routes and request boundary for the disposable demo database."""

import functools
import http
import urllib.parse

from core import app_utils
import demo
import demo.runtime
import frontend.common
import frontend.handlers
import frontend.user


READ_ENDPOINTS = {
    "data/stats", "data/export", "individual/get", "individual/filter",
    "media/get", "media/filter", "object/get", "object/filter",
    "match/get", "match/filter", "action/filter", "proposal/get", "proposal/filter",
    "demo/status", "demo/keys",
}
WRITE_ENDPOINTS = {"individual/merge", "individual/unmatch", "demo/randomize", "demo/clear"}
LIVE_ENDPOINTS = {"user/auth", "user/login", "user/logout", "user/get", "user/filter"}


@app_utils.accepts()
def configuration(handler, _path, **_kwargs):
    """Expose server capability; browser state cannot enable the feature."""
    session = getattr(handler, "demo_session", None)
    return {"demo_enabled": session is not None,
            "demo_round": session.round_id if session else None}, http.HTTPStatus.OK, None


def asset(handler, path, **kwargs):
    """Serve the single packaged demo module through the existing asset handler."""
    if getattr(handler, "demo_session", None) is None:
        return frontend.handlers.not_found(handler, path)
    return frontend.handlers._serve_file(  # pylint: disable=protected-access
        "demo.js", handler=handler, resource_package=demo)


@app_utils.accepts()
def status(_handler, _path, **_kwargs):
    """Return counts for the shared round."""
    return demo.runtime.current().status(), http.HTTPStatus.OK, None


@app_utils.accepts("count")
def randomize(_handler, _path, **kwargs):
    """Reject invalid input before restoring the previous round."""
    try:
        result = demo.runtime.current().randomize(kwargs.get("count"))
    except ValueError as error:
        return str(error), http.HTTPStatus.BAD_REQUEST, None
    return result, http.HTTPStatus.OK, None


@app_utils.accepts()
def clear(_handler, _path, **_kwargs):
    """Clear the shared randomized round by restoring its startup baseline."""
    return demo.runtime.current().clear(), http.HTTPStatus.OK, None


@app_utils.accepts("source_id", "reveal")
def keys(_handler, _path, **kwargs):
    """Return key availability or explicitly revealed answers for a profile."""
    return demo.runtime.current().keys(
        kwargs.get("source_id"), reveal=kwargs.get("reveal") is True
    ), http.HTTPStatus.OK, None


def dispatch(handler, path, data, target, authenticate, validate=None):
    """Authorize against live accounts before entering a serialized demo scope."""
    endpoint = urllib.parse.urlsplit(path).path.removeprefix("/api/")
    requested = getattr(handler, "headers", {}).get("X-Shark-Demo") == "true"
    if not requested and not endpoint.startswith("demo/"):
        return frontend.common.dispatch(handler, path, data, target, authenticate, validate)
    session = getattr(handler, "demo_session", None)
    if session is None or not requested:
        return _error(endpoint, "Demo mode is unavailable.", http.HTTPStatus.NOT_FOUND)
    if endpoint in LIVE_ENDPOINTS:
        return frontend.common.dispatch(handler, path, data, target, authenticate, validate)
    if endpoint not in READ_ENDPOINTS | WRITE_ENDPOINTS:
        return _error(endpoint, "This operation is unavailable in Demo mode.", http.HTTPStatus.FORBIDDEN)
    if endpoint in WRITE_ENDPOINTS and handler.command != "POST":
        return _error(endpoint, "Use POST for demo changes.", http.HTTPStatus.METHOD_NOT_ALLOWED)
    # Proposers are named collaborators in the existing role model.
    auth = frontend.common.wrap_api_handler(functools.partial(
        frontend.user.authenticate_user,
        min_role_required="collaborator" if endpoint in WRITE_ENDPOINTS else "viewer",
    ), "user/auth")
    result = auth(handler, path, **data)
    if result[1] != http.HTTPStatus.OK:
        return result
    if validate is not None:
        validate(data)
    with session.scope():
        if handler.headers.get("X-Shark-Demo-Round") != session.round_id:
            return _error(endpoint, "The demo round changed. Reload the page before continuing.",
                          http.HTTPStatus.CONFLICT)
        # Always use the authenticated author, never a submitted identity.
        data["user_id"] = handler.authenticated_user.id
        data["user_username"] = handler.authenticated_user.username
        return target(handler, path, **data)


def _error(endpoint, message, code):
    return (frontend.common.wrap_api_response(None, code, endpoint, message), code,
            {"qgg-body": True})
