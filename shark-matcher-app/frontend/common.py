"""API routing, request dispatch, and response formatting utilities."""

import functools
import http
import re

from core import app_constants
import frontend
import frontend.user


class APIError(ValueError):
    """An explicitly safe client message and HTTP status, never a raw exception."""

    def __init__(self, message, status=http.HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


def require_fields(kwargs, *names):
    """Raise a 400 APIError naming the first required field that is absent, null, or empty."""
    for name in names:
        if kwargs.get(name) in (None, ""):
            raise APIError(f"Missing required field: {name}.")


def reject_unknown_fields(data, accepted_fields):
    """Raise a 400 APIError naming the first request field the handler does not accept."""
    unknown = sorted(set(data) - accepted_fields)
    if unknown:
        raise APIError(f"Unsupported field: {unknown[0]}.")


def dispatch(handler, path, data, target, authenticate, validate=None):
    """Run authentication, then field validation, then the target for an HTTP request."""
    if authenticate is not None:
        result = authenticate(handler, path, **data)
        if result[1] != http.HTTPStatus.OK:
            return result
    if validate is not None:
        validate(data)
    if authenticate is not None:
        user = frontend.user.get_authenticated_user(handler)
        if user is not None:
            data['user_id'] = user.id
            data['user_username'] = user.username
    return target(handler, path, **data)


def build_api_route(endpoint, handler_func, token_required=True, min_role_required=None,
                    methods=('POST',)):
    """Build a route tuple (regex, handler, auth_handler) for an API endpoint."""
    regex = rf'^{app_constants.API_PREFIX}/{re.escape(endpoint.rstrip("*"))}$'
    accepted_fields = getattr(handler_func, "accepted_fields", None)
    handler_func = wrap_api_handler(handler_func, endpoint)
    handler_func.allowed_methods = methods
    handler_func.accepted_fields = accepted_fields

    # Add the authentication function if a token is required.
    auth_func = None
    if token_required:
        # Create a partial function of authenticate_user with min_role_required pre-filled.
        auth_handler = functools.partial(frontend.user.authenticate_user,
                                         min_role_required=min_role_required)
        auth_func = wrap_api_handler(auth_handler, r'user/auth')

    return (regex, handler_func, auth_func)


def wrap_api_handler(handler_func, endpoint):
    """Wrap a handler function intended for an API endpoint."""
    return functools.partial(_call_api_handler, handler_func, endpoint)


def wrap_api_response(data, code, endpoint, message = None, success = True):
    """Wrap API payload data into the standard API response structure."""
    success = (success and (code == http.HTTPStatus.OK))

    return {
        'success': success,
        'message': message,
        'endpoint': endpoint,
        'content': data,
        'server-version': frontend.__version__,
    }


def _call_api_handler(handler_func, endpoint, handler, path, **kwargs):
    """Execute an API handler with normalized error handling and response formatting."""
    payload = None
    code = http.HTTPStatus.OK
    headers = {}
    message = None

    try:
        result = handler_func(handler, path, **kwargs)
        if result is None:
            return None

        payload, code, headers = result
    except APIError as ex:
        code = ex.status
        message = str(ex)
    except Exception as ex:  # pylint: disable=broad-exception-caught
        print(f"Error when handling API endpoint '{endpoint}': {type(ex).__name__}")
        code = http.HTTPStatus.INTERNAL_SERVER_ERROR
        message = 'The request could not be completed.'

    # Deal with defaults.
    if code is None:
        code = http.HTTPStatus.OK

    # API functions can send a message in the payload spot if the code is not OK.
    if ((code != http.HTTPStatus.OK) and (isinstance(payload, str))):
        message = payload
        payload = None

    if headers is None:
        headers = {}

    payload = wrap_api_response(payload, code, endpoint, message = message)

    # Mark header to read the full body even when non-JSON error content contains a message.
    headers['qgg-body'] = True
    headers['Cache-Control'] = 'no-store'

    return payload, code, headers
