"""Handlers for web server API requests and static assets."""

import email.utils
import http
import http.cookies
import importlib.resources
import mimetypes
import os
import re
import urllib.parse

from core import app_constants
from core import app_utils
import backend.db.user
import frontend.common
import frontend.static

# Handler functions take (handler, path, **kwargs) and return (payload, code, headers).

# Static assets revalidate via If-Modified-Since ('no-cache') so deploys are never served stale.
STATIC_CACHE_CONTROL = 'no-cache'
# Protected images must be reauthorized on each read, including after logout.
IMAGE_CACHE_CONTROL = 'no-store'


def not_found(handler, path, **kwargs):
    """Handle 404 Not Found requests."""
    return "Route not found.", http.HTTPStatus.NOT_FOUND, None


def redirect(target):
    """Return a handler function that redirects requests to a target URL."""
    def handler_func(handler, path, **kwargs):
        return None, http.HTTPStatus.MOVED_PERMANENTLY, {'Location': target}

    return handler_func


def rewrite_prefix(prefix, target):
    """Return a handler function that rewrites a path's prefix and redirects."""

    def handler_func(handler, path, **kwargs):
        path = re.sub(prefix, target, path)
        return None, http.HTTPStatus.MOVED_PERMANENTLY, {'Location': path}

    return handler_func


def static(handler, path, **kwargs):
    """Serve a bundled static file from the package static directory."""
    # Note that the path is currently a URL path, and therefore separated with slashes.
    parts = urllib.parse.unquote(urllib.parse.urlsplit(path).path).split('/')

    # Build the static path skipping the '/static' part of the URL path.
    # importlib requires paths to use a '/' path seperator.
    static_path = '/'.join(parts[2:])

    return _serve_file(static_path, handler=handler)


def frontend_images(handler, path, **kwargs):
    """Serve authenticated shark images directly from IMAGES_DIR on the filesystem."""
    if handler.headers.get('Authorization'):
        result = frontend.user.authenticate_user(handler, path, min_role_required='viewer')
        if result[1] != http.HTTPStatus.OK:
            return result
    else:
        cookie = http.cookies.SimpleCookie()
        try:
            cookie.load(handler.headers.get('Cookie', ''))
            morsel = cookie.get(frontend.user.IMAGE_COOKIE)
            session, retry_after = frontend.user._AUTHENTICATION_LIMITER.verify_token(  # pylint: disable=protected-access
                handler, '', lambda: backend.db.user.get_session_user(morsel.value) if morsel else None)
        except http.cookies.CookieError:
            session, retry_after = None, None
        if retry_after is not None:
            return frontend.user._authentication_retry_response(retry_after)  # pylint: disable=protected-access
        if session is None or frontend.user.role_dict.get(session[0].role, -1) < 0:
            return 'Authentication required.', http.HTTPStatus.UNAUTHORIZED, frontend.user.image_cookie()
        handler.authenticated_user = session[0]
    parts = urllib.parse.unquote(urllib.parse.urlsplit(path).path).split('/')
    filename = '/'.join(parts[3:])
    if os.path.splitext(filename)[1].lower() not in {
            '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tif', '.tiff'}:
        return not_found(handler, path)
    return _serve_file(filename, handler=handler,
                       file_root=app_constants.IMAGES_DIR, cache_control=IMAGE_CACHE_CONTROL)


def _file_mtime(file_path):
    """Return a file's modification time, or None if it cannot be determined."""
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return None


def _client_copy_is_current(handler, mtime):
    """Return True if the client's If-Modified-Since header covers mtime."""
    if handler is None or mtime is None:
        return False

    ims_header = handler.headers.get('If-Modified-Since')
    if not ims_header:
        return False

    try:
        ims_time = email.utils.parsedate_to_datetime(ims_header).timestamp()
    except (TypeError, ValueError):
        return False

    # HTTP dates have one-second resolution, so allow a second of slack.
    return mtime <= ims_time + 1


def _serve_file(path, not_found_message = None, handler = None,
                resource_package=frontend.static, file_root=None,
                cache_control=STATIC_CACHE_CONTROL, **kwargs):
    """Serve a static file from disk or package, revalidating against client cache."""
    # Only public browser asset types may be served from the packaged tree.
    if file_root is None and os.path.splitext(path)[1].lower() not in {
            '.html', '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg',
            '.ico', '.woff', '.woff2', '.ttf'}:
        return not_found(handler, path)
    root = file_root if file_root is not None else importlib.resources.files(resource_package)
    try:
        resource = app_utils.resolve_served_path(root, path)
    except (ValueError, OSError):
        return not_found(handler, path)

    # File not found.
    if (not resource.is_file()):
        return not_found(handler, path)

    mtime = _file_mtime(str(resource))

    # Respond with 304 Not Modified if the client already has a current copy.
    if cache_control != 'no-store' and _client_copy_is_current(handler, mtime):
        return None, http.HTTPStatus.NOT_MODIFIED, {'Cache-Control': cache_control}

    data = resource.read_bytes()
    headers = {}

    mime_info = mimetypes.guess_type(path)
    if (mime_info is not None):
        headers['Content-Type'] = mime_info[0]

    headers['Cache-Control'] = cache_control
    if mtime is not None:
        headers['Last-Modified'] = email.utils.formatdate(mtime, usegmt=True)

    return data, http.HTTPStatus.OK, headers
