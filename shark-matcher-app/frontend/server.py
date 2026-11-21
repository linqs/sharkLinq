"""API utility functions. Original author Eriq Augustine (edulinq)."""

import functools
import http
import http.server
import json
import math
import os
import re
import sqlite3
import sys
import urllib.parse

from core import app_constants
from core import app_utils
import demo.api
import demo.runtime

import frontend.action
import frontend.inference_client
import frontend.common
import frontend.handlers
import frontend.match
import frontend.media
import frontend.project
import frontend.proposal
import frontend.review
import frontend.individual
import frontend.user

ROUTES = [
    frontend.common.build_api_route(r'runtime/config', demo.api.configuration,
                                    token_required=False, methods=('GET', 'POST')),
    frontend.common.build_api_route(r'demo/status', demo.api.status, min_role_required='viewer'),
    frontend.common.build_api_route(r'demo/randomize', demo.api.randomize, min_role_required='collaborator'),
    frontend.common.build_api_route(r'demo/clear', demo.api.clear, min_role_required='collaborator'),
    frontend.common.build_api_route(r'demo/keys', demo.api.keys, min_role_required='viewer'),
    (r'^/demo/demo.js$', demo.api.asset, None),
    # Static routes.
    (r'^/$', frontend.handlers.redirect('/static/index.html'), None),
    (r'^/index.html$', frontend.handlers.redirect('/static/index.html'), None),
    (r'^/static$', frontend.handlers.redirect('/static/index.html'), None),
    (r'^/static/$', frontend.handlers.redirect('/static/index.html'), None),

    # Shark images live outside the package, so this route precedes the /static/ handler.
    (r'^/static/frontend-images/', frontend.handlers.frontend_images, None),

    # Bundled static asset routes.
    (r'^/static/', frontend.handlers.static, None),
    (r'^/js/', frontend.handlers.rewrite_prefix('^/js/', '/static/js/'), None),

    # Dataset and embeddings routes.
    frontend.common.build_api_route(r'data/export*', frontend.project.export_data,
                                    min_role_required='collaborator'),
    frontend.common.build_api_route(r'data/stats*', frontend.project.get_dataset_stats,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'embeddings/update*', frontend.project.update_embeddings,
                                    min_role_required='admin'),
    frontend.common.build_api_route(
        r'embeddings/status*', frontend.project.get_embeddings_update_status,
        min_role_required='viewer', methods=('GET', 'POST')),
    # Disabled until review tab is complete; uncomment below and adjust webui/review.js to restore.
    # frontend.common.build_api_route(r'review/get*', frontend.review.get_review,
    #                                 min_role_required='viewer'),

    # User routes.
    frontend.common.build_api_route(r'user/get', frontend.user.get_user,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'user/auth', frontend.user.authenticate_user,
                                    token_required=False),
    frontend.common.build_api_route(r'user/login', frontend.user.login_user,
                                    token_required=False),
    frontend.common.build_api_route(r'user/logout', frontend.user.logout_user,
                                    token_required=False),
    frontend.common.build_api_route(r'user/filter', frontend.user.filter_users,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'user/post', frontend.user.post_user,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'user/delete', frontend.user.delete_user,
                                    min_role_required='viewer'),

    # Individual routes.
    frontend.common.build_api_route(r'individual/get*', frontend.individual.get_individuals,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'individual/filter*', frontend.individual.filter_individuals,
                                    min_role_required='viewer', methods=('GET', 'POST')),
    frontend.common.build_api_route(r'individual/post*', frontend.individual.post_individual,
                                    min_role_required='collaborator'),
    frontend.common.build_api_route(r'individual/delete*', frontend.individual.delete_individual,
                                    min_role_required='admin'),
    frontend.common.build_api_route(r'individual/merge*', frontend.individual.merge_individuals,
                                    min_role_required='admin'),
    frontend.common.build_api_route(r'individual/unmatch*', frontend.individual.unmatch_individual,
                                    min_role_required='admin'),
    frontend.common.build_api_route(r'individual/reference*', frontend.individual.update_reference,
                                    min_role_required='admin'),

    # Media routes.
    frontend.common.build_api_route(r'media/get*', frontend.media.get_medias,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'media/filter*', frontend.media.filter_medias,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'media/post*', frontend.media.post_media,
                                    min_role_required='collaborator'),
    frontend.common.build_api_route(r'media/delete*', frontend.media.delete_media,
                                    min_role_required='admin'),
    frontend.common.build_api_route(r'media/save*', frontend.media.save_media,
                                    min_role_required='collaborator'),
    frontend.common.build_api_route(r'media/upload_batch*', frontend.media.upload_batch,
                                    min_role_required='collaborator'),

    # Object routes.
    frontend.common.build_api_route(r'object/get*', frontend.individual.get_objects,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'object/filter*', frontend.individual.filter_objects,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'object/post*', frontend.individual.post_object,
                                    min_role_required='admin'),
    frontend.common.build_api_route(r'object/delete*', frontend.individual.delete_object,
                                    min_role_required='admin'),

    # Action routes.
    frontend.common.build_api_route(r'action/filter*', frontend.action.filter_actions,
                                    min_role_required='viewer'),

    # Proposal routes.
    frontend.common.build_api_route(r'proposal/get*', frontend.proposal.get_proposals,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'proposal/filter*', frontend.proposal.filter_proposals,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'proposal/post*', frontend.proposal.post_proposal,
                                    min_role_required='collaborator'),
    frontend.common.build_api_route(r'proposal/delete*', frontend.proposal.delete_proposal,
                                    min_role_required='collaborator'),

    # Match routes.
    frontend.common.build_api_route(r'match/get*', frontend.match.get_matches,
                                    min_role_required='viewer'),
    frontend.common.build_api_route(r'match/filter*', frontend.match.filter_matches,
                                    min_role_required='viewer', methods=('GET', 'POST')),
]

_server = None


def _create_server(address, handler):
    """Create the threaded server with a listen backlog sized for a page's parallel requests."""
    server_class = type('AppServer', (http.server.ThreadingHTTPServer,),
                        {'request_queue_size': app_constants.LISTEN_BACKLOG})
    return server_class(address, handler)


def local_url(host, port):
    """Return the browser address for a server listening on host and port."""
    return f"http://{host or '127.0.0.1'}:{port}"


def run(project_dir, port=app_constants.DEFAULT_PORT, demo_enabled=False,
        host=app_constants.DEFAULT_HOST):
    """Start the local server."""
    if not os.path.isdir(project_dir):
        raise ValueError(f"Project directory does not exist: '{str(project_dir)}'.")

    _check_database()

    print(f"Starting server on port {str(port)}, serving project at '{project_dir}'.")
    print("If a browser window does not open, you may use the following link:")
    print(local_url(host, port))

    _handler.init(project_dir, demo_enabled=demo_enabled)

    global _server
    try:
        # Verify the inference service is reachable at startup.
        _handler.init_retriever()

        _server = _create_server((host, port), _handler)

        print("Now listening for requests.")
        _server.serve_forever()
    finally:
        if _server is not None:
            _server.server_close()
            _server = None
        if _handler.demo_session is not None:
            _handler.demo_session.close()
            _handler.demo_session = None

# pylint: disable=invalid-name, inconsistent-return-statements
class _handler(http.server.BaseHTTPRequestHandler):
    _project_dir = None
    demo_session = None

    @classmethod
    def init(cls, project_dir, demo_enabled=False):
        """Initialize the handler class with the project directory."""
        project_dir = os.path.abspath(project_dir)
        if not os.path.isdir(project_dir):
            raise ValueError('Project directory is not a directory or does not exist.')
        cls._project_dir = project_dir
        if cls.demo_session is not None:
            cls.demo_session.close()
        cls.demo_session = demo.runtime.DemoSession(app_constants.DB_PATH) if demo_enabled else None

    @classmethod
    def init_retriever(cls):
        """Verify the inference service is reachable at startup."""
        print("Connecting to inference service.")
        if not frontend.inference_client.health():
            print(f"WARNING: Inference service is not reachable at {app_constants.INFERENCE_SERVICE_URL}.")
            print("Match requests will fail until the service is available.")

    # pylint: disable=arguments-differ, unused-argument
    def log_message(self, fmt, *args):
        """Reduce the logging noise."""
        return

    def handle(self):
        """Override handle() to ignore dropped connections."""
        try:
            return http.server.BaseHTTPRequestHandler.handle(self)
        except BrokenPipeError:
            print("Connection closed on the client side.")

    def do_POST(self):
        """Handle POST requests."""
        self.handle_request(self._get_post_data)

    def do_GET(self):
        """Handle GET requests."""
        self.handle_request(self._get_get_data)

    do_HEAD = do_GET
    do_PUT = do_GET
    do_PATCH = do_GET
    do_DELETE = do_GET
    do_OPTIONS = do_GET

    def handle_request(self, data_handler):
        """Dispatch an incoming HTTP request through routing, authentication, and execution."""
        target_function = None
        auth_function = None
        response_headers = None
        self.authenticated_user = None

        try:
            url = urllib.parse.urlsplit(self.path)
            route_path = urllib.parse.unquote(url.path, errors='strict')
            if (url.scheme or url.netloc or '\\' in route_path or '%' in route_path
                    or any(part in ('.', '..') for part in route_path.split('/'))
                    or any(ord(char) < 32 or ord(char) == 127 for char in route_path)):
                raise frontend.common.APIError('Invalid request path.')
            query = urllib.parse.parse_qs(url.query)
            if {'user_token', 'user_username', 'password', 'token'} & query.keys():
                raise frontend.common.APIError('Send credentials in headers, not URLs.')
            # Find the API endpoint to call.
            target_function, auth_function = self._route(route_path)
            methods = getattr(target_function, 'allowed_methods', ('GET', 'HEAD'))
            if target_function == frontend.handlers.not_found:
                raise frontend.common.APIError('Route not found.', http.HTTPStatus.NOT_FOUND)
            if self.command not in methods:
                response_headers = {'Allow': ', '.join(methods)}
                raise frontend.common.APIError('Method not allowed.', http.HTTPStatus.METHOD_NOT_ALLOWED)

            # Handle by running the endpoint's auth and/or target functions.
            response_code = http.HTTPStatus.OK
            data = data_handler()
            # Internal keyword names are refused before authentication, which receives the same keywords.
            if {'return_objects', 'return_ids_only', 'connection', 'min_role_required'} & data.keys():
                raise frontend.common.APIError('Unsupported request parameter.')
            validate = functools.partial(
                _validate_request, route_path, getattr(target_function, 'accepted_fields', None),
                from_query=data_handler == self._get_get_data)

            payload, response_code, response_headers = demo.api.dispatch(
                self, route_path, data, target_function, auth_function, validate)
            if (response_code == http.HTTPStatus.OK and route_path.removeprefix(
                    f'{app_constants.API_PREFIX}/') in frontend.project.STATS_CHANGING_ENDPOINTS):
                frontend.project.invalidate_dataset_stats()

        except frontend.common.APIError as ex:
            response_code = ex.status
            payload = frontend.common.wrap_api_response(
                None, response_code, 'request', message=str(ex))
        except Exception as ex:  # pylint: disable=broad-exception-caught
            print(f"Request handling failed: {type(ex).__name__}")
            response_code = http.HTTPStatus.BAD_REQUEST
            payload = frontend.common.wrap_api_response(
                None, response_code, 'request', message='Invalid request.')

        # Prepare headers and payload for the response.
        headers = {'qgg-body': 'true', 'X-Content-Type-Options': 'nosniff'}
        if self.path.startswith('/api/') or response_code >= 400:
            headers['Cache-Control'] = 'no-store'

        # Copy the payload depending on the format it is in.
        if isinstance(payload, dict):
            payload = json.dumps(payload)
            headers['Content-Type'] = 'application/json'

        if isinstance(payload, str):
            payload = payload.encode(app_constants.ENCODING)

        if payload is not None:
            headers['Content-Length'] = len(payload)

        # Set the headers based on response headers.
        if response_headers is not None:
            for key, value in response_headers.items():
                headers[key] = value

        # Send response, headers, and payload.
        self.send_response(response_code)

        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()

        if payload is not None and self.command != 'HEAD':
            self.wfile.write(payload)

    def _route(self, path):
        """Resolve a request URL path against registered routes to find handler and auth functions."""
        path = urllib.parse.urlsplit(path.strip()).path
        auth = None
        target = frontend.handlers.not_found
        for (regex, handler_func, auth_func) in ROUTES:
            if re.search(regex, path) is not None:
                auth = auth_func
                target = handler_func
                break

        # If no target function is found, this endpoind is incorrect.
        return target, auth

    def _get_get_data(self):
        """Parse GET request data from the URL query string."""
        path = self.path.strip()
        url = urllib.parse.urlparse(path)

        raw_params = urllib.parse.parse_qs(url.query, keep_blank_values=True)
        params = {}

        for (key, values) in raw_params.items():
            if len(values) == 0:
                continue
            if len(values) == 1:
                params[key] = values[0]
            else:
                params[key] = values

        return params

    def _get_post_data(self):
        """Parse POST request data from the request body."""
        if self.headers.get('Transfer-Encoding'):
            raise frontend.common.APIError('Unsupported transfer encoding.')
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            raise frontend.common.APIError('Expected application/json.', http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        try:
            length = int(self.headers['Content-Length'])
        except (ValueError, TypeError) as ex:
            raise frontend.common.APIError('Content-Length is required.') from ex
        limit = (app_constants.MAX_UPLOAD_BYTES
                 if urllib.parse.urlsplit(self.path).path in ('/api/media/save', '/api/media/upload_batch')
                 else app_constants.MAX_JSON_BYTES)
        if length < 0 or length > limit:
            raise frontend.common.APIError('Request body exceeds the configured limit.',
                                           http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        payload = self.rfile.read(length).decode(app_constants.ENCODING)

        try:
            request = json.loads(payload)
        except Exception as ex:
            raise frontend.common.APIError('Payload is not valid JSON.') from ex

        if not isinstance(request, dict):
            raise frontend.common.APIError('Expected a JSON object.')

        return request


def _check_database():
    """Verify the database exists and has been initialised before starting the server."""
    db_path = app_constants.DB_PATH

    if not os.path.isfile(db_path):
        print(f"ERROR: Database file not found at '{db_path}'.")
        print("The database must be initialised before starting the server.")
        print("Run the following commands:")
        print(f"  python3 -m db.database init")
        print(f"  python3 -m db.database populate --all --data-path <path-to-dataset>")
        sys.exit(1)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='Individuals'"
            )
            if cursor.fetchone() is None:
                print(f"ERROR: Database at '{db_path}' exists but has not been initialised.")
                print("Run the following commands:")
                print(f"  python3 -m db.database init")
                print(f"  python3 -m db.database populate --all --data-path <path-to-dataset>")
                sys.exit(1)

            token_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(Tokens)")
            }
            if "expires" not in token_columns:
                print(f"ERROR: Database at '{db_path}' uses the legacy session schema.")
                print("Run the authentication migration before starting the server:")
                print("  python3 -m db.database migrate-auth")
                sys.exit(1)
    except sqlite3.DatabaseError as ex:
        print(f"ERROR: Could not open database at '{db_path}': {ex}")
        sys.exit(1)


def _is_number(value):
    """Return whether a value is a finite number or a numeric string."""
    if value is None or value == '':
        return True
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value)) if isinstance(value, (int, float, str)) else False
    except ValueError:
        return False


def _is_id(value):
    """Return whether a value is an integer identifier, its decimal string, or empty."""
    if isinstance(value, str):
        return value.strip() == '' or re.fullmatch(r'-?\d+', value.strip()) is not None
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _is_id_list(value):
    """Return whether a value is one identifier, a comma-separated string, or a list of them."""
    if isinstance(value, list):
        return all(_is_id(item) and item is not None for item in value)
    if isinstance(value, str):
        return all(_is_id(part) for part in value.split(','))
    return _is_id(value)


def _is_scalar(value):
    """Return whether a value is absent, text, or a number."""
    return value is None or isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _is_text(value):
    """Return whether a value is absent or text."""
    return value is None or isinstance(value, str)


def _is_text_list(value):
    """Return whether a value is text or a list of texts."""
    if isinstance(value, list):
        return all(isinstance(item, str) for item in value)
    return _is_text(value)


def _is_sex(value):
    """Return whether a value is absent, empty, or an accepted sex value in any letter case."""
    try:
        app_utils.normalize_sex(value)
    except ValueError:
        return False
    return True


def _is_entry_list(value):
    """Return whether a value is text or a list of texts, numbers, and objects."""
    if isinstance(value, list):
        return all(isinstance(item, dict) or (_is_scalar(item) and item is not None) for item in value)
    return _is_text(value)


def _is_flag(value):
    """Return whether a value is absent, a boolean, or text such as 'true'."""
    return value is None or isinstance(value, (bool, str))


def _is_object(value):
    """Return whether a value is absent or a JSON object."""
    return value is None or isinstance(value, dict)


# Accepted JSON shape per body key; keys not listed are left to their handlers.
_BODY_SHAPES = {
    **dict.fromkeys(('id', 'source', 'media', 'individual', 'reference', 'major_id', 'minor_id',
                     'user_id', 'source_id', 'original', 'final'), _is_id),
    **dict.fromkeys(('ids', 'medias'), _is_id_list),
    'sex': _is_sex,
    **dict.fromkeys(('name', 'method', 'model', 'content', 'format', 'attribute', 'order',
                     'object', 'action', 'status', 'author', 'email', 'username', 'role',
                     'password', 'review_type', 'data_url', 'image_name', 'user_username',
                     'user_token', 'export_mode', 'filename', 'hash', 'table'), _is_text),
    **dict.fromkeys(('type', 'quality', 'location', 'metadata', 'time', 'length'), _is_scalar),
    **dict.fromkeys(('min_length', 'max_length', 'min_time', 'max_time', 'min_associated',
                     'max_associated', 'page', 'page_size', 'count'), _is_number),
    **dict.fromkeys(('locations', 'qualities', 'max_qualities', 'names', 'keywords', 'roles'),
                    _is_text_list),
    **dict.fromkeys(('alternate_names', 'labels', 'tags'), _is_entry_list),
    **dict.fromkeys(('allow_unknown', 'get_full', 'get_source', 'paginate', 'include_images',
                     'public', 'sync', 'reveal', 'force'), _is_flag),
    'preview': _is_object,
}

# Route-specific overrides when a field name carries a different JSON shape on one endpoint.
_ROUTE_BODY_SHAPES = {
    '/api/proposal/post': {'original': _is_object, 'final': _is_object},
}


def _validate_request(route_path, accepted_fields, data, from_query=False):
    """Validate request fields after authentication, so signed-out callers learn nothing about them."""
    if accepted_fields is not None:
        frontend.common.reject_unknown_fields(data, accepted_fields)
    _validate_body(data, route_path=route_path)
    if from_query and route_path in ('/api/individual/filter', '/api/match/filter'):
        _validate_query(data, matching=route_path == '/api/match/filter')


def _validate_body(data, route_path=None):
    """Reject a request body or query whose known keys hold impossible JSON shapes."""
    route_shapes = _ROUTE_BODY_SHAPES.get(route_path, {})
    for key, value in data.items():
        shape = route_shapes.get(key, _BODY_SHAPES.get(key))
        if shape is not None and not shape(value):
            raise frontend.common.APIError(f'Invalid value for field: {key}.')


def _validate_query(params, matching=False):
    """Validate the two external GET contracts before shared handler dispatch."""
    lists = {'locations', 'qualities', 'max_qualities', 'alternate_names',
             'tags', 'labels', 'ids', 'names', 'keywords'}
    numbers = {'min_length', 'max_length', 'min_time', 'max_time',
               'min_associated', 'max_associated'}
    allowed = lists | numbers | {'sex', 'allow_unknown'}
    allowed |= ({'image_name', 'method', 'model'} if matching
                else {'paginate', 'page', 'page_size', 'attribute', 'order'})
    if params.keys() - allowed:
        raise frontend.common.APIError('Unsupported query parameter.')
    for key, value in params.items():
        if value == '' or (isinstance(value, list) and '' in value):
            raise frontend.common.APIError('Query parameters must not be empty.')
        if key not in lists and isinstance(value, list):
            raise frontend.common.APIError('Scalar parameters must occur once.')
        if key in numbers | {'page', 'page_size'}:
            try:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError()
                if key in {'page', 'page_size', 'min_associated', 'max_associated'}:
                    if not number.is_integer() or number < (1 if key.startswith('page') else 0):
                        raise ValueError()
                    params[key] = str(int(number))
            except (ValueError, TypeError) as ex:
                raise frontend.common.APIError('Invalid numeric query parameter.') from ex
        if key in {'paginate', 'allow_unknown'} and value.lower() not in {'true', 'false'}:
            raise frontend.common.APIError('Boolean query parameters must be true or false.')
    for lower in ('min_length', 'min_time', 'min_associated'):
        upper = lower.replace('min_', 'max_')
        if lower in params and upper in params and float(params[lower]) > float(params[upper]):
            raise frontend.common.APIError('Minimum must not exceed maximum.')
    if int(params.get('page_size', 1)) > app_constants.MAX_PAGE_SIZE:
        raise frontend.common.APIError('Page size exceeds the configured limit.')
    if params.get('order', 'ascending') not in ('ascending', 'descending'):
        raise frontend.common.APIError('Invalid sort order.')
    if 'sex' in params and (params['sex'] in (None, '') or not _is_sex(params['sex'])):
        raise frontend.common.APIError('Invalid sex filter.')
    if 'ids' in params:
        values = params['ids'] if isinstance(params['ids'], list) else [params['ids']]
        try:
            for value in values:
                for item in value.split(','):
                    int(item)
        except ValueError as ex:
            raise frontend.common.APIError('IDs must be integers.') from ex
    if matching:
        if not params.get('image_name'):
            raise frontend.common.APIError('image_name is required.')
        params.setdefault('method', 'nearest neighbor')
        if params['method'] not in app_constants.MATCH_METHOD_DICT:
            raise frontend.common.APIError('Unknown matching method.')
