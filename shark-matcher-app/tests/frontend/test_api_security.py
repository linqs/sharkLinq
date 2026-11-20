"""Exercise shared security and GET/POST parity through production HTTP handlers."""

import functools
import http.client
import json
import sqlite3
import urllib.parse

from core import app_constants
import backend.db.user
import frontend.common
import frontend.individual
import frontend.server
import frontend.user
from tests.browser.conftest import browser_app  # pylint: disable=unused-import
from tests.frontend.test_upload_batch import _make_batch_item


def test_get_search_and_match_post_parity(browser_app):
    """The same filters produce byte-equivalent logical result objects."""
    auth, legacy, _ = credentials(browser_app)
    cases = [{}, {'sex': 'F'}, {'keywords': 'merged boy'}, {'tags': 'test-tag'},
             {'labels': '*'}, {'locations': ['AN', 'FAR']}, {'qualities': 'good'},
             {'min_length': '10', 'max_length': '12'}, {'min_associated': '3'},
             {'min_time': '1318662000', 'max_time': '1350284400'},
             {'sex': 'F', 'labels': 'fresh'}, {'allow_unknown': 'false'},
             {'paginate': 'true', 'page': '1', 'page_size': '2', 'attribute': 'id', 'order': 'descending'}]
    for filters in cases:
        query = urllib.parse.urlencode(filters, doseq=True)
        get = request(browser_app, 'GET', '/api/individual/filter?' + query, headers=auth)
        post = request(browser_app, 'POST', '/api/individual/filter', {**legacy, **filters})
        assert get[0] == post[0] == 200, (filters, get, post)
        assert get[2] == post[2]
    image_name = browser_app.media[0][1]
    for method in app_constants.MATCH_METHOD_DICT:
        filters = {'image_name': image_name, 'method': method, 'sex': 'F'}
        get = request(browser_app, 'GET', '/api/match/filter?' + urllib.parse.urlencode(filters), headers=auth)
        post = request(browser_app, 'POST', '/api/match/filter', {
            **legacy, 'name': image_name, 'method': method, 'sex': 'F'})
        assert get[0] == post[0] == 200, (get, post)
        assert get[2] == post[2]
    with sqlite3.connect(browser_app.database) as connection:
        connection.execute('UPDATE Media SET name = ? WHERE id = ?', ('x', browser_app.media[0][0]))
    assert request(browser_app, 'GET', '/api/match/filter?image_name=x', headers=auth)[0] == 200
    assert request(browser_app, 'GET', '/api/match/filter?image_name=absent', headers=auth)[0] == 404


def test_all_routes_reject_unauthenticated_requests_and_write_gets(browser_app):
    """Every registered protected API route rejects callers before side effects."""
    auth, _, _ = credentials(browser_app)
    public = {'runtime/config', 'user/login', 'user/logout'}
    for _, target, authenticate in frontend.server.ROUTES:
        if not isinstance(target, functools.partial):
            continue
        endpoint = getattr(target, 'args')[1].rstrip('*')
        if endpoint not in public:
            status, _, _ = request(browser_app, 'POST', '/api/' + endpoint, {}, {'X-Shark-Demo': 'true'} if endpoint.startswith('demo/') else {})
            assert status == 401, (endpoint, status)
        if 'GET' not in getattr(target, 'allowed_methods'):
            status, headers, _ = request(browser_app, 'GET', '/api/' + endpoint, headers=auth)
            assert status == 405, endpoint
            assert headers['Allow'] == 'POST'
    assert request(browser_app, 'GET', '/api/unknown', headers=auth)[0] == 404
    assert request(browser_app, 'GET', '/api/individual/filterrrrr', headers=auth)[0] == 404


def test_signed_out_requests_get_401_before_any_input_check(browser_app):
    """Malformed queries and bodies from a signed-out caller are refused as unauthenticated, not as bad input."""
    for path in ('/api/match/filter', '/api/match/filter?sex=bogus', '/api/individual/filter?page_size=abc'):
        assert request(browser_app, 'GET', path)[0] == 401, path
    for endpoint, body in (('individual/post', {'sex': 'bogus'}), ('media/filter', {'ids': {'x': 1}})):
        assert request(browser_app, 'POST', '/api/' + endpoint, body)[0] == 401, endpoint
    auth, _, _ = credentials(browser_app)
    assert request(browser_app, 'GET', '/api/match/filter', headers=auth)[0] == 400


def test_image_cookie_is_read_only_revocable_and_private(browser_app):
    """The cookie serves images but cannot authenticate any API operation."""
    auth, legacy, cookie = credentials(browser_app)
    filename = browser_app.media[0][2]
    path = '/static/frontend-images/' + urllib.parse.quote(filename) + '?v=protected-1'
    assert request(browser_app, 'GET', path)[0] == 401
    status, headers, image = request(browser_app, 'GET', path, headers={'Cookie': cookie})
    assert status == 200 and image == (browser_app.images / filename).read_bytes()
    assert headers['Cache-Control'] == 'no-store'
    assert request(browser_app, 'HEAD', path, headers={'Cookie': cookie})[0] == 200
    for method, endpoint in [('GET', 'individual/filter'), ('POST', 'individual/post')]:
        assert request(browser_app, method, '/api/' + endpoint,
                       {} if method == 'POST' else None, {'Cookie': cookie})[0] == 401
    conflicting = dict(legacy, user_username='browser_viewer')
    assert request(browser_app, 'POST', '/api/individual/filter', conflicting, auth)[0] == 401
    assert request(browser_app, 'POST', '/api/user/logout', legacy)[0] == 200
    assert request(browser_app, 'GET', path, headers={'Cookie': cookie})[0] == 401
    assert request(browser_app, 'GET', '/api/individual/filter', headers=auth)[0] == 401


def test_all_existing_roles_can_use_get_and_images_are_only_images(browser_app):
    """Bearer access keeps the existing viewer-or-above read permission."""
    query = urllib.parse.urlencode({'image_name': browser_app.media[0][1]})
    for role in frontend.user.role_dict:
        auth, _, cookie = credentials(browser_app, role)
        assert request(browser_app, 'GET', '/api/individual/filter?sex=F', headers=auth)[0] == 200
        assert request(browser_app, 'GET', '/api/match/filter?' + query, headers=auth)[0] == 200
    (browser_app.images / 'private.txt').write_text('private marker')
    assert request(browser_app, 'GET', '/static/frontend-images/private.txt',
                   headers={'Cookie': cookie})[0] == 404
    assert request(browser_app, 'GET', '/static/frontend-images/' +
                   urllib.parse.quote(browser_app.media[0][2]), headers=auth)[0] == 200


def test_files_cannot_escape_read_write_or_export_roots(browser_app, tmp_path):
    """Traversal and symlinks fail without touching an outside marker."""
    auth, legacy, cookie = credentials(browser_app)
    outside = tmp_path / 'outside.jpeg'
    outside.write_bytes(b'private marker')
    (browser_app.images / 'escape.jpeg').symlink_to(outside)
    for suffix in ('../outside.jpeg', '%2e%2e/outside.jpeg', '%252e%252e/outside.jpeg',
                   'escape.jpeg', '..%5coutside.jpeg'):
        status, _, value = request(browser_app, 'GET', '/static/frontend-images/' + suffix,
                                   headers={'Cookie': cookie})
        assert status in (400, 404), (suffix, status)
        assert b'private marker' not in str(value).encode()
    for path in ('/static/../server.py', '/static/%2e%2e/server.py', '/static/__init__.py',
                 '/static/index.html?file_root=' + urllib.parse.quote(str(tmp_path))):
        status, _, value = request(browser_app, 'GET', path)
        assert b'private marker' not in str(value).encode()
        if 'file_root' not in path:
            assert status in (400, 404)
    item = _make_batch_item('../outside')
    assert request(browser_app, 'POST', '/api/media/upload_batch', {**legacy, 'items': [item]})[0] == 400
    assert request(browser_app, 'POST', '/api/media/save', {
        **legacy, 'filename': str(outside), 'data_url': item['data_url']})[0] == 400
    assert outside.read_bytes() == b'private marker'
    assert request(browser_app, 'GET', '/api/individual/filter?user_token=secret', headers=auth)[0] == 400


def test_query_validation_and_sanitized_errors(browser_app, monkeypatch, tmp_path):
    """Invalid input and unexpected exceptions never disclose internal details."""
    auth, legacy, _ = credentials(browser_app)
    for query in ('sex=F&sex=M', 'page=bad', 'page_size=1001', 'allow_unknown=maybe',
                  'min_length=20&max_length=10', 'return_objects=true', 'bogus=1',
                  'sex=invalid', 'sex=F&sex=', 'ids=bad', 'bogus='):
        assert request(browser_app, 'GET', '/api/individual/filter?' + query, headers=auth)[0] == 400
    assert request(browser_app, 'GET', '/api/%69ndividual/filter?bogus=1', headers=auth)[0] == 400
    assert request(browser_app, 'POST', '/api/individual/filter',
                   {**legacy, 'return_ids_only': True})[0] == 400
    private_path = str(tmp_path / 'private-database.db')
    def fail(*_args, **_kwargs):
        raise RuntimeError(private_path)
    monkeypatch.setattr(frontend.individual.backend.db.individual,
                        'get_individuals', fail)
    status, _, value = request(browser_app, 'POST', '/api/individual/filter', legacy)
    assert status == 500 and value['success'] is False
    assert private_path not in json.dumps(value)


def test_limits_and_upload_rollback(browser_app, monkeypatch):
    """Rejected batches leave database and image directory unchanged."""
    _, legacy, _ = credentials(browser_app)
    item = _make_batch_item('safe-upload')
    with sqlite3.connect(browser_app.database) as connection:
        before = connection.execute('SELECT COUNT(*) FROM Media').fetchone()
    files = sorted(browser_app.images.iterdir())
    monkeypatch.setattr(app_constants, 'MAX_UPLOAD_ITEMS', 1)
    assert request(browser_app, 'POST', '/api/media/upload_batch',
                   {**legacy, 'items': [item, item]})[0] == 413
    monkeypatch.setattr(app_constants, 'MAX_IMAGE_PIXELS', 1)
    assert request(browser_app, 'POST', '/api/media/upload_batch',
                   {**legacy, 'items': [item]})[0] == 413
    with sqlite3.connect(browser_app.database) as connection:
        assert connection.execute('SELECT COUNT(*) FROM Media').fetchone() == before
    assert sorted(browser_app.images.iterdir()) == files


def test_account_binding_and_server_authorship(browser_app):
    """Caller IDs cannot select another account or falsify action authors."""
    auth, legacy, _ = credentials(browser_app, 'viewer')
    other = browser_app.users['owner']
    assert request(browser_app, 'POST', '/api/user/get',
                   {**legacy, 'id': other.id, 'user_id': other.id}, auth)[0] == 403
    status, _, response = request(browser_app, 'POST', '/api/user/get', legacy)
    assert status == 200 and response['content']['id'] == browser_app.users['viewer'].id
    auth, legacy, _ = credentials(browser_app, 'collaborator')
    status, _, _ = request(browser_app, 'POST', '/api/media/upload_batch', {
        **legacy, 'user_id': other.id, 'items': [_make_batch_item('authorship-upload')]}, auth)
    assert status == 200
    with sqlite3.connect(browser_app.database) as connection:
        authors = connection.execute('SELECT author FROM Actions ORDER BY id DESC LIMIT 2').fetchall()
    assert authors == [(browser_app.users['collaborator'].id,)] * 2


def test_proposal_ownership_and_status_permissions(browser_app):
    """Collaborators create/delete their own proposals; admins change status."""
    _, author, _ = credentials(browser_app, 'collaborator')
    _, admin, _ = credentials(browser_app, 'admin')
    assert request(browser_app, 'POST', '/api/proposal/post', {
        **author, 'action': 'create', 'object': 'label', 'status': 'open',
        'original': None, 'final': {'id': -1, 'name': 'proposed label'},
    })[0] == 200
    with sqlite3.connect(browser_app.database) as connection:
        proposal_id, author_id = connection.execute(
            'SELECT id, author FROM Proposals ORDER BY id DESC LIMIT 1').fetchone()
    assert author_id == browser_app.users['collaborator'].id
    assert request(browser_app, 'POST', '/api/proposal/post', {
        **author, 'id': proposal_id, 'status': 'archive'})[0] == 403
    assert request(browser_app, 'POST', '/api/proposal/post', {
        **admin, 'id': proposal_id, 'status': 'archive'})[0] == 200
    assert request(browser_app, 'POST', '/api/proposal/delete', {
        **admin, 'id': proposal_id})[0] == 403
    assert request(browser_app, 'POST', '/api/proposal/delete', {
        **author, 'id': proposal_id})[0] == 200


def test_stored_unsafe_names_block_delete_rename_and_export(browser_app, tmp_path):
    """Even legacy malicious database filenames cannot escape image storage."""
    _, legacy, _ = credentials(browser_app)
    outside = tmp_path / 'private.jpeg'
    outside.write_bytes(b'private marker')
    media_id = browser_app.media[0][0]
    with sqlite3.connect(browser_app.database) as connection:
        individual = connection.execute('SELECT id FROM Individuals WHERE media = ?',
                                        (media_id,)).fetchone()[0]
        connection.execute('UPDATE Media SET filename = ? WHERE id = ?', (str(outside), media_id))
    assert request(browser_app, 'POST', '/api/media/delete', {**legacy, 'id': media_id})[0] == 400
    assert request(browser_app, 'POST', '/api/media/post', {
        **legacy, 'id': media_id, 'individual': individual, 'name': 'safe-rename'})[0] == 400
    status, _, response = request(browser_app, 'POST', '/api/data/export', {
        **legacy, 'content': 'all', 'format': 'snapshot', 'include_images': True})
    assert status == 500
    assert str(outside) not in json.dumps(response)
    assert outside.read_bytes() == b'private marker'
    with sqlite3.connect(browser_app.database) as connection:
        assert connection.execute('SELECT filename FROM Media WHERE id = ?',
                                  (media_id,)).fetchone() == (str(outside),)


def test_body_and_decoded_image_limits(browser_app, monkeypatch):
    """Body and decoded-image limits reject before writes; JSON must be an object."""
    auth, legacy, _ = credentials(browser_app)
    assert request(browser_app, 'POST', '/api/individual/filter', [], auth)[0] == 400
    assert request(browser_app, 'POST', '/api/individual/filter', headers=auth)[0] == 415
    monkeypatch.setattr(app_constants, 'MAX_JSON_BYTES', 1)
    assert request(browser_app, 'POST', '/api/individual/filter', legacy)[0] == 413
    for data in ('data:malformed', 'base64,invalid!', None):
        assert request(browser_app, 'POST', '/api/media/save', {
            **legacy, 'filename': 'invalid.jpeg', 'data_url': data})[0] == 400
    monkeypatch.setattr(app_constants, 'MAX_IMAGE_BYTES', 1)
    assert request(browser_app, 'POST', '/api/media/upload_batch', {
        **legacy, 'items': [_make_batch_item('oversized')]})[0] == 413


def request(app, method, path, body=None, headers=None):
    """Use raw HTTP paths, including adversarial paths browsers normalize away."""
    address = urllib.parse.urlsplit(app.url)
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=10)
    headers = dict(headers or {})
    if body is not None:
        body = json.dumps(body)
        headers['Content-Type'] = 'application/json'
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        value = json.loads(data) if data and response.getheader('Content-Type') == 'application/json' else data
        return response.status, dict(response.getheaders()), value
    finally:
        connection.close()


def credentials(app, role='owner'):
    """Obtain a real session through the existing password login contract."""
    status, headers, response = request(app, 'POST', '/api/user/login', {
        'username': f'browser_{role}',
        'password': backend.db.user.derive_client_password('browser-password'),
    })
    assert status == 200
    token = response['content']['token']
    return {'Authorization': f'Bearer {token}'}, {
        'user_username': f'browser_{role}', 'user_token': token,
        'user_id': app.users[role].id,
    }, headers['Set-Cookie'].split(';')[0]


def test_upload_refreshes_the_cached_dataset_stats(browser_app):
    """Stats requested right after an upload count the new image instead of a warm cached copy."""
    auth, legacy, _ = credentials(browser_app)
    before = request(browser_app, 'POST', '/api/data/stats', legacy, auth)[2]['content']
    assert request(browser_app, 'POST', '/api/media/upload_batch',
                   {**legacy, 'items': [_make_batch_item('AN26111901')]}, auth)[0] == 200
    after = request(browser_app, 'POST', '/api/data/stats', legacy, auth)[2]['content']
    assert 'AN26111901' in after['media_without_embedding_names']
    assert len(after['media_without_embedding_names']) == len(before['media_without_embedding_names']) + 1


def test_unknown_request_fields_are_refused_after_sign_in(browser_app):
    """A misspelled or unsupported field is a 400 naming it, instead of being silently ignored."""
    auth, legacy, _ = credentials(browser_app)
    for endpoint, body, field in (('user/get', {'usernames': 'not-a-list'}, 'usernames'),
                                  ('action/filter', {'limit': 'bad'}, 'limit')):
        status, _, response = request(browser_app, 'POST', '/api/' + endpoint, {**legacy, **body}, auth)
        assert status == 400, endpoint
        assert response['message'] == f'Unsupported field: {field}.'
    status, _, _ = request(browser_app, 'GET', '/api/individual/filter?unknown=1', headers=auth)
    assert status == 400
    assert request(browser_app, 'POST', '/api/user/get', legacy, auth)[0] == 200


def test_every_api_route_declares_the_fields_it_accepts():
    """New API handlers must list their fields so unknown ones are refused."""
    for regex, target, _ in frontend.server.ROUTES:
        if isinstance(target, functools.partial):
            assert getattr(target, 'accepted_fields', None), regex
