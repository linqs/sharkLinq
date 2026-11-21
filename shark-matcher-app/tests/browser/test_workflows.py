"""Characterize real browser workflows before and after server hardening."""

import json
import re
import sqlite3
import time
import zipfile

import pytest
from playwright.sync_api import expect

from core import app_constants
import frontend.project


@pytest.mark.parametrize("role", ["viewer", "collaborator", "admin", "dev", "owner"])
def test_navigation_search_images_and_session(page, browser_app, login, role):
    """All roles retain navigation, rendered data, image access and reloads."""
    login(role)
    for route in ("search?sex=F", "history", "proposals", "settings", "upload"):
        page.goto(browser_app.url + "/#" + route)
        page.wait_for_load_state("networkidle")
        expect(page.locator("body")).to_have_attribute("data-page", route.split("?")[0])
        expect(page.locator(".loader")).not_to_be_visible()
    page.goto(browser_app.url + "/#search?sex=F")
    page.wait_for_load_state("networkidle")
    expect(page.locator(".results-item").first).to_be_visible()
    page.wait_for_function("Array.from(document.querySelectorAll('.content img')).some(i => i.complete && i.naturalWidth > 0)")
    page.reload()
    page.wait_for_load_state("networkidle")
    expect(page.locator(".results-item").first).to_be_visible()
    page.goto(browser_app.url + "/#match?name=0&method=nearest%20neighbor&model=ViT")
    page.wait_for_load_state("networkidle")
    expect(page.locator(".match.query-column")).not_to_be_empty()
    expect(page.locator(".match.results-column")).not_to_be_empty()


def test_match_failure_shows_an_inline_error_without_a_dialog(page, browser_app, login):
    """A failed match request leaves a visible message in the results, not a blocking alert."""
    login()
    dialogs = []
    page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
    page.route("**/api/match/filter", lambda route: route.fulfill(
        status=502, content_type="application/json",
        body='{"success": false, "message": "Inference service unavailable."}'))
    page.goto(browser_app.url + "/#match?name=0&method=nearest%20neighbor&model=ViT")
    expect(page.locator(".match-error")).to_contain_text("Error executing filter match search")
    assert dialogs == []


def test_leaving_a_page_mid_request_raises_no_dialog(page, browser_app, login):
    """Requests the browser cancels while leaving a page are not reported as errors."""
    login()
    dialogs = []
    page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
    page.evaluate("Object.keys(localStorage).filter(k => k.startsWith('cache.'))"
                  ".forEach(k => localStorage.removeItem(k))")
    page.route("**/api/data/stats", lambda route: None)
    with page.expect_request("**/api/data/stats"):
        page.reload()
    page.goto(browser_app.url + "/#settings")
    expect(page.locator("body")).to_have_attribute("data-page", "settings")
    page.wait_for_timeout(500)
    assert dialogs == []


def test_navigation_tabs_work_from_the_keyboard(page, browser_app, login):
    """Each tab is a focusable button that Enter and Space activate."""
    login()
    page.goto(browser_app.url + "/#search")
    page.wait_for_load_state("networkidle")
    for key, tab, route in (("Enter", "match", "match"), ("Space", "search", "search")):
        locator = page.get_by_role("button", name=tab.capitalize(), exact=True)
        locator.focus()
        page.keyboard.press(key)
        expect(page.locator("body")).to_have_attribute("data-page", route)


def test_password_fields_sit_in_forms_and_enter_signs_in(page, browser_app):
    """Every password input has a form ancestor, and Enter in the password field signs in once."""
    page.goto(browser_app.url + "/#login")
    page.locator(".login.username-input").fill("browser_owner")
    page.locator(".login.password-input").fill("browser-password")
    assert page.locator("input[type=password]:not(form input)").count() == 0
    requests = []
    page.on("request", lambda request: requests.append(request.url)
            if request.url.endswith("/api/user/login") else None)
    page.locator(".login.password-input").press("Enter")
    page.wait_for_function("location.hash !== '#login'")
    page.wait_for_load_state("networkidle")
    assert len(requests) == 1
    page.goto(browser_app.url + "/#settings")
    page.locator(".user.properties-button").click()
    assert page.locator("input[type=password]:not(form input)").count() == 0


@pytest.mark.parametrize("route", [
    "", "search?sex=F", "match?name=0&method=nearest%20neighbor&model=ViT",
    "upload", "proposals", "history", "settings", "individual?name=0"])
def test_pages_do_not_scroll_sideways_on_a_phone(page, browser_app, login, route):
    """At 390 px wide no page is wider than the screen."""
    page.set_viewport_size({"width": 390, "height": 844})
    login()
    page.goto(browser_app.url + "/#" + route)
    page.wait_for_load_state("networkidle")
    assert page.evaluate("document.documentElement.scrollWidth") <= 390


@pytest.mark.parametrize("width", [390, 1024, 1280, 1440])
@pytest.mark.parametrize("route", ["search", "history", "proposals"])
def test_sort_button_is_not_covered_by_the_search_box(page, browser_app, login, route, width):
    """The sort button's centre is clickable and the controls never overlap the search box."""
    page.set_viewport_size({"width": width, "height": 900})
    login()
    page.goto(browser_app.url + "/#" + route)
    sort = page.locator(".sort-container > .sort-pill-btn")
    expect(sort).to_be_visible()
    box = sort.bounding_box()
    hit = page.evaluate(
        "([x, y]) => !!document.elementFromPoint(x, y)?.closest('.sort-container')",
        [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2])
    assert hit
    controls = page.locator(".controls-right-side").bounding_box()
    search = page.locator(".smart-search-container").bounding_box()
    overlaps = (controls["x"] < search["x"] + search["width"] and search["x"] < controls["x"] + controls["width"]
                and controls["y"] < search["y"] + search["height"] and search["y"] < controls["y"] + controls["height"])
    assert not overlaps
    assert page.evaluate("document.documentElement.scrollWidth") <= width


def test_embeddings_button_is_disabled_only_in_demo_mode(page, browser_app, login):
    """An admin keeps the Update Embeddings button outside Demo mode and loses it inside."""
    login("admin")
    page.goto(browser_app.url + "/#home")
    button = page.locator(".embeddings-button")
    expect(button).to_be_visible()
    expect(button).not_to_have_attribute("aria-disabled", "true")
    page.locator(".mode-selector").select_option("demo")
    expect(page.locator(".embeddings-button")).to_have_attribute("aria-disabled", "true")


def test_page_load_requests_no_missing_static_files(page, browser_app):
    """Loading the app never requests a /static/ file the server cannot serve."""
    failed = []
    page.on("response", lambda response: failed.append((response.url, response.status))
            if "/static/" in response.url and response.status >= 400 else None)
    page.goto(browser_app.url + "/#login")
    page.wait_for_load_state("networkidle")
    assert failed == []


def test_login_failure_and_logout(page, browser_app, login):
    """Incorrect login and logout retain their visible behavior."""
    dialogs = []
    page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.accept()))
    page.goto(browser_app.url + "/#login")
    page.locator(".login.username-input").fill("browser_owner")
    page.locator(".login.password-input").fill("incorrect")
    page.locator(".login.submit").click()
    page.wait_for_function("!document.querySelector('.loader').classList.contains('active')")
    page.wait_for_load_state("networkidle")
    assert dialogs == ["Incorrect username/password combination."]
    login()
    page.locator(".logout").click()
    page.wait_for_url("**#login")
    assert page.evaluate("localStorage.getItem('SHARKMATCHER.CREDENTIALS')") is None


def test_profile_and_user_management(page, browser_app, login):
    """Settings updates and owner account management retain their form flows."""
    login()
    page.on('dialog', lambda dialog: dialog.accept())
    page.goto(browser_app.url + '/#settings')
    page.locator('.user.properties-button').click()
    page.locator('.settings.name-input').fill('Updated owner')
    with page.expect_response('**/api/user/post') as updated:
        page.locator('.user.properties.save').click()
    assert updated.value.status == 200
    with sqlite3.connect(browser_app.database) as connection:
        assert connection.execute('SELECT name FROM Users WHERE id = ?',
                                  (browser_app.users['owner'].id,)).fetchone() == ('Updated owner',)
    page.locator('.user-creation-toggle-btn').click()
    for field, value in {'name': 'Created User', 'email': 'created@example.test',
                         'username': 'created_user', 'password': 'created-password',
                         'verify': 'created-password'}.items():
        page.locator(f'.create-user.{field}-input').fill(value)
    with page.expect_response('**/api/user/post') as created:
        page.locator('.create-user.save').click()
    assert created.value.status == 200
    page.reload()
    page.locator('.user-list-button').click()
    card = page.locator('.results-item').filter(has_text='created_user')
    expect(card).to_be_visible()
    with page.expect_response('**/api/user/delete') as deleted:
        card.locator('.object.delete').click()
    assert deleted.value.status == 200
    with sqlite3.connect(browser_app.database) as connection:
        assert connection.execute("SELECT id FROM Users WHERE username = 'created_user'").fetchone() is None
    page.wait_for_load_state('networkidle')
    page.locator('.user.properties-button').click()
    page.locator('.settings.password-input').fill('updated-password')
    page.locator('.settings.verify-input').fill('updated-password')
    with page.expect_response('**/api/user/post') as password_update:
        page.locator('.user.properties.save').click()
    assert password_update.value.status == 200
    page.locator('.logout').click()
    page.wait_for_url('**#login')
    login(password='updated-password')
    expect(page.locator('.logout')).to_be_visible()


def test_upload_duplicate_and_export(page, browser_app, login):
    """A real file upload, duplicate rejection and filtered ZIP download work."""
    login()
    dialogs = []
    page.on('dialog', lambda dialog: (dialogs.append(dialog.message), dialog.accept()))
    image = (browser_app.images / browser_app.media[0][2]).read_bytes()
    upload = {'name': 'browser-upload.jpeg', 'mimeType': 'image/jpeg', 'buffer': image}
    page.goto(browser_app.url + '/#upload')
    page.locator('.upload.file-input').set_input_files(upload)
    with page.expect_response('**/api/media/upload_batch') as response:
        page.locator('.upload.save').click()
    assert response.value.status == 200
    page.wait_for_url('**#search?**')
    with sqlite3.connect(browser_app.database) as connection:
        count = connection.execute('SELECT COUNT(*) FROM Media').fetchone()
    page.goto(browser_app.url + '/#upload')
    page.locator('.upload.file-input').set_input_files(upload)
    with page.expect_response('**/api/media/upload_batch') as duplicate:
        page.locator('.upload.save').click()
    assert duplicate.value.status == 409
    with sqlite3.connect(browser_app.database) as connection:
        assert connection.execute('SELECT COUNT(*) FROM Media').fetchone() == count
    page.goto(browser_app.url + '/#search?sex=F')
    page.locator('.export-include-images-checkbox').check()
    with page.expect_download() as download:
        page.locator('.export-filtered-button').click()
    with zipfile.ZipFile(download.value.path()) as archive:
        assert any(name.endswith('.csv') for name in archive.namelist())
        assert any(name.endswith('.jpeg') for name in archive.namelist())


@pytest.mark.parametrize("letter, sex", [("f", "F"), ("m", "M"), ("u", "Unknown"), ("x", "")])
def test_home_route_and_lowercase_sex_suffix_in_upload_names(page, browser_app, login, letter, sex):
    """#home opens Home like the empty route, and a filename sex letter maps to a Sex selector value."""
    login()
    page.goto(browser_app.url + '/#home')
    expect(page.locator('.embeddings-button')).to_be_visible()
    expect(page.get_by_text('Location Not Found')).to_have_count(0)

    image = (browser_app.images / browser_app.media[0][2]).read_bytes()
    page.goto(browser_app.url + '/#upload')
    page.locator('.upload.file-input').set_input_files(
        {'name': f'AB12345678_{letter}.jpeg', 'mimeType': 'image/jpeg', 'buffer': image})
    expect(page.locator('#media-name-input-0')).to_have_value('AB12345678')
    expect(page.locator('#media-sex-input-0')).to_have_value(sex)


def test_edit_and_demo_round(page, browser_app, login):
    """Individual property writes and isolated demo rounds keep working."""
    login()
    page.on('dialog', lambda dialog: dialog.accept())
    page.goto(browser_app.url + '/#individual?name=0')
    page.locator('.properties-button').first.click()
    page.locator('.length-input').first.fill('123')
    with page.expect_response('**/api/individual/post') as edited:
        page.locator('.properties.save').first.click()
    assert edited.value.status == 200
    page.wait_for_load_state('networkidle')
    page.locator('.properties-button').first.click()
    expect(page.locator('.length-input').first).to_have_value('123')
    page.locator('.mode-selector').select_option('demo')
    page.locator('.demo-toggle').click()
    page.locator('#demo-count').fill('1')
    with page.expect_response('**/api/demo/randomize') as randomized:
        page.locator('.demo-randomize').click()
    assert randomized.value.status == 200
    page.wait_for_load_state('networkidle')
    page.locator('.demo-toggle').click()
    with page.expect_response('**/api/demo/clear') as cleared:
        page.locator('.demo-clear').click()
    assert cleared.value.status == 200


def test_expired_session_cannot_reload_images(page, browser_app, login):
    """Reloading with expired browser credentials returns to the login form."""
    login('viewer')
    page.goto(browser_app.url + '/#search?sex=F')
    expect(page.locator('.results-item').first).to_be_visible()
    cookies = page.context.cookies()
    cookie = next(value for value in cookies if value['name'] == 'shark_image_session')
    assert cookie['httpOnly'] and cookie['sameSite'] == 'Lax'
    assert cookie['path'] == '/static/frontend-images/'
    with sqlite3.connect(browser_app.database) as connection:
        connection.execute('UPDATE Tokens SET expires = 1')
    page.on('dialog', lambda dialog: dialog.accept())
    page.reload()
    expect(page.locator('.login.submit')).to_be_visible()


def test_proposal_create_archive_and_delete(page, browser_app, login):
    """Proposal controls follow the existing author and administrator rules."""
    login('collaborator')
    page.on('dialog', lambda dialog: dialog.accept())
    page.goto(browser_app.url + '/#individual?name=0')
    page.locator('.properties-button').first.click()
    page.locator('.length-input').first.fill('123')
    with page.expect_response('**/api/proposal/post') as created:
        page.locator('.properties.save').first.click()
    assert created.value.status == 200
    with sqlite3.connect(browser_app.database) as connection:
        proposal_id = connection.execute('SELECT MAX(id) FROM Proposals').fetchone()[0]
    selector = f'[data-proposal-id="{proposal_id}"]'
    page.goto(browser_app.url + '/#proposals?status=open')
    expect(page.locator('.proposal-archive-btn' + selector)).to_be_disabled()
    expect(page.locator('.proposal-delete-btn' + selector)).to_be_enabled()
    page.locator('.logout').click()
    page.wait_for_url('**#login')
    login('owner')
    page.goto(browser_app.url + '/#proposals?status=open')
    expect(page.locator('.proposal-delete-btn' + selector)).to_be_disabled()
    with page.expect_response('**/api/proposal/post') as archived:
        page.locator('.proposal-archive-btn' + selector).click()
    assert archived.value.status == 200
    page.wait_for_load_state('networkidle')
    page.locator('.logout').click()
    page.wait_for_url('**#login')
    login('collaborator')
    page.goto(browser_app.url + f'/#proposals?status=archive&ids={proposal_id}')
    with page.expect_response('**/api/proposal/delete') as deleted:
        page.locator('.proposal-delete-btn' + selector).click()
    assert deleted.value.status == 200


def test_proposal_search_with_empty_notes_renders_without_error(page, browser_app, login):
    """Unfiltered proposal search renders proposals with empty notes without raising an alert."""
    login()
    dialogs = []
    page.on('dialog', lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
    page.goto(browser_app.url + '/#proposals?status=all')
    page.wait_for_load_state('networkidle')
    assert dialogs == []
    empty_proposal = page.locator('.results-item', has_text='Proposal ID: 10')
    expect(empty_proposal).to_be_visible()
    expect(empty_proposal).to_contain_text('Source Object ID: Empty')
    expect(empty_proposal).to_contain_text('Target Object ID: Empty')


def test_merge_and_embedding_update(page, browser_app, login):
    """Merge writes and the app-owned embedding snapshot flow remain available."""
    login()
    page.on('dialog', lambda dialog: dialog.accept())
    page.goto(browser_app.url + '/#merge?name1=0&name2=2')
    with page.expect_response('**/api/individual/merge') as merged:
        page.locator('.output.save').click()
    assert merged.value.status == 200
    page.wait_for_url('**#individual?**')
    with sqlite3.connect(browser_app.database) as connection:
        sources = connection.execute("SELECT source FROM Individuals WHERE name IN ('0', '2')").fetchall()
    assert len({row[0] for row in sources}) == 1
    page.goto(browser_app.url + '/#')
    warnings = []
    page.on('console', lambda message: message.type == 'warning' and warnings.append(message.text))
    with page.expect_response('**/api/embeddings/update') as updated:
        page.locator('.embeddings-button').click()
    assert updated.value.status == 200
    # Completion polling reloads Home so the refreshed embedding counts appear.
    with page.expect_event('load', timeout=20000):
        pass
    assert not [warning for warning in warnings if 'polling embeddings' in warning], warnings


@pytest.mark.parametrize('operation', ['reference', 'unmatch', 'delete'])
def test_image_actions(page, browser_app, login, operation):
    """Reference, unmatch and deletion keep their real gallery controls."""
    login()
    page.on('dialog', lambda dialog: dialog.accept())
    page.goto(browser_app.url + '/#individual?name=0')
    image = page.locator('.gallery-scroll .gallery-item').first
    media_id = int(image.get_attribute('object-id'))
    image.locator('img').click()
    with sqlite3.connect(browser_app.database) as connection:
        individual_id, source = connection.execute(
            'SELECT id, source FROM Individuals WHERE media = ?', (media_id,)).fetchone()
        filename = connection.execute('SELECT filename FROM Media WHERE id = ?',
                                      (media_id,)).fetchone()[0]
    endpoint = 'media/delete' if operation == 'delete' else 'individual/' + operation
    with page.expect_response('**/api/' + endpoint) as response:
        page.locator('.media.' + operation).click()
    assert response.value.status == 200
    with sqlite3.connect(browser_app.database) as connection:
        if operation == 'reference':
            assert connection.execute('SELECT reference FROM Individuals WHERE id = ?',
                                      (source,)).fetchone() == (individual_id,)
        elif operation == 'unmatch':
            assert connection.execute('SELECT source FROM Individuals WHERE id = ?',
                                      (individual_id,)).fetchone() != (source,)
        else:
            assert connection.execute('SELECT id FROM Media WHERE id = ?', (media_id,)).fetchone() is None
            assert not (browser_app.images / filename).exists()


def _home_value(page, label):
    """Return the value locator next to one exactly named Home overview label."""
    return page.locator('.info-row', has=page.get_by_text(label, exact=True)).locator('.value')


def test_home_counts_include_an_upload_immediately(page, browser_app, login):
    """Home shows the new image as waiting for embeddings straight after the upload, not a cached count."""
    login()
    expect(page.locator('.embeddings-button')).to_be_visible()
    expect(_home_value(page, 'Images without Embeddings:')).to_have_text('0')
    image = (browser_app.images / browser_app.media[0][2]).read_bytes()
    page.goto(browser_app.url + '/#upload')
    page.locator('.upload.file-input').set_input_files(
        {'name': 'AN26111901.jpeg', 'mimeType': 'image/jpeg', 'buffer': image})
    with page.expect_response('**/api/media/upload_batch') as response:
        page.locator('.upload.save').click()
    assert response.value.status == 200
    page.goto(browser_app.url + '/#home')
    expect(_home_value(page, 'Images without Embeddings:')).to_have_text(re.compile(r'^1\b'))


def test_home_reports_an_unreachable_matcher(page, browser_app, login, monkeypatch):
    """With the matcher down, Home says so instead of showing a lone U and zero embeddings."""
    monkeypatch.setattr(app_constants, 'INFERENCE_SERVICE_URL', 'http://127.0.0.1:1')
    frontend.project.invalidate_dataset_stats()
    login()
    expect(page.locator('.matcher-unavailable')).to_be_visible()
    expect(_home_value(page, 'Embeddings Last Updated:')).to_have_text('Unavailable')
    expect(_home_value(page, 'Embeddings:')).to_have_text('Unavailable')
    expect(page.locator('.embeddings-button')).to_be_disabled()


@pytest.mark.parametrize("app_update_age, disabled", [(None, False), (60, True)])
def test_update_button_follows_only_app_triggered_updates(page, browser_app, login,
                                                          app_update_age, disabled):
    """Fresh command-line embeddings leave the button usable; a recent app update disables it."""
    def fresh_embeddings(route):
        body = route.fetch().json()
        body['content']['start_time'] = time.time()
        body['content']['update_cooldown_started_at'] = (
            None if app_update_age is None else time.time() - app_update_age)
        route.fulfill(status=200, content_type='application/json', body=json.dumps(body))

    page.route('**/api/data/stats', fresh_embeddings)
    login('admin')
    button = page.locator('.embeddings-button')
    if disabled:
        expect(button).to_be_disabled()
    else:
        expect(button).to_be_enabled()
