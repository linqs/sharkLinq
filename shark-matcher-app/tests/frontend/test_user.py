"""Tests for user-facing HTTP authentication and account policy."""

import hashlib
import http
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import argon2

import backend.db.user
import frontend.user

APP_ROOT = Path(__file__).resolve().parents[2]


class MockHandler:
    """Minimal request handler state used by direct frontend function tests."""

    pass


class TestAuthenticationAttemptLimiter(unittest.TestCase):
    """Exercise the shared process-local authentication limiter."""

    def test_bucket_keys_use_operation_direct_peer_and_normalized_username(self):
        """Buckets isolate operations, direct peers, and normalized usernames."""
        limiter = frontend.user._AuthenticationAttemptLimiter(1, 60)
        first_handler = MockHandler()
        first_handler.client_address = ("192.0.2.20", 8000)
        first_handler.headers = {"X-Forwarded-For": "198.51.100.1"}
        second_handler = MockHandler()
        second_handler.client_address = ("192.0.2.21", 8001)
        second_handler.headers = {"X-Forwarded-For": "198.51.100.1"}

        result, retry_after = limiter.verify_password(
            first_handler,
            " Limited_User ",
            lambda: backend.db.user.app_constants.ERROR,
        )
        self.assertEqual(result, backend.db.user.app_constants.ERROR)
        self.assertIsNone(retry_after)

        verifier_called = False

        def unexpected_verifier():
            nonlocal verifier_called
            verifier_called = True
            return False

        _, retry_after = limiter.verify_password(
            first_handler, "limited_user", unexpected_verifier
        )
        self.assertIsNotNone(retry_after)
        self.assertFalse(verifier_called)

        _, retry_after = limiter.verify_token(
            first_handler, "limited_user", lambda: False
        )
        self.assertIsNone(retry_after)
        _, retry_after = limiter.verify_password(
            second_handler,
            "limited_user",
            lambda: backend.db.user.app_constants.ERROR,
        )
        self.assertIsNone(retry_after)
        _, retry_after = limiter.verify_password(
            first_handler,
            "another_user",
            lambda: backend.db.user.app_constants.ERROR,
        )
        self.assertIsNone(retry_after)

    def test_full_token_bucket_admits_valid_tokens_but_still_refuses_bad_ones(self):
        """Bad tokens from one browser tab cannot lock out a valid session on the same computer."""
        limiter = frontend.user._AuthenticationAttemptLimiter(2, 60)
        peer = MockHandler()
        peer.client_address = ("192.0.2.40", 8000)
        for _ in range(2):
            limiter.verify_token(peer, "", lambda: False)

        result, retry_after = limiter.verify_token(peer, "", lambda: True)
        self.assertIs(result, True)
        self.assertIsNone(retry_after)

        result, retry_after = limiter.verify_token(peer, "", lambda: False)
        self.assertIsNone(result)
        self.assertIsNotNone(retry_after)

    def test_full_password_bucket_still_refuses_without_checking(self):
        """Password guesses stay blocked outright once the bucket is full."""
        limiter = frontend.user._AuthenticationAttemptLimiter(1, 60)
        peer = MockHandler()
        peer.client_address = ("192.0.2.41", 8000)
        limiter.verify_password(peer, "someone", lambda: backend.db.user.app_constants.ERROR)
        checked = []
        _, retry_after = limiter.verify_password(peer, "someone", lambda: checked.append(1) or True)
        self.assertIsNotNone(retry_after)
        self.assertEqual(checked, [])

    def test_forgiving_token_failures_reopens_only_that_peers_token_buckets(self):
        """A fresh login clears the peer's token lockout without touching other peers or password buckets."""
        limiter = frontend.user._AuthenticationAttemptLimiter(1, 60)
        peer = MockHandler()
        peer.client_address = ("192.0.2.30", 8000)
        other = MockHandler()
        other.client_address = ("192.0.2.31", 8001)
        for handler in (peer, other):
            limiter.verify_token(handler, "", lambda: None)
            limiter.verify_password(handler, "user", lambda: backend.db.user.app_constants.ERROR)

        limiter.forgive_token_failures(peer)

        self.assertIsNone(limiter.verify_token(peer, "", lambda: False)[1])
        self.assertIsNotNone(limiter.verify_token(other, "", lambda: False)[1])
        self.assertIsNotNone(limiter.verify_password(peer, "user", lambda: True)[1])

    def test_failures_expire_from_rolling_window(self):
        """A bucket reopens when its recorded failures leave the window."""
        now = [100.0]
        limiter = frontend.user._AuthenticationAttemptLimiter(
            1, 10, clock=lambda: now[0]
        )
        handler = MockHandler()
        handler.client_address = ("192.0.2.22", 8000)

        _, retry_after = limiter.verify_password(
            handler,
            "limited_user",
            lambda: backend.db.user.app_constants.ERROR,
        )
        self.assertIsNone(retry_after)
        _, retry_after = limiter.verify_password(
            handler,
            "limited_user",
            lambda: backend.db.user.app_constants.ERROR,
        )
        self.assertEqual(retry_after, 10)

        now[0] = 109.25
        _, retry_after = limiter.verify_password(
            handler,
            "limited_user",
            lambda: backend.db.user.app_constants.ERROR,
        )
        self.assertEqual(retry_after, 1)

        now[0] = 110.0
        _, retry_after = limiter.verify_password(
            handler, "limited_user", lambda: {"token": "opaque-token"}
        )
        self.assertIsNone(retry_after)

    def test_pending_verification_reserves_bucket_capacity(self):
        """Concurrent requests cannot all pass the shared attempt gate."""
        limiter = frontend.user._AuthenticationAttemptLimiter(1, 60)
        handler = MockHandler()
        handler.client_address = ("192.0.2.23", 8000)
        verification_started = threading.Event()
        release_verification = threading.Event()

        def slow_verifier():
            verification_started.set()
            release_verification.wait(timeout=2)
            return backend.db.user.app_constants.ERROR

        thread = threading.Thread(
            target=limiter.verify_password,
            args=(handler, "limited_user", slow_verifier),
        )
        thread.start()
        self.assertTrue(verification_started.wait(timeout=2))

        verifier_called = False

        def unexpected_verifier():
            nonlocal verifier_called
            verifier_called = True
            return False

        _, retry_after = limiter.verify_password(
            handler, "limited_user", unexpected_verifier
        )
        release_verification.set()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(retry_after)
        self.assertFalse(verifier_called)

    def test_concurrent_token_verifications_share_the_bucket(self):
        """A browser's parallel session checks do not reject each other while one is in flight."""
        limiter = frontend.user._AuthenticationAttemptLimiter(1, 60)
        handler = MockHandler()
        handler.client_address = ("192.0.2.26", 8000)
        verification_started = threading.Event()
        release_verification = threading.Event()

        def slow_verifier():
            verification_started.set()
            release_verification.wait(timeout=2)
            return True

        thread = threading.Thread(
            target=limiter.verify_token,
            args=(handler, "session_user", slow_verifier),
        )
        thread.start()
        self.assertTrue(verification_started.wait(timeout=2))

        result, retry_after = limiter.verify_token(
            handler, "session_user", lambda: True
        )
        release_verification.set()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertIs(result, True)
        self.assertIsNone(retry_after)

    def test_token_failures_still_close_the_bucket(self):
        """Recorded token failures make further invalid tokens a 429 within the window."""
        limiter = frontend.user._AuthenticationAttemptLimiter(1, 60)
        handler = MockHandler()
        handler.client_address = ("192.0.2.27", 8000)

        _, retry_after = limiter.verify_token(handler, "session_user", lambda: False)
        self.assertIsNone(retry_after)

        result, retry_after = limiter.verify_token(handler, "session_user", lambda: False)
        self.assertIsNone(result)
        self.assertIsNotNone(retry_after)

    def test_success_clears_only_matching_failure_bucket(self):
        """Successful verification does not reset another user's failures."""
        limiter = frontend.user._AuthenticationAttemptLimiter(2, 60)
        handler = MockHandler()
        handler.client_address = ("192.0.2.24", 8000)

        def failure():
            return backend.db.user.app_constants.ERROR

        def success():
            return {"token": "opaque-token"}

        limiter.verify_password(handler, "first_user", failure)
        limiter.verify_password(handler, "second_user", failure)
        limiter.verify_password(handler, "first_user", success)

        for _ in range(2):
            _, retry_after = limiter.verify_password(
                handler, "first_user", failure
            )
            self.assertIsNone(retry_after)
        _, retry_after = limiter.verify_password(handler, "first_user", failure)
        self.assertIsNotNone(retry_after)

        _, retry_after = limiter.verify_password(handler, "second_user", failure)
        self.assertIsNone(retry_after)
        _, retry_after = limiter.verify_password(handler, "second_user", failure)
        self.assertIsNotNone(retry_after)

    def test_verifier_exception_releases_pending_capacity(self):
        """An unexpected verifier error does not leave a bucket reserved."""
        limiter = frontend.user._AuthenticationAttemptLimiter(1, 60)
        handler = MockHandler()
        handler.client_address = ("192.0.2.25", 8000)

        def failing_verifier():
            raise RuntimeError("verification unavailable")

        with self.assertRaises(RuntimeError):
            limiter.verify_password(
                handler, "limited_user", failing_verifier
            )

        result, retry_after = limiter.verify_password(
            handler, "limited_user", lambda: {"token": "opaque-token"}
        )
        self.assertEqual(result, {"token": "opaque-token"})
        self.assertIsNone(retry_after)


class TestBrowserUserPasswordFlow(unittest.TestCase):
    """Pin the shared browser password and user-request source boundaries."""

    def test_settings_flows_share_password_derivation(self):
        """Profile updates and account creation use one password helper."""
        settings_source = (
            APP_ROOT / "frontend/static/js/modules/webui/settings.js"
        ).read_text(encoding="utf-8")

        self.assertEqual(settings_source.count("deriveSettingsPassword("), 3)
        self.assertEqual(settings_source.count("Util.sha256(password)"), 1)
        self.assertIn("async function executeUserSave()", settings_source)
        self.assertIn("async function executeCreateUser()", settings_source)
        self.assertNotIn("currentUser.username", settings_source)
        self.assertNotIn("Cannot update Test User credentials.", settings_source)
        self.assertIn('class="create-user password-input"', settings_source)
        self.assertIn('class="create-user verify-input"', settings_source)

    def test_browser_hashing_flows_share_one_utility_authority(self):
        """Password and image hashes delegate to the one utility function."""
        util_source = (
            APP_ROOT / "frontend/static/js/modules/webui/util.js"
        ).read_text(encoding="utf-8")
        login_source = (
            APP_ROOT / "frontend/static/js/modules/webui/login.js"
        ).read_text(encoding="utf-8")
        settings_source = (
            APP_ROOT / "frontend/static/js/modules/webui/settings.js"
        ).read_text(encoding="utf-8")
        upload_source = (
            APP_ROOT / "frontend/static/js/modules/webui/upload.js"
        ).read_text(encoding="utf-8")

        self.assertEqual(util_source.count("function sha256(text)"), 1)
        self.assertEqual(util_source.count("sha256,"), 1)
        self.assertEqual(login_source.count("Util.sha256(cleartext)"), 1)
        self.assertEqual(settings_source.count("Util.sha256(password)"), 1)
        self.assertEqual(upload_source.count("Util.sha256("), 1)

    def test_user_request_wrapper_does_not_accept_or_send_salt(self):
        """Browser user writes send only the server-owned password derivative."""
        user_source = (
            APP_ROOT / "frontend/static/js/modules/shark_matcher/user.js"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "function post_user(id, name, email, username, role, password)",
            user_source,
        )
        self.assertNotIn("'salt'", user_source)

    def test_logout_reuses_core_credential_injection(self):
        """Logout does not maintain a second browser credential payload."""
        user_source = (
            APP_ROOT / "frontend/static/js/modules/shark_matcher/user.js"
        ).read_text(encoding="utf-8")
        login_source = (
            APP_ROOT / "frontend/static/js/modules/webui/login.js"
        ).read_text(encoding="utf-8")

        logout_source = user_source.split("function logout_user()", 1)[1]
        logout_source = logout_source.split("/* eslint-disable camelcase */", 1)[0]
        self.assertIn("endpoint: 'user/logout'", logout_source)
        self.assertNotIn("payload:", logout_source)
        self.assertIn("SharkMatcher.User.logout_user()", login_source)

    def test_dead_action_credential_wrapper_is_retired(self):
        """Action requests do not retain the copied, unserved user-write path."""
        action_source = (
            APP_ROOT / "frontend/static/js/modules/shark_matcher/action.js"
        ).read_text(encoding="utf-8")

        self.assertNotIn("function post_action", action_source)
        self.assertNotIn("endpoint: 'action/post'", action_source)
        self.assertNotIn("'password'", action_source)
        self.assertNotIn("'salt'", action_source)

    def test_browser_log_context_uses_shared_credential_redaction(self):
        """Every browser log call sanitizes nested context at one boundary."""
        log_source = (
            APP_ROOT / "frontend/static/js/modules/webui/log.js"
        ).read_text(encoding="utf-8")

        self.assertIn("function sanitizeLogValue(value", log_source)
        self.assertIn("SENSITIVE_LOG_KEY_PATTERN", log_source)
        self.assertIn("'context': sanitizeLogValue(context)", log_source)


class TestUserFrontend(unittest.TestCase):
    """Exercise frontend user handlers and their backend integration."""

    def setUp(self):
        self.handler = MockHandler()
        self.handler.client_address = ("192.0.2.10", 8080)
        self.ph = argon2.PasswordHasher()
        limiter = getattr(frontend.user, "_AUTHENTICATION_LIMITER", None)
        if limiter is not None:
            limiter.clear()
        # Do not clear DB. Seeded data is persistent..
        
        # Create user (transient).
        self.username = "user_tester"
        u = backend.db.user.User(id=889, username=self.username, name="User Tester", email="user@test.com", role="admin", password="pass", salt="salt")
        backend.db.user.upsert_user(u)
        self.created_users = []

    def tearDown(self):
        limiter = getattr(frontend.user, "_AUTHENTICATION_LIMITER", None)
        if limiter is not None:
            limiter.clear()

        # Cleanup transient user.
        try:
            backend.db.user.delete_user(889)
        except:
            pass
        
        for u_id in self.created_users:
            try:
                for token in backend.db.user.get_tokens():
                    if token.user == u_id:
                        backend.db.user.delete_token(token.id)
                backend.db.user.delete_user(u_id)
            except:
                pass

    def authenticate_as(self, username):
        """Simulate what authenticate_user does for a real request: put the verified User on the handler so the target function can read it."""
        users = backend.db.user.get_users(usernames=[username])
        self.handler.authenticated_user = users[0] if users else None
        return self.handler.authenticated_user

    def create_test_user(self, username, password, role="viewer"):
        """Create and track one legacy-format user for handler tests."""
        salt = "salty_salt"
        hashed = self.ph.hash(password, salt=salt.encode("utf-8"))
        # Let DB assign ID (autoincrement) to avoid collision with seeded users.
        u = backend.db.user.User(id=None, name=username, username=username, 
                                 password=hashed, salt=salt, role=role, email=f"{username}@test.com")
        res = backend.db.user.upsert_user(u)
        if res and res.id:
            self.created_users.append(res.id)
        return res

    def test_get_user(self):
        """Test getting a user with seeded data (success)."""
        # Use seeded user 'testowner' (ID 0).
        u_id = 0
        username = "testowner"
        
        # Test getting self (allowed).
        resp, code, _ = frontend.user.get_user(self.handler, "", id=u_id, username=username, user_username=username, user_id=u_id)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertEqual(resp["username"], username)

    def test_get_user_unauthorized(self):
        """Test getting a user unauthorized (fetching another user)."""
        # Seeded user 'testowner' (requestor) trying to fetch 'test1' (target).
        requestor = "testowner"
        target_username = "test1"
        
        resp, code, _ = frontend.user.get_user(self.handler, "", id="", username=target_username, user_username=requestor, user_id=1)
        self.assertEqual(code, http.HTTPStatus.FORBIDDEN)

    def test_filter_users(self):
        """Test filtering users with seeded data, as an admin."""
        # Seeded: the owner Test Owner and two collaborators exist.
        self.authenticate_as(self.username)

        # Filter by role "collaborator".
        users, code, _ = frontend.user.filter_users(
            self.handler, "", ids=None, roles=["collaborator"], keywords=None)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertGreaterEqual(len(users), 1)
        # Check specific collaborator ID 2 (test2) is present.
        ids = [u["id"] for u in users]
        self.assertIn(2, ids)

        # Filter by keyword "Test Owner".
        users, code, _ = frontend.user.filter_users(
            self.handler, "", ids=None, roles=None, keywords=["Test Owner"])
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertGreaterEqual(len(users), 1)
        self.assertEqual(users[0]["username"], "testowner")

    def test_filter_users_never_returns_email(self):
        """No caller of user/filter renders an email, so none is ever sent."""
        self.authenticate_as(self.username)

        users, code, _ = frontend.user.filter_users(
            self.handler, "", ids=None, roles=None, keywords=["Test Owner"])
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertGreaterEqual(len(users), 1)
        for user in users:
            self.assertNotIn("email", user)

    def test_filter_users_viewer_gets_no_identity_fields(self):
        """Viewers get id and name for author attribution, and nothing more."""
        viewer = self.create_test_user("filter_viewer", "pass", role="viewer")
        self.authenticate_as(viewer.username)

        users, code, _ = frontend.user.filter_users(
            self.handler, "", ids=None, roles=None, keywords=["Test Owner"])
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertGreaterEqual(len(users), 1)
        for user in users:
            self.assertEqual(set(user.keys()), {"id", "name"})

    def test_filter_users_unknown_requester_is_not_admin(self):
        """An unresolvable requester is treated as least-privileged, not most."""
        self.handler.authenticated_user = None

        users, code, _ = frontend.user.filter_users(
            self.handler, "", ids=None, roles=None, keywords=["Test Owner"])
        self.assertEqual(code, http.HTTPStatus.OK)
        for user in users:
            self.assertEqual(set(user.keys()), {"id", "name"})

    def test_post_user_update_self(self):
        """Test updating self."""
        u = self.create_test_user("updater", "pass")
        
        kwargs = {
            "id": u.id,
            "name": "New Name",
            "email": "new@test.com",
            "username": "updater",
            "password": None,
            "user_username": "updater",
            "role": "viewer"
        }
        
        self.authenticate_as("updater")
        resp, code, _ = frontend.user.post_user(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertEqual(resp["name"], "New Name")
        
        # Verify in DB.
        u_db = backend.db.user.get_users(ids=[u.id])[0]
        self.assertEqual(u_db.name, "New Name")

    def test_post_user_rejects_injection_name(self):
        """A display name is rendered as author attribution, so it is validated."""
        u = self.create_test_user("namechecker", "pass")

        kwargs = {
            "id": u.id,
            "name": "<img src=x onerror=alert(1)>",
            "email": "namechecker@test.com",
            "username": "namechecker",
            "password": None,
            "user_username": "namechecker",
            "role": "viewer"
        }

        self.authenticate_as("namechecker")
        _, code, _ = frontend.user.post_user(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.BAD_REQUEST)

        # The rejected write must not have changed the stored name.
        unchanged = backend.db.user.get_users(ids=[u.id])[0]
        self.assertEqual(unchanged.name, "namechecker")

    def test_post_user_update_other_unauthorized(self):
        """Test updating another user fails."""
        u1 = self.create_test_user("victim", "pass")
        u2 = self.create_test_user("attacker", "pass")
        
        kwargs = {
            "id": u1.id,
            "name": "Hacked Name",
            "email": "hacked@test.com",
            "username": "victim", # target.
            "password": None,
            "user_username": "attacker", # requestor.
            "role": "viewer"
        }
        
        resp, code, _ = frontend.user.post_user(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.FORBIDDEN)

    def test_post_user_create_unauthorized(self):
        """Test creating a new user fails (as post_user only supports self-update)."""
        # Create the requestor user (viewer) in the DB.
        self.create_test_user("viewer", "pass", role="viewer")

        kwargs = {
            "id": None,
            "name": "New User",
            "email": "new@user.com",
            "username": "newuser",
            "password": "pass",
            "user_username": "viewer", # Requestor.
            "role": "viewer"
        }
        
        # Authenticate as the viewer so this exercises the role check, not a
        # missing-requester short circuit.
        self.authenticate_as("viewer")
        resp, code, _ = frontend.user.post_user(self.handler, "", **kwargs)
        # Should be authorized only if username == user_username.
        self.assertEqual(code, http.HTTPStatus.FORBIDDEN)

    def test_post_user_create_success(self):
        """Test creating a new user (requires Admin)."""
        # User 'testowner' (ID 0) is Owner.
        kwargs = {
            "id": None,
            "name": "Created User",
            "email": "created@test.com",
            "username": "created_user",
            "password": "password",
            "user_username": "testowner", # Requestor (Admin).
            "role": "viewer"
        }
        
        self.authenticate_as("testowner")
        resp, code, _ = frontend.user.post_user(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertEqual(resp["username"], "created_user")
        
        # Verify in DB.
        users = backend.db.user.get_users(usernames=["created_user"])
        self.assertEqual(len(users), 1)

    def test_post_user_creation_requires_password(self):
        """An administrator cannot create an account with an implicit password."""
        kwargs = {
            "id": None,
            "name": "No Password User",
            "email": "no-password@example.test",
            "username": "no_password_user",
            "password": None,
            "user_username": self.username,
            "role": "viewer",
        }

        self.authenticate_as(self.username)
        _, code, _ = frontend.user.post_user(self.handler, "", **kwargs)
        users = backend.db.user.get_users(usernames=["no_password_user"])
        self.created_users.extend(user.id for user in users)

        self.assertEqual(code, http.HTTPStatus.BAD_REQUEST)
        self.assertEqual(users, [])

    def test_post_user_creation_uses_selected_password(self):
        """Account creation persists the supplied browser password derivative."""
        selected_derivative = hashlib.sha256(
            b"chosen-account-password"
        ).hexdigest()
        legacy_default_derivative = hashlib.sha256(b"shark").hexdigest()
        kwargs = {
            "id": None,
            "name": "Selected Password User",
            "email": "selected-password@example.test",
            "username": "selected_password_user",
            "password": selected_derivative,
            "user_username": self.username,
            "role": "viewer",
        }

        self.authenticate_as(self.username)
        response, code, _ = frontend.user.post_user(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.created_users.append(response["id"])

        selected_login = backend.db.user.login_user(
            "selected_password_user", selected_derivative
        )
        default_login = backend.db.user.login_user(
            "selected_password_user", legacy_default_derivative
        )
        self.assertNotEqual(selected_login, backend.db.user.app_constants.ERROR)
        self.assertEqual(default_login, backend.db.user.app_constants.ERROR)

    def test_post_user_create_existing_username(self):
        """Test creating a user with existing username."""
        # Use 'testowner' (seeded owner) as creator.
        # Try to create 'test1' (existing collaborator).
        
        kwargs = {
            "id": None,
            "name": "Duplicate",
            "email": "dup@test.com",
            "username": "test1", # Exists.
            "password": "pass",
            "user_username": "testowner", # Owner.
            "role": "viewer"
        }
        
        self.authenticate_as("testowner")
        resp, code, _ = frontend.user.post_user(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.BAD_REQUEST)
        self.assertIn("exists", resp)

    def test_delete_user(self):
        """Test delete user (Owner only)."""
        # Create a transient user to delete.
        u = self.create_test_user("todelete", "pass", role="viewer")
        
        # Requestor: 'testowner' (seeded Owner).
        self.authenticate_as("testowner")
        resp, code, _ = frontend.user.delete_user(self.handler, "", id=u.id, user_username="testowner")
        self.assertEqual(code, http.HTTPStatus.OK)
        
        # Verify gone.
        gone = backend.db.user.get_users(ids=[u.id])
        self.assertEqual(len(gone), 0)

    def test_delete_user_unauthorized_role(self):
        """Test delete user requires Owner."""
        u = self.create_test_user("nodelete", "pass")
        
        # Requestor: 'test1' (seeded collaborator) - collaborator < Owner (4).
        self.authenticate_as("test1")
        resp, code, _ = frontend.user.delete_user(self.handler, "", id=u.id, user_username="test1")
        self.assertEqual(code, http.HTTPStatus.FORBIDDEN)
        
        # Requestor: 'test2' (seeded collaborator).
        self.authenticate_as("test2")
        resp, code, _ = frontend.user.delete_user(self.handler, "", id=u.id, user_username="test2")
        self.assertEqual(code, http.HTTPStatus.FORBIDDEN)

    def test_authenticate_user(self):
        """Test authentication logic."""
        u = self.create_test_user("auther", "pass", role="admin")
        login_resp, _, _ = frontend.user.login_user(self.handler, "", username="auther", password="pass")
        token = login_resp["token"]
        
        # Valid auth.
        resp, code, _ = frontend.user.authenticate_user(self.handler, "", user_username="auther", user_token=token)
        self.assertEqual(code, http.HTTPStatus.OK)
        
        # Valid auth with role.
        # "collaborator" < "admin" (assumed).
        resp, code, _ = frontend.user.authenticate_user(self.handler, "", min_role_required="collaborator", user_username="auther", user_token=token)
        self.assertEqual(code, http.HTTPStatus.OK)
        
        # Insufficient role.
        self.create_test_user("viewer", "pass", role="viewer")
        login_resp_v, _, _ = frontend.user.login_user(self.handler, "", username="viewer", password="pass")
        token_v = login_resp_v["token"]
        
        resp, code, msg = frontend.user.authenticate_user(self.handler, "", min_role_required="admin", user_username="viewer", user_token=token_v)
        self.assertEqual(code, http.HTTPStatus.FORBIDDEN)

    def test_login_user(self):
        """Test user login."""
        self.create_test_user("loginner", "secret_pass")
        
        # Success.
        resp, code, _ = frontend.user.login_user(self.handler, "", username="loginner", password="secret_pass")
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertIn("token", resp)
        
        # Failure.
        resp, code, _ = frontend.user.login_user(self.handler, "", username="loginner", password="wrong_pass")
        self.assertFalse(resp)

    def test_login_user_rate_limits_before_additional_verification(self):
        """Repeated failed logins are blocked before another password check."""
        attempt_limit = frontend.user.AUTHENTICATION_ATTEMPT_LIMIT
        limiter = frontend.user._AUTHENTICATION_LIMITER
        limiter.clear()

        with patch.object(
            backend.db.user,
            "login_user",
            return_value=backend.db.user.app_constants.ERROR,
        ) as backend_login:
            for _ in range(attempt_limit):
                _, code, _ = frontend.user.login_user(
                    self.handler,
                    "",
                    username="limited_login_user",
                    password="wrong",
                )
                self.assertEqual(code, http.HTTPStatus.OK)

            _, code, headers = frontend.user.login_user(
                self.handler,
                "",
                username="limited_login_user",
                password="wrong",
            )

        self.assertEqual(code, http.HTTPStatus.TOO_MANY_REQUESTS)
        self.assertIn("Retry-After", headers)
        self.assertEqual(backend_login.call_count, attempt_limit)

    def test_authenticate_user_rate_limits_before_additional_verification(self):
        """Repeated invalid sessions share one limiter at the HTTP auth boundary and end in 429."""
        attempt_limit = frontend.user.AUTHENTICATION_ATTEMPT_LIMIT
        limiter = frontend.user._AUTHENTICATION_LIMITER
        limiter.clear()
        self.handler.client_address = ("192.0.2.11", 8080)

        with patch.object(
            backend.db.user,
            "authenticate_user",
            return_value=False,
        ) as backend_authenticate:
            for _ in range(attempt_limit):
                _, code, _ = frontend.user.authenticate_user(
                    self.handler,
                    "",
                    user_username="limited_token_user",
                    user_token="invalid",
                )
                self.assertEqual(code, http.HTTPStatus.UNAUTHORIZED)

            _, code, headers = frontend.user.authenticate_user(
                self.handler,
                "",
                user_username="limited_token_user",
                user_token="invalid",
            )

        self.assertEqual(code, http.HTTPStatus.TOO_MANY_REQUESTS)
        self.assertIn("Retry-After", headers)
        # Tokens are still checked past the limit so a valid session is never refused.
        self.assertEqual(backend_authenticate.call_count, attempt_limit + 1)

    def test_successful_login_clears_matching_failure_bucket(self):
        """A successful login resets only its matching client/user failure state."""
        attempt_limit = frontend.user.AUTHENTICATION_ATTEMPT_LIMIT
        limiter = frontend.user._AUTHENTICATION_LIMITER
        limiter.clear()
        failed = backend.db.user.app_constants.ERROR
        authenticated = {
            "id": 1,
            "name": "Limited User",
            "email": "limited@example.test",
            "username": "limited_user",
            "role": "viewer",
            "token": "opaque-token",
        }
        results = [failed] * (attempt_limit - 1)
        results.append(authenticated)
        results.extend([failed] * attempt_limit)

        with patch.object(
            backend.db.user,
            "login_user",
            side_effect=results,
        ):
            for _ in range(attempt_limit - 1):
                _, code, _ = frontend.user.login_user(
                    self.handler, "", username="limited_user", password="wrong"
                )
                self.assertEqual(code, http.HTTPStatus.OK)

            response, code, _ = frontend.user.login_user(
                self.handler, "", username="limited_user", password="correct"
            )
            self.assertEqual(code, http.HTTPStatus.OK)
            self.assertEqual(response, authenticated)

            for _ in range(attempt_limit):
                _, code, _ = frontend.user.login_user(
                    self.handler, "", username="limited_user", password="wrong"
                )
                self.assertEqual(code, http.HTTPStatus.OK)

            _, code, _ = frontend.user.login_user(
                self.handler, "", username="limited_user", password="wrong"
            )

        self.assertEqual(code, http.HTTPStatus.TOO_MANY_REQUESTS)

    def test_logout_user(self):
        """Test user logout."""
        self.create_test_user("outter", "pass")
        # Login.
        resp, code, _ = frontend.user.login_user(self.handler, "", username="outter", password="pass")
        token = resp["token"]
        
        # Verify authenticated.
        is_auth = backend.db.user.authenticate_user("outter", token)
        self.assertTrue(is_auth)
        
        # Logout.
        resp, code, _ = frontend.user.logout_user(self.handler, "", user_username="outter", user_token=token)
        self.assertEqual(code, http.HTTPStatus.OK)
        
        # Verify not authenticated.
        is_auth = backend.db.user.authenticate_user("outter", token)
        self.assertFalse(is_auth)


if __name__ == '__main__':
    unittest.main()
