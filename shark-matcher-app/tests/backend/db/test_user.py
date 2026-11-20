"""Tests for user models, persistence, and authentication behavior."""

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import argon2

from core import app_utils
import backend.db.user
import db.database
from backend.db.user import (
    User, Token, upsert_user, get_users, get_matching_users, delete_user,
    upsert_token, get_tokens, delete_token,
    authenticate_user, login_user, logout_user, update_user_password
)
from core.app_constants import ERROR, SUCCESS

class TestUser(unittest.TestCase):
    """Exercise user and session database operations."""

    def setUp(self):
        # Do not clear Users/Tokens tables. Use seeded data where appropriate..
        self.ph = argon2.PasswordHasher()

    def _delete_user_and_sessions(self, user_id):
        """Remove a transient user and every session created for it."""
        for token in get_tokens():
            if token.user == user_id:
                delete_token(token.id)
        delete_user(user_id)

    # --- User Class ---
    def test_upsert_user_create(self):
        """Test Creating a User."""
        u_id = 10001
        try:
            user = User(id=u_id, name="Test User", email="test@test.com", username="testuser", 
                        role="user", password="hashed_pw", salt="salt")
            res = upsert_user(user)
            self.assertEqual(res.name, "Test User")
            
            # Verify it was inserted.
            fetched = get_users(ids=[u_id])[0]
            self.assertEqual(fetched.username, "testuser")
        finally:
            delete_user(u_id)

    def test_upsert_user_update(self):
        """Test Updating a User."""
        u_id = 10003
        try:
            user = User(id=u_id, name="Test User", email="test@test.com", username="testuser_upd", 
                        role="user", password="hashed_pw", salt="salt")
            upsert_user(user)

            user.name = "Test User Updated"
            res_update = upsert_user(user)
            self.assertEqual(res_update.name, "Test User Updated")
        finally:
            delete_user(u_id)

    def test_get_users_single(self):
        """Test Reading a single User."""
        fetched = get_users(ids=[0])
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0].id, 0)

    def test_get_users_batch(self):
        """Test Reading multiple Users."""
        ids = [0, 1]
        fetched = get_users(ids=ids)
        self.assertEqual(len(fetched), 2)
        res_ids_set = {u.id for u in fetched}
        self.assertEqual(res_ids_set, {0, 1})

    def test_get_users_by_username(self):
        """Test get_users by Username."""
        fetched_name = get_users(usernames=["testowner"])
        self.assertEqual(len(fetched_name), 1)
        self.assertEqual(fetched_name[0].id, 0)
        self.assertEqual(fetched_name[0].username, "testowner")

    def test_get_users_non_existent(self):
        """Test get_users with non-existent ID."""
        res = get_users(ids=[-9999])
        self.assertEqual(res, [])

    def test_get_matching_users(self):
        """Test get_matching_users function with seeded data."""
        # Seeded users: testowner (0) owner, test1 (1) and test2 (2) collaborators.
        
        # Match by keyword (e.g., name match).
        # The owner is found by its exact name, "Test Owner".
        res = get_matching_users(keywords=["Test Owner"])
        self.assertGreaterEqual(len(res), 1)
        # Check testowner is in there.
        self.assertIn(0, [u.id for u in res])

        # Match by role "collaborator" (test1).
        res_role = get_matching_users(roles=["collaborator"])
        self.assertGreaterEqual(len(res_role), 1)
        self.assertIn(1, [u.id for u in res_role])

        # Match by role "collaborator" (test1, test2).
        res_admin = get_matching_users(roles=["collaborator"])
        # Multiple collaborators in CSV.
        self.assertGreaterEqual(len(res_admin), 2)
        self.assertIn(2, [u.id for u in res_admin])

        # No match.
        res_none = get_matching_users(keywords=["GhostUser"])
        self.assertEqual(len(res_none), 0)

    def test_get_matching_users_combined(self):
        """Test get_matching_users with combined criteria."""
        # Verify that get_matching_users unions keyword matches with role filter results.
        
        # Case 1: user 0 matches both keyword and role.
        res = get_matching_users(keywords=["Test Owner"], roles=["owner"])
        self.assertIn(0, [u.id for u in res])
        
        # Case 2: union keyword match (user 0) and role matches (collaborators).
        res_mixed = get_matching_users(keywords=["Test Owner"], roles=["collaborator"])
        self.assertIn(0, [u.id for u in res_mixed])
        self.assertIn(2, [u.id for u in res_mixed])
        
        # Case 3: No match on either.
        res_none = get_matching_users(keywords=["GhostUser"], roles=["GhostRole"])
        self.assertEqual(len(res_none), 0)

    def test_delete_user(self):
        """Test delete_user function."""
        u_id = 10002
        try:
            u = User(id=u_id, username="del", name="Del", email="d", role="u", password="p", salt="s")
            upsert_user(u)
            
            status = delete_user(u_id)
            self.assertEqual(status, SUCCESS)
            
            res = get_users(ids=[u_id])
            self.assertEqual(len(res), 0)
        finally:
            delete_user(u_id)

    def test_delete_user_non_existent(self):
        """Test delete_user with non-existent ID."""
        res = delete_user(-9999)
        self.assertEqual(res, ERROR)

    # --- Token Class ---
    def test_upsert_token_create(self):
        """Test Creating a Token."""
        t_id = 1
        u_id = 100
        try:
            t = Token(id=t_id, token="token_str", user=u_id, expires=2_100_000_000)
            res = upsert_token(t)
            self.assertEqual(res.token, "token_str")
            self.assertEqual(res.expires, 2_100_000_000)
            self.assertEqual(
                res.to_dict(),
                {
                    "id": t_id,
                    "token": "token_str",
                    "user": u_id,
                    "expires": 2_100_000_000,
                },
            )
            
            fetched = get_tokens(ids=[t_id])[0]
            self.assertEqual(fetched.token, "token_str")
        finally:
            delete_token(t_id)

    def test_upsert_token_update(self):
        """Test Updating a Token."""
        t_id = 2
        u_id = 101
        try:
            t = Token(id=t_id, token="token_orig", user=u_id, expires=2_100_000_000)
            upsert_token(t)

            t.token = "token_updated"
            res = upsert_token(t)
            self.assertEqual(res.token, "token_updated")
            
            fetched = get_tokens(ids=[t_id])[0]
            self.assertEqual(fetched.token, "token_updated")
        finally:
            delete_token(t_id)

    def test_get_tokens_single(self):
        """Test Reading a single Token."""
        t_id = 2
        try:
            t = Token(id=t_id, token="unique_token", user=101, expires=2_100_000_000)
            upsert_token(t)
            
            # By ID.
            res_id = get_tokens(ids=[t_id])[0]
            self.assertEqual(res_id.token, "unique_token")
        finally:
            delete_token(t_id)

    def test_get_tokens_by_token_string(self):
        """Test get_tokens by Token String."""
        t_id = 2
        try:
            t = Token(id=t_id, token="unique_token", user=101, expires=2_100_000_000)
            upsert_token(t)
            
            # By Token String.
            res_str = get_tokens(tokens=["unique_token"])[0]
            self.assertEqual(res_str.id, t_id)
        finally:
            delete_token(t_id)

    def test_get_tokens_batch(self):
        """Test Reading multiple Tokens."""
        ids = [3, 4]
        try:
            t1 = Token(id=ids[0], token="t1", user=1, expires=2_100_000_000)
            t2 = Token(id=ids[1], token="t2", user=1, expires=2_100_000_000)
            upsert_token(t1)
            upsert_token(t2)
            
            res = get_tokens(ids=ids)
            self.assertEqual(len(res), 2)
        finally:
            delete_token(ids[0])
            delete_token(ids[1])

    def test_get_tokens_non_existent(self):
        """Test get_tokens with non-existent ID."""
        res = get_tokens(ids=[-9999])
        self.assertEqual(res, [])

    def test_delete_token(self):
        """Test delete_token function."""
        t_id = 5
        try:
            t = Token(id=t_id, token="del_tok", user=1, expires=2_100_000_000)
            upsert_token(t)
            
            status = delete_token(t_id)
            self.assertEqual(status, SUCCESS)
            
            res = get_tokens(ids=[t_id])
            self.assertEqual(len(res), 0)
        finally:
            delete_token(t_id)

    def test_delete_token_non_existent(self):
        """Test delete_token with non-existent ID."""
        res = delete_token(-9999)
        self.assertEqual(res, ERROR)

    # --- Authentication Tests ---
    def test_authenticate_user(self):
        """Test authenticate_user function."""
        u_id = 50
        t_id = 500
        try:
            # Setup user and token.
            salt = "salt_for_auth_user_test"
            
            u = User(id=u_id, username="auth_u", password="pw", salt=salt, name="A", email="a", role="u")
            upsert_user(u)
            
            token_plain = "plain_token_string"
            token_hash = backend.db.user._session_token_digest(token_plain)
            
            t = Token(
                id=t_id,
                token=token_hash,
                user=u_id,
                expires=2_100_000_000,
            )
            upsert_token(t)
            
            # Authenticate.
            is_auth = authenticate_user("auth_u", token_plain)
            self.assertTrue(is_auth)
            
            # Fail.
            is_auth_fail = authenticate_user("auth_u", "wrong_token")
            self.assertFalse(is_auth_fail)
        finally:
            delete_token(t_id)
            delete_user(u_id)

    def test_login_user(self):
        """Test login_user function."""
        u_id = 60
        try:
            ph = argon2.PasswordHasher()
            salt = "login_salt_secure_enough"
            pw = "password123"
            hashed = ph.hash(pw, salt=salt.encode("utf-8"))
            
            u = User(id=u_id, username="login_u", password=hashed, salt=salt, name="L", email="l", role="u")
            upsert_user(u)
            
            # Success.
            res = login_user("login_u", "password123")
            self.assertNotEqual(res, ERROR)
            self.assertIn("token", res)
            
            # Failure.
            res_fail = login_user("login_u", "wrong")
            self.assertEqual(res_fail, ERROR)
        finally:
            # Need to cleanup the created token from login.
            self._delete_user_and_sessions(u_id)

    def test_create_user_uses_unique_argon2_salts(self):
        """Shared account creation lets Argon2 generate a unique secure salt."""
        created_users = []
        try:
            first = backend.db.user.create_user(
                name="Salt One",
                email="salt-one@example.test",
                username="salt_one",
                role="viewer",
                password="same-client-derivative",
            )
            second = backend.db.user.create_user(
                name="Salt Two",
                email="salt-two@example.test",
                username="salt_two",
                role="viewer",
                password="same-client-derivative",
            )
            created_users.extend([first.id, second.id])

            self.assertNotEqual(first.password, second.password)
            self.assertTrue(self.ph.verify(first.password, "same-client-derivative"))
            self.assertTrue(self.ph.verify(second.password, "same-client-derivative"))
            self.assertEqual(first.salt, backend.db.user.PASSWORD_SCHEME)
            self.assertEqual(second.salt, backend.db.user.PASSWORD_SCHEME)
            self.assertEqual(
                backend.db.user.create_user(
                    name="Duplicate",
                    email="duplicate@example.test",
                    username="salt_one",
                    role="viewer",
                    password="different-client-derivative",
                ),
                ERROR,
            )
        finally:
            for user_id in created_users:
                delete_user(user_id)

    def test_login_migrates_legacy_password_hash(self):
        """A successful legacy login rewrites the hash through the shared writer."""
        user_id = 61
        password = "legacy-client-derivative"
        salt = "deterministic_legacy_salt"
        legacy_hash = self.ph.hash(password, salt=salt.encode("utf-8"))
        try:
            upsert_user(User(
                id=user_id,
                name="Legacy User",
                email="legacy@example.test",
                username="legacy_login_user",
                role="viewer",
                password=legacy_hash,
                salt=salt,
            ))

            result = login_user("legacy_login_user", password)
            self.assertNotEqual(result, ERROR)

            migrated = get_users(ids=[user_id])[0]
            self.assertNotEqual(migrated.password, legacy_hash)
            self.assertEqual(migrated.salt, backend.db.user.PASSWORD_SCHEME)
            self.assertTrue(self.ph.verify(migrated.password, password))
        finally:
            self._delete_user_and_sessions(user_id)

    def test_login_rejects_malformed_password_hash_safely(self):
        """A malformed stored hash is an ordinary failed login, not an exception."""
        user_id = 62
        try:
            upsert_user(User(
                id=user_id,
                name="Malformed Hash",
                email="malformed@example.test",
                username="malformed_hash_user",
                role="viewer",
                password="not-an-argon2-hash",
                salt="long_enough_legacy_salt",
            ))

            self.assertEqual(login_user("malformed_hash_user", "candidate"), ERROR)
            self.assertEqual(login_user("malformed_hash_user", None), ERROR)
        finally:
            delete_user(user_id)

    def test_provisioning_derivative_matches_browser_sha256_contract(self):
        """Trusted provisioning derives the same lowercase SHA-256 sent by the UI."""
        cleartext = "first-owner-password"
        expected = hashlib.sha256(cleartext.encode("utf-8")).hexdigest()

        self.assertEqual(
            backend.db.user.derive_client_password(cleartext),
            expected,
        )

    def test_first_owner_provisioning_reuses_shared_creation_and_refuses_repeat(self):
        """Trusted provisioning creates one owner and cannot seed populated users."""
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "owner.db"
            schema_path = (
                Path(db.database.__file__).parent
                / "sql"
                / "create_database.sql"
            )
            with sqlite3.connect(database_path) as connection:
                connection.executescript(schema_path.read_text(encoding="utf-8"))

            cleartext_password = "first-owner-password"
            args = db.database.argparse.Namespace(
                name=None,
                display_name="First Owner",
                email="first-owner@example.test",
                username="first_owner",
            )
            with (
                patch.object(
                    db.database,
                    "_configured_database_path",
                    return_value=str(database_path),
                ),
                patch.object(
                    db.database.getpass,
                    "getpass",
                    side_effect=[cleartext_password, cleartext_password],
                ) as password_prompt,
                patch("builtins.print") as print_mock,
            ):
                db.database.handle_provision_owner(args)

            password_prompt.assert_has_calls([
                call("Owner password: "),
                call("Confirm owner password: "),
            ])
            printed_output = " ".join(
                str(call.args[0]) for call in print_mock.call_args_list if call.args
            )
            self.assertNotIn(cleartext_password, printed_output)

            with sqlite3.connect(database_path) as connection:
                owners = backend.db.user.get_users(connection=connection)
            self.assertEqual(len(owners), 1)
            self.assertEqual(owners[0].username, "first_owner")
            self.assertEqual(owners[0].role, "owner")
            self.assertEqual(owners[0].salt, backend.db.user.PASSWORD_SCHEME)
            self.assertTrue(self.ph.verify(
                owners[0].password,
                backend.db.user.derive_client_password(cleartext_password),
            ))

            with self.assertRaises(ValueError):
                db.database.provision_first_owner(
                    db_path=str(database_path),
                    display_name="Second Owner",
                    email="second-owner@example.test",
                    username="second_owner",
                    password="another-password",
                )

            with (
                patch.object(db.database, "_configured_database_path",
                             return_value=str(database_path)),
                patch.object(db.database.getpass, "getpass") as second_prompt,
                patch("builtins.print"),
                self.assertRaises(SystemExit) as raised,
            ):
                db.database.handle_provision_owner(args)
            self.assertEqual(raised.exception.code, 1)
            second_prompt.assert_not_called()

    def test_login_uses_cryptographically_secure_token_source(self):
        """Session issuance obtains the browser token from secrets, not random."""
        user_id = 63
        password = "secure-token-password"
        salt = "secure_token_legacy_salt"
        try:
            upsert_user(User(
                id=user_id,
                name="Secure Token",
                email="secure-token@example.test",
                username="secure_token_user",
                role="viewer",
                password=self.ph.hash(password, salt=salt.encode("utf-8")),
                salt=salt,
            ))

            with patch.object(
                backend.db.user.secrets,
                "token_urlsafe",
                return_value="opaque-secure-session-token",
            ) as token_urlsafe:
                result = login_user("secure_token_user", password)

            token_urlsafe.assert_called_once()
            self.assertGreaterEqual(token_urlsafe.call_args.args[0], 32)
            self.assertEqual(result["token"], "opaque-secure-session-token")
        finally:
            self._delete_user_and_sessions(user_id)

    def test_login_persists_session_expiry(self):
        """A newly issued session stores its expiry alongside the token digest."""
        user_id = 64
        issued_at = 2_000_000_000
        password = "expiry-password"
        salt = "expiry_legacy_salt"
        try:
            upsert_user(User(
                id=user_id,
                name="Expiring Session",
                email="expiry@example.test",
                username="expiry_user",
                role="viewer",
                password=self.ph.hash(password, salt=salt.encode("utf-8")),
                salt=salt,
            ))

            with patch.object(backend.db.user.time, "time", return_value=issued_at):
                result = login_user("expiry_user", password)

            self.assertNotEqual(result, ERROR)
            self.assertEqual(
                backend.db.user.SESSION_TTL_SECONDS,
                7 * 24 * 60 * 60,
            )
            records = [token for token in get_tokens() if token.user == user_id]
            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0].expires,
                issued_at + backend.db.user.SESSION_TTL_SECONDS,
            )
            self.assertNotEqual(records[0].token, result["token"])
            self.assertEqual(
                records[0].token,
                backend.db.user._session_token_digest(result["token"]),
            )

            with patch.object(
                backend.db.user.time,
                "time",
                return_value=issued_at + 1,
            ):
                self.assertTrue(authenticate_user("expiry_user", result["token"]))
        finally:
            self._delete_user_and_sessions(user_id)

    def test_authenticate_rejects_and_removes_expired_session(self):
        """Expired or malformed session rows are rejected and cleaned up."""
        user_id = 65
        token_id = 650
        token = "expired-browser-token"
        try:
            upsert_user(User(
                id=user_id,
                name="Expired Session",
                email="expired@example.test",
                username="expired_session_user",
                role="viewer",
                password="unused",
                salt=backend.db.user.PASSWORD_SCHEME,
            ))
            upsert_token(Token(
                id=token_id,
                token=backend.db.user._session_token_digest(token),
                user=user_id,
                expires=99,
            ))

            with patch.object(backend.db.user.time, "time", return_value=100):
                self.assertFalse(authenticate_user("expired_session_user", token))

            self.assertEqual(get_tokens(ids=[token_id]), [])

            malformed_token_id = token_id + 1
            malformed_token = "malformed-expiry-token"
            upsert_token(Token(
                id=malformed_token_id,
                token=backend.db.user._session_token_digest(malformed_token),
                user=user_id,
                expires="invalid",
            ))
            self.assertFalse(
                authenticate_user("expired_session_user", malformed_token)
            )
            self.assertEqual(get_tokens(ids=[malformed_token_id]), [])
        finally:
            delete_token(token_id)
            delete_token(token_id + 1)
            delete_user(user_id)

    def test_logout_user(self):
        """Test logout_user function."""
        u_id = 70
        t_id = 700
        try:
            # Create user and active token.
            salt = "logout_salt_xyz"
            u = User(id=u_id, username="logout_u", salt=salt, password="p", name="LO", email="e", role="u")
            upsert_user(u)
            
            token_plain = "logout_token"
            token_hash = backend.db.user._session_token_digest(token_plain)
            t = Token(
                id=t_id,
                token=token_hash,
                user=u_id,
                expires=2_100_000_000,
            )
            upsert_token(t)

            # A valid token presented with another username cannot revoke its
            # owner's session.
            self.assertEqual(logout_user("testowner", token_plain), SUCCESS)
            self.assertEqual(len(get_tokens(ids=[t_id])), 1)
            
            # Logout.
            status = logout_user("logout_u", token_plain)
            self.assertEqual(status, SUCCESS)
            
            # Check token is gone.
            res = get_tokens(ids=[t_id])
            self.assertEqual(len(res), 0)
        finally:
            delete_token(t_id)
            delete_user(u_id)

    def test_update_user_password(self):
        """Test update_user_password function."""
        u_id = 80
        try:
            ph = argon2.PasswordHasher()
            salt = "update_pw_salt"
            old_pw = "old"
            old_hash = ph.hash(old_pw, salt=salt.encode("utf-8"))
            
            u = User(id=u_id, username="upd_pw_u", password=old_hash, salt=salt, name="U", email="e", role="u")
            upsert_user(u)
            
            # Update.
            new_pw = "new_secret"
            res = update_user_password("upd_pw_u", new_pw)
            self.assertEqual(res, SUCCESS)
            
            # Verify login with new.
            login_res = login_user("upd_pw_u", new_pw)
            self.assertNotEqual(login_res, ERROR)
            updated_user = get_users(ids=[u_id])[0]
            self.assertEqual(updated_user.salt, backend.db.user.PASSWORD_SCHEME)
            self.assertTrue(self.ph.verify(updated_user.password, new_pw))
            self.assertEqual(update_user_password("upd_pw_u", None), ERROR)
        finally:
            self._delete_user_and_sessions(u_id)

    def test_production_credential_debug_paths_are_retired(self):
        """Production modules expose neither default users nor secret printers."""
        self.assertFalse(hasattr(backend.db.user, "create_salt"))
        self.assertFalse(hasattr(backend.db.user, "_show_user_credentials"))
        self.assertFalse(hasattr(backend.db.user, "calculate_hash"))
        self.assertFalse(hasattr(app_utils, "print_password"))
        self.assertFalse(hasattr(backend.db.user.User, "print"))
        self.assertFalse(hasattr(backend.db.user.Token, "print"))
        database_source = Path(db.database.__file__).read_text(encoding="utf-8")
        self.assertNotIn("default_users.csv", database_source)

    def test_database_print_reports_auth_counts_without_sensitive_rows(self):
        """Local database diagnostics do not print user or session records."""
        with patch.object(
            db.database, "run_sqlite_cmd"
        ) as run_sqlite_cmd, patch.object(
            db.database, "print_indexes"
        ), patch.object(
            db.database,
            "get_image_sync_status",
            return_value=([], []),
        ):
            db.database.handle_print(None)

        commands = [call_args.args[1] for call_args in run_sqlite_cmd.call_args_list]
        user_command = next(
            command for command in commands if "COUNT(*) from Users" in command
        )
        token_command = next(
            command for command in commands if "COUNT(*) from Tokens" in command
        )
        self.assertNotIn("SELECT *", user_command)
        self.assertNotIn("SELECT *", token_command)


def test_authentication_schema_migration_preserves_users_and_drops_sessions():
    """The idempotent migration preserves accounts and invalidates old sessions."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "legacy.db"
        with sqlite3.connect(database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE Users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    username TEXT NOT NULL UNIQUE,
                    role TEXT NOT NULL,
                    password TEXT NOT NULL,
                    salt TEXT NOT NULL
                );
                CREATE TABLE Tokens (
                    id INTEGER PRIMARY KEY,
                    token TEXT NOT NULL,
                    user INTEGER REFERENCES Users(id) ON DELETE CASCADE NOT NULL
                );
                INSERT INTO Users VALUES (
                    1, 'Owner', 'owner@example.test', 'owner', 'owner', 'hash', 'salt'
                );
                INSERT INTO Tokens VALUES (1, 'legacy-token-hash', 1);
                """
            )

        db.database.migrate_authentication_schema(database_path)
        db.database.migrate_authentication_schema(database_path)

        with sqlite3.connect(database_path) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(Tokens)")
            }
            users = connection.execute("SELECT username FROM Users").fetchall()
            tokens = connection.execute("SELECT * FROM Tokens").fetchall()

        assert "expires" in columns
        assert users == [("owner",)]
        assert tokens == []
