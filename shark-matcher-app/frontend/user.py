"""API handlers for user authentication, session management, and account administration."""

from collections import deque
import http
import http.cookies
import math
import threading
import time

from core import app_constants
from core import app_utils
import backend.db.user

AUTHENTICATION_ATTEMPT_LIMIT = 5
AUTHENTICATION_WINDOW_SECONDS = 300
IMAGE_COOKIE = 'shark_image_session'

role_dict = {
    "viewer": 0,
    "collaborator": 1,
    "admin": 2,
    "dev": 3,
    "owner": 4,
}


# pylint: disable=unused-argument, too-many-return-statements
@app_utils.accepts("id", "name", "email", "username", "password", "role")
def post_user(handler, path, **kwargs):
    """Create a new user account or update an existing user's profile."""
    print("Post User function called")

    scalar_params = ["id", "name", "email", "username", "password", "user_username", "role"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    # The display name is rendered as author attribution across the app.
    if not app_utils.valid_name(kwargs["name"]):
        return "Invalid name.", http.HTTPStatus.BAD_REQUEST, None

    # Create or update a user.
    # Check if we are creating a new user (id is None).
    if kwargs["id"] is None:
        # Require admin or higher role to create new users.
        creator_user = get_authenticated_user(handler)
        if creator_user is None:
            return None, http.HTTPStatus.UNAUTHORIZED, None
        if role_dict[creator_user.role] < 2:
            return "Only admins can create users.", http.HTTPStatus.FORBIDDEN, None

        # Check if username exists.
        existing_users = backend.db.user.get_users(usernames=[kwargs["username"]])
        if existing_users:
            return "Username already exists.", http.HTTPStatus.BAD_REQUEST, None

        # Validate role (cannot be dev or owner).
        if kwargs["role"] in ["dev", "owner"]:
            return "Cannot assign dev or owner role.", http.HTTPStatus.FORBIDDEN, None

        if not isinstance(kwargs["password"], str) or not kwargs["password"]:
            return "Password is required.", http.HTTPStatus.BAD_REQUEST, None

        posted_object = backend.db.user.create_user(
            name=kwargs["name"],
            email=kwargs["email"],
            username=kwargs["username"],
            role=kwargs["role"],
            password=kwargs["password"],
        )
        if posted_object == app_constants.ERROR:
            return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None
        return posted_object.to_secure_dict(), http.HTTPStatus.OK, None

    # Disallow editing profiles other than the user's own.
    if kwargs["username"] != kwargs["user_username"]:
        return "You can only edit your own profile.", http.HTTPStatus.FORBIDDEN, None

    users = backend.db.user.get_users(ids=kwargs["id"])
    if not users:
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None
    original_object = users[0]

    requester = get_authenticated_user(handler)
    if requester is not None and requester.id != original_object.id:
        return None, http.HTTPStatus.FORBIDDEN, None

    # Check if role is changed, disallow.
    if kwargs["role"] != original_object.role:
        return "You cannot change your own role.", http.HTTPStatus.FORBIDDEN, None

    # Update user properties if changed, then update password if provided.
    updated_object = None
    if kwargs["name"] != original_object.name or kwargs["email"] != original_object.email:
        updated_object = backend.db.user.User(original_object.id, kwargs["name"], kwargs["email"],
                                              kwargs["username"], original_object.role,
                                              original_object.password, original_object.salt)
        posted_object = backend.db.user.upsert_user(updated_object)
        if posted_object == app_constants.ERROR:
            return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None
        updated_object = posted_object

    # 2. Update password if provided.
    if kwargs["password"] is not None and kwargs["password"] != "":
        update_status = backend.db.user.update_user_password(kwargs["username"], kwargs["password"])
        if update_status == app_constants.ERROR:
            return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    if updated_object is None:
        updated_object = original_object

    return updated_object.to_secure_dict(), http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("id", "username")
def get_user(handler, path, **kwargs):
    """Get a user profile by ID or username."""
    print("Get User function called")

    # Robust parameter normalization via utils.
    list_params = ["id", "username"]
    scalar_params = ["user_username", "user_id"]

    # Using default splitter (comma) for list params
    app_utils.normalize_list_params(kwargs, list_params)
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    # Only get self. TODO, if user has approriate role, may bypass.
    requester = get_authenticated_user(handler)
    if requester is not None:
        kwargs['user_username'] = requester.username
        kwargs['user_id'] = requester.id
    if kwargs["username"] and (kwargs["user_username"] not in kwargs["username"]):
        return "You can only view your own profile.", http.HTTPStatus.FORBIDDEN, None

    # Check ID match.
    if kwargs["id"]:
        if any(str(x) != str(kwargs["user_id"]) for x in kwargs["id"]):
            return "You can only view your own profile.", http.HTTPStatus.FORBIDDEN, None

    # Accept ids in the form of list, int, string.
    users = backend.db.user.get_users(ids=kwargs["id"] or [kwargs['user_id']],
                                      usernames=kwargs["username"])
    if not users:
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None
    user = users[0]
    if not isinstance(user, backend.db.user.User):
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    return user.to_secure_dict(), http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("ids", "keywords", "roles")
def filter_users(handler, path, **kwargs):
    """Filter user accounts based on provided query parameters."""
    print("Filter Users function called")

    # Robust parameter normalization via utils.
    list_params = ["ids", "roles", "keywords"]
    app_utils.normalize_list_params(kwargs, list_params)
    app_utils.normalize_scalar_params(kwargs, ["user_username"])

    # Initialize a set of IDs that meets filter constraints.
    # If a list of ids is given, filter this list, otherwise filter starting with all possible.
    if kwargs["ids"] is not None:
        candidate_ids = set(kwargs["ids"])
    else:
        candidate_ids = None

    # Check if any database filters are active.
    db_filters = ["roles", "keywords"]
    has_db_filters = any(kwargs.get(key) is not None for key in db_filters)

    # Only call the database filter if necessary.
    # Note: get_matching_users handles None for ids/roles/keywords by ignoring them (returning all if all are None)

    # Check if we have any ids to filter by
    ids_list = list(candidate_ids) if candidate_ids is not None else None

    user_list = backend.db.user.get_matching_users(
        ids=ids_list,
        roles=kwargs["roles"],
        keywords=kwargs["keywords"]
    )

    # Only admins see usernames and roles; viewers only get IDs and names for attribution.
    requester = get_authenticated_user(handler)
    is_admin = (requester is not None
                and role_dict.get(requester.role, 0) >= role_dict["admin"])

    user_list = [user.to_list_dict(include_identity=is_admin) for user in user_list]

    return user_list, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("id")
def delete_user(handler, path, **kwargs):
    """Delete a user account from the database."""
    print("Delete User function called")

    scalar_params = ["id", "user_username"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    # Get the requesting user.
    requesting_user = get_authenticated_user(handler)
    if requesting_user is None:
        return None, http.HTTPStatus.UNAUTHORIZED, None

    # Require owner role to delete user accounts.
    if role_dict[requesting_user.role] < 4:
        return "Only owners can delete users.", http.HTTPStatus.FORBIDDEN, None

    if kwargs["id"] is None:
        return "Missing required field: id.", http.HTTPStatus.BAD_REQUEST, None
    if not backend.db.user.get_users(ids=[kwargs["id"]]):
        return "User not found.", http.HTTPStatus.NOT_FOUND, None

    # Delete the requested user.
    status = backend.db.user.delete_user(kwargs["id"])
    if status == app_constants.ERROR:
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    return None, http.HTTPStatus.OK, None


# pylint: disable=unused-argument, too-many-return-statements
@app_utils.accepts()
def authenticate_user(handler, path, min_role_required=None, **kwargs):
    """Authenticate a user session token and verify required role permissions."""
    print("Auth user function called")

    scalar_params = ["user_username", "user_token"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    authorization = getattr(handler, 'headers', {}).get('Authorization')
    if isinstance(authorization, str) and authorization:
        scheme, _, token = authorization.partition(' ')
        if scheme.lower() != 'bearer' or not token or token != token.strip():
            return None, http.HTTPStatus.UNAUTHORIZED, image_cookie()
        session, retry_after = _AUTHENTICATION_LIMITER.verify_token(
            handler, kwargs.get('user_username') or '',
            lambda: backend.db.user.get_session_user(token))
        if retry_after is not None:
            return _authentication_retry_response(retry_after)
        if session is None:
            return None, http.HTTPStatus.UNAUTHORIZED, image_cookie()
        user = session[0]
        if ((kwargs.get('user_token') is not None and kwargs['user_token'] != token)
                or (kwargs.get('user_username') is not None
                    and kwargs['user_username'] != user.username)):
            return None, http.HTTPStatus.UNAUTHORIZED, image_cookie()
        handler.authenticated_user = user
        if (min_role_required is not None
                and role_dict.get(user.role, -1) < role_dict[min_role_required]):
            return None, http.HTTPStatus.FORBIDDEN, None
        return None, http.HTTPStatus.OK, image_cookie(token)

    # Check if user and token are provided.
    if ((kwargs.get("user_username") is None) or (kwargs.get("user_token") is None)):
        return None, http.HTTPStatus.UNAUTHORIZED, image_cookie()

    # Authenticate token through the shared HTTP attempt limiter.
    auth_success, retry_after = _AUTHENTICATION_LIMITER.verify_token(
        handler,
        kwargs["user_username"],
        lambda: backend.db.user.authenticate_user(
            username=kwargs["user_username"], token=kwargs["user_token"]
        ),
    )
    if retry_after is not None:
        return _authentication_retry_response(retry_after)

    if auth_success == app_constants.ERROR:
        print("Could not authenticate user.")
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    if auth_success is not True:
        print("User not authorized.")
        return None, http.HTTPStatus.UNAUTHORIZED, image_cookie()

    users = backend.db.user.get_users(usernames=kwargs["user_username"])
    if not users or not isinstance(users[0], backend.db.user.User):
        # This should not happen if token auth succeeded, but as a safeguard.
        print("User not found for token authentication.")
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None
    user = users[0]

    # Attach authenticated user to the handler to avoid redundant database reads in the target function.
    handler.authenticated_user = user

    # If a minimum role is required, check the user's role.
    if min_role_required is not None:
        user_role_level = role_dict.get(user.role)
        required_role_level = role_dict.get(min_role_required)

        if user_role_level is None or required_role_level is None:
            print("User role or required role not found.")
            # This indicates a configuration error (e.g., a typo in the role name).
            return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

        if user_role_level < required_role_level:
            print(f"Insufficient permissions for role {user.role}.")
            return None, http.HTTPStatus.FORBIDDEN, None

    return None, http.HTTPStatus.OK, image_cookie(kwargs['user_token'])


def get_authenticated_user(handler):
    """Return the User verified for this request, or None if unauthenticated."""
    return getattr(handler, "authenticated_user", None)


# pylint: disable=unused-argument
@app_utils.accepts("username", "password")
def login_user(handler, path, **kwargs):
    """Authenticate a user by username and password, returning session token on success."""
    print("Login user function called")

    scalar_params = ["username", "password"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    # Login through the shared HTTP attempt limiter.
    user_obj, retry_after = _AUTHENTICATION_LIMITER.verify_password(
        handler,
        kwargs["username"],
        lambda: backend.db.user.login_user(
            username=kwargs["username"], password=kwargs["password"]
        ),
    )
    if retry_after is not None:
        return _authentication_retry_response(retry_after)

    # No login in case no user object is returned or there's an error.
    if user_obj is None or user_obj == app_constants.ERROR:
        return False, http.HTTPStatus.OK, None

    # Stale-session image requests count as token failures; a fresh login must not stay locked out by them.
    _AUTHENTICATION_LIMITER.forgive_token_failures(handler)
    return user_obj, http.HTTPStatus.OK, image_cookie(user_obj['token'])


# pylint: disable=unused-argument
@app_utils.accepts()
def logout_user(handler, path, **kwargs):
    """Revoke an active user session token."""
    print("Logout user function called")

    scalar_params = ["user_username", "user_token"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    authorization = getattr(handler, 'headers', {}).get('Authorization')
    if isinstance(authorization, str) and authorization:
        result = authenticate_user(handler, path, **kwargs)
        if result[1] != http.HTTPStatus.OK:
            return result
        kwargs['user_username'] = handler.authenticated_user.username
        kwargs['user_token'] = authorization.partition(' ')[2]

    if ((kwargs.get("user_username") is None) or (kwargs.get("user_token") is None)):
        print("Warning: no token to delete.")
        return None, http.HTTPStatus.OK, image_cookie()

    # Delete the token from database.
    logout_status = backend.db.user.logout_user(
        username=kwargs["user_username"], token=kwargs["user_token"])

    if logout_status is app_constants.ERROR:
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    return None, http.HTTPStatus.OK, image_cookie()


def image_cookie(token=None):
    """Build the image-only cookie without extending its server-side session."""
    session = backend.db.user.get_session_user(token) if token else None
    cookie = http.cookies.SimpleCookie()
    cookie[IMAGE_COOKIE] = token if session else ''
    cookie[IMAGE_COOKIE]['path'] = '/static/frontend-images/'
    cookie[IMAGE_COOKIE]['httponly'] = True
    cookie[IMAGE_COOKIE]['samesite'] = 'Lax'
    cookie[IMAGE_COOKIE]['max-age'] = max(0, session[1] - int(time.time())) if session else 0
    return {'Set-Cookie': cookie.output(header='').strip()}


class _AuthenticationAttemptLimiter:
    """Limit authentication verification attempts within one server process."""

    def __init__(self, attempt_limit, window_seconds, clock=time.monotonic):
        """Initialize an authentication limiter."""
        self._attempt_limit = attempt_limit
        self._window_seconds = window_seconds
        self._clock = clock
        self._failures = {}
        self._pending = {}
        self._next_attempt_id = 0
        self._lock = threading.Lock()

    @staticmethod
    def _key(operation, handler, username):
        """Return the operation, direct peer, and normalized-user bucket key."""
        client_address = getattr(handler, "client_address", ())
        peer_address = client_address[0] if client_address else ""
        normalized_username = (
            username.strip().casefold() if isinstance(username, str) else ""
        )
        return operation, str(peer_address), normalized_username

    def _prune_failures(self, key, now):
        """Remove failures outside the rolling window and return recent failures."""
        cutoff = now - self._window_seconds
        failures = self._failures.get(key, deque())
        while failures and failures[0] <= cutoff:
            failures.popleft()

        if failures:
            self._failures[key] = failures
        else:
            self._failures.pop(key, None)
        return failures

    def verify_password(self, handler, username, verifier):
        """Run one password verifier under its matching attempt bucket."""
        return self._verify(
            self._key("password", handler, username),
            verifier,
            lambda result: bool(result) and result != app_constants.ERROR,
        )

    def verify_token(self, handler, username, verifier):
        """Run one session-token verifier under its matching attempt bucket."""
        # Unguessable random session tokens let a browser's parallel requests verify concurrently; failures still close the bucket.
        return self._verify(
            self._key("token", handler, username),
            verifier,
            lambda result: (result is True or
                            (isinstance(result, tuple) and isinstance(result[0], backend.db.user.User))),
            reserve_capacity=False,
        )

    def _verify(self, key, verifier, succeeded, reserve_capacity=True):
        """Run one verification attempt when the bucket has capacity; a full token bucket still admits valid tokens."""
        with self._lock:
            now = self._clock()
            failures = self._prune_failures(key, now)
            pending = self._pending.get(key, {}) if reserve_capacity else {}
            bucket_full = len(failures) + len(pending) >= self._attempt_limit
            if bucket_full and reserve_capacity:
                return None, self._retry_after(failures, now)

            attempt_id = None
            if reserve_capacity:
                attempt_id = self._next_attempt_id
                self._next_attempt_id += 1
                self._pending.setdefault(key, {})[attempt_id] = now

        try:
            result = verifier()
            verification_succeeded = succeeded(result)
        except Exception:
            with self._lock:
                self._finish_pending(key, attempt_id)
            raise

        with self._lock:
            self._finish_pending(key, attempt_id)
            if verification_succeeded:
                # A valid token admitted past a full bucket leaves the recorded failures in place.
                if not bucket_full:
                    self._failures.pop(key, None)
            else:
                failure_time = self._clock()
                failures = self._prune_failures(key, failure_time)
                failures.append(failure_time)
                self._failures[key] = failures
                if bucket_full:
                    return None, self._retry_after(failures, failure_time)

        return result, None

    def _retry_after(self, failures, now):
        """Return whole seconds until the oldest failure in the window expires."""
        retry_after = self._window_seconds
        if failures:
            retry_after = self._window_seconds - (now - failures[0])
        return max(1, math.ceil(retry_after))

    def _finish_pending(self, key, attempt_id):
        """Remove an attempt reservation while the limiter lock is held."""
        pending = self._pending.get(key)
        if pending is None:
            return

        pending.pop(attempt_id, None)
        if not pending:
            self._pending.pop(key, None)

    def forgive_token_failures(self, handler):
        """Drop this peer's token failures after a successful password login."""
        peer_address = self._key("token", handler, "")[1]
        with self._lock:
            for key in [key for key in self._failures
                        if key[0] == "token" and key[1] == peer_address]:
                del self._failures[key]

    def clear(self):
        """Clear all process-local limiter state."""
        with self._lock:
            self._failures.clear()
            self._pending.clear()


_AUTHENTICATION_LIMITER = _AuthenticationAttemptLimiter(
    AUTHENTICATION_ATTEMPT_LIMIT,
    AUTHENTICATION_WINDOW_SECONDS,
)


def _authentication_retry_response(retry_after):
    """Return the shared HTTP response for a blocked authentication attempt."""
    return None, http.HTTPStatus.TOO_MANY_REQUESTS, {
        "Retry-After": str(retry_after)
    }
