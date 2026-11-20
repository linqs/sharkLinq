"""
User class and database CRUD functions for user objects.
"""

import hashlib
import secrets
import time

import argon2

from core import app_constants
from core import app_utils

PASSWORD_SCHEME = "argon2-v1"
SESSION_TOKEN_BYTES = 32
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60

_PASSWORD_HASHER = argon2.PasswordHasher()


def _hash_password(password):
    """Hash a browser-derived password using the current Argon2 configuration."""
    return _PASSWORD_HASHER.hash(password), PASSWORD_SCHEME


def _verify_password(password_hash, password):
    """Verify a browser-derived password against an encoded Argon2 hash."""
    if not isinstance(password_hash, str) or not isinstance(password, str):
        return False

    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (argon2.exceptions.InvalidHashError,
            argon2.exceptions.VerificationError):
        return False


class User: # pylint: disable=too-many-instance-attributes
    """
    User class.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, redefined-builtin
    def __init__(self, id=None, name=None, email=None, username=None,
                 role=None, password=None, salt=None):
        """Initialize a User record with the provided attributes."""
        self.id = id
        self.name = name
        self.email = email
        self.username = username
        self.role = role
        self.password = password
        self.salt = salt

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False): # pylint: disable=unused-argument
        """Convert the User object into a dictionary representation, including sensitive fields."""
        item_dict = {"id": self.id,
                     "name": self.name,
                     "email": self.email,
                     "username": self.username,
                     "role": self.role,
                     "password": self.password,
                     "salt": self.salt}

        return item_dict

    def to_secure_dict(self):
        """Convert the User object into a secure dictionary representation, excluding password and salt."""
        item_dict = {"id": self.id,
                "name": self.name,
                "email": self.email,
                "username": self.username,
                "role": self.role}

        return item_dict

    def to_list_dict(self, include_identity=False):
        """Convert the User object into the representation used by multi-user listing endpoints."""
        item_dict = {"id": self.id,
                     "name": self.name}

        if include_identity:
            item_dict["username"] = self.username
            item_dict["role"] = self.role

        return item_dict

def db_to_user(db_object):
    """Convert a database row tuple into a User object."""
    if db_object is None:
        return app_constants.ERROR

    user_id = db_object[0]
    name = db_object[1]
    email = db_object[2]
    username = db_object[3]
    role = db_object[4]
    password = db_object[5]
    salt = db_object[6]

    return User(id=user_id, name=name, email=email, username=username,
                role=role, password=password, salt=salt)

class Token:
    """
    Stored browser-session token.
    """
    # pylint: disable=redefined-builtin
    def __init__(self, id=None, token=None, user=None, expires=None):
        """Initialize a Token record with the provided attributes."""
        self.id = id
        self.token = token
        self.user = user
        self.expires = expires

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False): # pylint: disable=unused-argument
        """Convert the Token object into a dictionary representation."""
        item_dict = {"id": self.id,
                     "token": self.token,
                     "user": self.user,
                     "expires": self.expires}

        return item_dict

def db_to_token(db_object):
    """Convert a database row tuple into a Token object."""
    if db_object is None:
        return app_constants.ERROR

    token_id = db_object[0]
    token = db_object[1]
    user = db_object[2]
    expires = db_object[3]

    return Token(token_id, token, user, expires)

# Create/update functions.
# pylint: disable=too-many-arguments, too-many-positional-arguments
def create_user(name, email, username, role, password, connection=None):
    """Create a user record with hashed password in the database."""
    if not isinstance(password, str) or not password:
        return app_constants.ERROR
    if get_users(usernames=[username], connection=connection):
        return app_constants.ERROR

    password_hash, password_scheme = _hash_password(password)
    user = User(
        name=name,
        email=email,
        username=username,
        role=role,
        password=password_hash,
        salt=password_scheme,
    )
    return upsert_user(user, connection=connection)


def upsert_user(user, connection=None):
    """Insert or update a user record in the database."""

    data = ({"id": user.id,
             "name": user.name,
             "email": user.email,
             "username": user.username,
             "role": user.role,
             "password": user.password,
             "salt": user.salt})
    sql = """INSERT INTO Users (id, name, email, username, role, password, salt)
    VALUES (:id, :name, :email, :username, :role, :password, :salt)
    ON CONFLICT(id) DO UPDATE SET name = :name, email = :email,
    username = :username, role = :role, password = :password, salt = :salt
    RETURNING *;"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    return db_to_user(result[0])

def upsert_token(token, connection=None):
    """Insert or update a token record in the database."""
    data = ({"id": token.id,
             "token": token.token,
             "user": token.user,
             "expires": token.expires})
    sql = """INSERT INTO Tokens (id, token, user, expires)
    VALUES (:id, :token, :user, :expires)
    ON CONFLICT(id) DO UPDATE SET token = :token, user = :user,
    expires = :expires RETURNING *;"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    return db_to_token(result[0])

# Getter functions.
def get_users(ids=None, usernames=None, limit=None, connection=None):
    """Retrieve users from the database matching the provided IDs or usernames."""
    # Build where clauses.
    where_clauses = []
    data = {}

    if ids is not None:
        if not isinstance(ids, list):
            ids = [ids]
        sql_ids, sql_values = app_utils.build_sql_placeholders(ids, basename="id")
        where_clauses.append(f"(id IN {sql_ids})")
        data.update(sql_values)

    if usernames is not None:
        if not isinstance(usernames, list):
            usernames = [usernames]
        sql_usernames, sql_values = app_utils.build_sql_placeholders(
            usernames, basename="username")
        where_clauses.append(f"(username IN {sql_usernames})")
        data.update(sql_values)

    ids_to_fetch = []

    limit_string = ""
    if limit is not None:
        limit_string = f" LIMIT {limit}"

    # If no filters provided, get keys for all.
    if not where_clauses:
        sql = f"""SELECT id, name, email, username, role, password, salt FROM Users{limit_string};"""
        user_rows = app_utils.execute_sql(sql, {}, connection=connection)

        if user_rows != app_constants.ERROR:
            ids_to_fetch = [row[0] for row in user_rows]
    else:
        where_string = " WHERE " + " OR ".join(where_clauses)

        sql = f"""SELECT id, name, email, username, role, password, salt FROM Users{where_string}{limit_string};"""
        user_rows = app_utils.execute_sql(sql, data, connection=connection)

    if user_rows == app_constants.ERROR:
        return []

    user_dict = {row[0]: db_to_user(row) for row in user_rows}

    # If we filtered, we need to establish the retrieval order or just use the keys.
    if where_clauses:
        ids_to_fetch = list(user_dict.keys())

    return [user_dict[user_id] for user_id in ids_to_fetch if user_id in user_dict]


# pylint: disable=unused-argument, too-many-locals# pylint: disable=too-many-locals
def get_matching_users(ids=None, roles=None, keywords=None,
                       allow_unknown=False, limit=None, connection=None):
    """Retrieve a list of User objects matching the specified filtering criteria."""
    # Build WHERE clauses and query data.
    where_clauses = []
    keyword_clause = ""
    data = {}

    # Value filters are currently unused.

    # List filters.
    for filter_item in [["id", ids, "id"], ["name", keywords, "keyword"], ["role", roles, "role"]]:
        if filter_item[1] is not None and filter_item[1] != "":
            sql_ids, sql_values = app_utils.build_sql_placeholders(
                filter_item[1], basename=f"x_{filter_item[2]}")
            if filter_item[2] == "keyword":
                keyword_clause = f"{filter_item[0]} COLLATE NOCASE in {sql_ids} OR "
            else:
                where_clauses.append(f"({filter_item[0]} COLLATE NOCASE in {sql_ids})")
            data = data | sql_values

    where_string = ""
    if len(where_clauses) == 0 and keyword_clause != "":
        where_string = " WHERE " + keyword_clause[:-4]
    elif len(where_clauses) != 0:
        where_string = " WHERE " + keyword_clause + app_utils.list_to_string(
            where_clauses, " AND ")

    # Build the limit string.
    limit_string = ""
    if limit is not None:
        limit_string = f" LIMIT ${limit}"
        data = data | {"limit": limit}

    # Build SQL statement.
    sql = f"""SELECT id, name, email, username, role, password, salt FROM Users{where_string}{limit_string};"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    user_list = [db_to_user(row) for row in result]

    return user_list

def get_tokens(ids=None, tokens=None, connection=None):
    """Retrieve Token objects matching the provided IDs or token strings."""
    # Build where clauses.
    where_clauses = []
    data = {}

    if ids is not None:
        if not isinstance(ids, list):
            ids = [ids]
        sql_ids, sql_values = app_utils.build_sql_placeholders(ids, basename="id")
        where_clauses.append(f"(id IN {sql_ids})")
        data.update(sql_values)

    if tokens is not None:
        if not isinstance(tokens, list):
            tokens = [tokens]
        sql_tokens, sql_values = app_utils.build_sql_placeholders(tokens, basename="token")
        where_clauses.append(f"(token IN {sql_tokens})")
        data.update(sql_values)

    # If no filters provided, get all.
    if not where_clauses:
        ids = app_utils.get_all_ids("Tokens", connection=connection)
        if not ids:
            return []
        # Fallback to get all behavior if needed, or query directly.

    where_string = ""
    if where_clauses:
        where_string = " WHERE " + " OR ".join(where_clauses)

    sql = f"""SELECT id, token, user, expires FROM Tokens{where_string};"""

    token_rows = app_utils.execute_sql(sql, data, connection=connection)
    # pylint: disable=no-else-return
    if token_rows == app_constants.ERROR or not token_rows:
        return []

    return [db_to_token(row) for row in token_rows]

# Delete functions.
def delete_user(user_id, connection=None):
    """Delete a user record from the database by its ID."""
    if app_utils.id_exists(user_id, "Users", connection=connection) is False:
        return app_constants.ERROR

    data = {"id": user_id}
    sql = """DELETE FROM Users where id = :id;"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    return app_constants.SUCCESS

def delete_token(token_id, connection=None):
    """Delete a token record from the database by its ID."""
    if app_utils.id_exists(token_id, "Tokens", connection=connection) is False:
        return app_constants.ERROR

    data = {"id": token_id}
    sql = """DELETE FROM Tokens where id = :id;"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    return app_constants.SUCCESS

# Authentication functions.
def _session_token_digest(token):
    """Build the database lookup digest for an opaque browser session token."""
    if not isinstance(token, str) or not token:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_session_user(token, connection=None):
    """Resolve a live opaque session to its account and absolute expiry."""
    digest = _session_token_digest(token)
    if digest is None:
        return None
    tokens = get_tokens(tokens=[digest], connection=connection)
    if not tokens:
        return None
    session = tokens[0]
    try:
        expires = int(session.expires)
    except (TypeError, ValueError):
        expires = 0
    if expires <= int(time.time()):
        delete_token(session.id, connection=connection)
        return None
    users = get_users(ids=[session.user], connection=connection)
    if not users or not isinstance(users[0], User):
        return None
    return users[0], expires


def authenticate_user(username, token, connection=None):
    """Verify whether a session token matches the given username."""
    session = get_session_user(token, connection=connection)
    return session is not None and session[0].username == username

def login_user(username, password, connection=None):
    """Authenticate a user with username and password, returning account details and a session token."""
    users = get_users(usernames=[username], connection=connection)
    if not users:
        return app_constants.ERROR
    user = users[0]
    if not isinstance(user, User):
        return app_constants.ERROR

    # If user is authenticated, migrate its password representation when needed,
    # then generate a token and return the username/token combo.
    if _verify_password(user.password, password):
        if (user.salt != PASSWORD_SCHEME
                or _PASSWORD_HASHER.check_needs_rehash(user.password)):
            user.password, user.salt = _hash_password(password)
            migrated_user = upsert_user(user, connection=connection)
            if migrated_user == app_constants.ERROR:
                return app_constants.ERROR
            user = migrated_user

        # Generate the browser token and store only its lookup digest.
        token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
        token_digest = _session_token_digest(token)
        expires = int(time.time()) + SESSION_TTL_SECONDS
        generated_token = Token(None, token_digest, user.id, expires)
        token_status = upsert_token(generated_token, connection=connection)
        if token_status == app_constants.ERROR:
            print("Issue saving token.")
            return app_constants.ERROR

        user_dict = user.to_secure_dict()
        user_dict["token"] = token

        return user_dict

    return app_constants.ERROR

def logout_user(username, token, connection=None):
    """Log out a user by deleting their active session token."""
    # Get token and user from database.
    users = get_users(usernames=[username], connection=connection)
    if not users:
        return app_constants.SUCCESS
    user = users[0]

    # If this is not a valid user, there is no token, just log out.
    if not isinstance(user, User):
        return app_constants.SUCCESS

    token_digest = _session_token_digest(token)
    if token_digest is None:
        return app_constants.SUCCESS

    returned_tokens = get_tokens(tokens=[token_digest], connection=connection)
    if not returned_tokens:
        print("Warning, no token to delete.")
        return app_constants.SUCCESS
    returned_token = returned_tokens[0]

    if returned_token.user != user.id:
        return app_constants.SUCCESS

    status = delete_token(returned_token.id, connection=connection)

    return status

def update_user_password(username, password, connection=None):
    """Update the password for a specific user."""
    if not isinstance(password, str) or not password:
        return app_constants.ERROR

    # Get the original user object.
    users = get_users(usernames=[username], connection=connection)
    if not users:
        return app_constants.ERROR
    user = users[0]
    if not isinstance(user, User):
        return app_constants.ERROR

    user.password, user.salt = _hash_password(password)
    updated_user = upsert_user(user, connection=connection)
    if updated_user == app_constants.ERROR:
        return app_constants.ERROR

    return app_constants.SUCCESS

def derive_client_password(cleartext_password):
    """Derive the SHA-256 client password representation produced by the browser before transport."""
    return hashlib.sha256(cleartext_password.encode("utf-8")).hexdigest()
