// Endpoint functions for user objects.

import * as Core from './core.js';

function filter_users(ids, roles, keywords) {
  return Core.sendRequest({
    endpoint: 'user/filter',
    payload: {
      'ids': ids,
      'roles': roles,
      'keywords': keywords,
    },
  });
}

function post_user(id, name, email, username, role, password) {
  return Core.sendRequest({
    endpoint: 'user/post',
    payload: {
      'id': id,
      'name': name,
      'email': email,
      'username': username,
      'role': role,
      'password': password,
    },
  });
}

function delete_user(id) {
  return Core.sendRequest({
    endpoint: 'user/delete',
    payload: {
      'id': id,
    },
  });
}

function login_user(username, password) {
  return Core.sendRequest({
    endpoint: 'user/login',
    payload: {
      'username': username,
      'password': password,
    },
  });
}

function logout_user() {
  return Core.sendRequest({
    endpoint: 'user/logout',
  });
}

export {
  filter_users,
  post_user,
  delete_user,
  login_user,
  logout_user,
};
