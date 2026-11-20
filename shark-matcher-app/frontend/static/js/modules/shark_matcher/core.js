// Shared browser API transport and local application-state primitives.

import * as Util from '/js/modules/webui/util.js';

const CREDENTIALS_KEY = 'SHARKMATCHER.CREDENTIALS';
const FILTERS_PREFIX = 'SHARKMATCHER.FILTERS.';

const EMAIL_KEY = 'user_email';
const ID_KEY = 'user_id';
const NAME_KEY = 'user_display_name';
const ROLE_KEY = 'user_role';
const TOKEN_KEY = 'user_token';
const USERNAME_KEY = 'user_username';

// One cache primitive for every piece of reference data the app keeps in the
// browser. Entries share a key prefix, an envelope shape ({time, value}), and
// the same corrupt-data recovery, so a new cache needs no bespoke TTL bookkeeping.
const CACHE_PREFIX = 'cache.';

// In-flight fetches, keyed the same way. Collapses the several callers that ask
// for the same data during one page render into a single request.
const inFlight = new Map();

function dataScope() {
  const config = window.sharkMatcher?.config;
  return config?.userMode === 'demo' ? `demo.${config.demo_round}.` : '';
}

// Read a cache entry, or null when absent, corrupt, or the wrong shape. A bad
// entry is dropped rather than left to throw and abort a render.
function readCacheEntry(key) {
  const storageKey = CACHE_PREFIX + dataScope() + key;
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) {
      return null;
    }
    const entry = JSON.parse(raw);
    if (entry && typeof entry.time === 'number') {
      return entry;
    }
  } catch {
    // Parsing failures use the shared invalid-entry cleanup below.
  }

  console.log(`Warning, discarding unreadable cache entry '${key}'.`);
  localStorage.removeItem(storageKey);
  return null;
}

// Synchronous read for render paths. Returns undefined when there is nothing
// usable cached, so callers can fall back to a placeholder.
function getCachedValue(key) {
  const entry = readCacheEntry(key);
  return entry ? entry.value : undefined;
}

function clearCache(key) {
  inFlight.delete(dataScope() + key);
  localStorage.removeItem(CACHE_PREFIX + dataScope() + key);
}

// Resolve to the cached value when it is younger than ttlMs, otherwise call
// fetcher and cache what it returns. Call clearCache(key) first to force a refetch.
function getCached(key, ttlMs, fetcher) {
  const scopedKey = dataScope() + key;
  if (inFlight.has(scopedKey)) {
    return inFlight.get(scopedKey);
  }

  const entry = readCacheEntry(key);
  if (entry && (Date.now() - entry.time) < ttlMs) {
    return Promise.resolve(entry.value);
  }

  // A request that clearCache dropped while it ran must not write its older value back.
  const isCurrent = () => inFlight.get(scopedKey) === request;
  const request = Promise.resolve()
      .then(fetcher)
      .then(function(value) {
        if (isCurrent()) {
          inFlight.delete(scopedKey);
          localStorage.setItem(
              CACHE_PREFIX + scopedKey, JSON.stringify({'time': Date.now(), 'value': value}));
        }
        return value;
      })
      .catch(function(error) {
        // Do not cache failures; the next caller retries.
        if (isCurrent()) {
          inFlight.delete(scopedKey);
        }
        throw error;
      });

  inFlight.set(scopedKey, request);
  return request;
}

// Session keys to clear on logout, excluding persistent device preferences like theme and contrast.
const SESSION_KEYS = [CREDENTIALS_KEY, 'userMode'];

// Clear all cached server responses and session values on logout to prevent data leakage across
// users.
function clearAllLocal() {
  inFlight.clear();

  const keys = [...SESSION_KEYS];
  for (const key of Object.keys(localStorage)) {
    if (key.startsWith(CACHE_PREFIX) || key.startsWith(FILTERS_PREFIX)) {
      keys.push(key);
    }
  }

  for (const key of keys) {
    localStorage.removeItem(key);
  }
}

// Search filters are stored per tab. The search and match tabs have separate
// filter sets, and sharing one key let each overwrite the other's.
function filtersKey(page) {
  return FILTERS_PREFIX + dataScope() + page;
}

function getFilters(page) {
  try {
    const filters = localStorage.getItem(filtersKey(page));
    return filters ? JSON.parse(filters) : undefined;
  } catch {
    console.log(`Warning, discarding unreadable filters for '${page}'.`);
    localStorage.removeItem(filtersKey(page));
    return undefined;
  }
}

// Merge the supplied filter values over whatever is already stored for the tab.
// Keys with a null/undefined value are left untouched.
function setFilters(page, filters) {
  const stored = getFilters(page) || {};

  for (const [key, value] of Object.entries(filters)) {
    if (value !== null && value !== undefined) {
      stored[key] = value;
    }
  }

  return localStorage.setItem(filtersKey(page), JSON.stringify(stored));
}

function clearFilters(page) {
  localStorage.removeItem(filtersKey(page));
}

// Functions for checking, storing, clearing credentials stored
// in local storage.
function hasCredentials() {
  return Boolean(localStorage.getItem(CREDENTIALS_KEY));
}

function getCredentials() {
  const credentials = localStorage.getItem(CREDENTIALS_KEY);
  if (credentials) {
    return JSON.parse(credentials);
  }

  return undefined;
}

function setCredentials(email, id, name, role, token, username) {
  const credentials = {
    [EMAIL_KEY]: email,
    [ID_KEY]: id,
    [NAME_KEY]: name,
    [ROLE_KEY]: role,
    [TOKEN_KEY]: token,
    [USERNAME_KEY]: username,
  };

  return localStorage.setItem(CREDENTIALS_KEY, JSON.stringify(credentials));
}

// Functions for managing UI Theme in local storage.
const THEME_KEY = 'SHARKMATCHER.THEME';

function getTheme() {
  return localStorage.getItem(THEME_KEY);
}

function setTheme(themeName) {
  if (themeName) {
    localStorage.setItem(THEME_KEY, themeName);
  } else {
    localStorage.removeItem(THEME_KEY);
  }
}

// Custom error class for API errors.
class APIError extends Error {
  constructor(message, status, endpoint) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.endpoint = endpoint;
    this.isAuthError = (status === 401 || status === 403);
  }
}

async function resolveAPIResponse(response) {
  // Check if we should read the full body.
  const readBody = response.headers.get('qgg-body') ?? false;

  if (!response.ok && !readBody) {
    console.error('API returned an error.', response);
    return Promise.reject(
        new APIError('API returned an error.', response.status, response.url));
  }

  const body = await response.json();

  if (!body.success) {
    // Catch unauthorized responses.
    if (response.status === 401) {
      return Promise.reject(new APIError(
          body.message || 'Unauthorized API call.',
          response.status, body.endpoint));
    }
    return Promise.reject(new APIError(
        body.message || 'API call failed.', response.status, body.endpoint));
  }

  return Promise.resolve(body.content);
}

async function resolveAPIError(response) {
  console.error('Failed to send API request.');
  console.error(response);

  let message = 'Failed to send API request.';
  let status = 0;

  if (response instanceof Response) {
    status = response.status;
    try {
      message = await response.text();
      // Try to parse as JSON if possible to get structured message
      try {
        const json = JSON.parse(message);
        if (json.message) message = json.message;
      } catch { }
    } catch {
      message = response.statusText || message;
    }
  } else if (typeof response === 'string') {
    message = response;
  } else if (response.message) {
    message = response.message;
  }

  return Promise.reject(
      new APIError(JSON.stringify(message), status, 'unknown'));
}

const ASYNC_ENDPOINTS = ['user/auth', 'embeddings/update', 'embeddings/status'];
function sendRequest({endpoint = undefined, payload = {}}) {
  if (!endpoint) {
    throw new Error('Endpoint not specified.');
  }

  // Show the loader for most API routes.
  if (!ASYNC_ENDPOINTS.includes(endpoint)) {
    Util.loadingStart();
  }

  // Credentials check.
  const credentials = getCredentials();

  if (credentials) {
    payload[ID_KEY] = credentials.user_id;
    payload[USERNAME_KEY] = credentials.user_username;
    payload[TOKEN_KEY] = credentials.user_token;
  }

  const url = `/api/${endpoint}`;
  const body = JSON.stringify(payload);
  const headers = {'Content-Type': 'application/json'};
  const config = window.sharkMatcher?.config;
  if (config?.userMode === 'demo') {
    headers['X-Shark-Demo'] = 'true';
    headers['X-Shark-Demo-Round'] = config.demo_round;
  }

  const response = fetch(url, {
    'method': 'POST',
    'body': body,
    headers,
  });

  return response.finally(() => {
    if (!ASYNC_ENDPOINTS.includes(endpoint)) Util.loadingStop();
  }).then(resolveAPIResponse, resolveAPIError).then((result) => {
    if (config?.userMode === 'demo' && /\/(merge|unmatch|randomize|clear)$/.test(endpoint)) {
      clearCache('dataset_stats');
      clearCache('selectors');
    }
    return result;
  });
}

export {
  getCached,
  getCachedValue,
  clearCache,
  clearAllLocal,

  getCredentials,
  hasCredentials,
  setCredentials,

  getFilters,
  setFilters,
  clearFilters,

  sendRequest,

  getTheme,
  setTheme,
};
