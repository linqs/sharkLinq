import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
// import * as Log from './log.js';
import * as Util from './util.js';

const routes = [];

const DEFAULT_HANDLER = handlerNotFound;

// The current hash/location we are routed to.
// Should be prefixed with a hash symbol.
let currentHash = undefined;
let previousHash = undefined;

// Start listening for routing events and do an initial route.
function init() {
  window.addEventListener('hashchange', function() {
    route();
  });

  route();
}

// Register a route pattern with a handler accepting (path, pathParams, context).
function addRoute(pattern, handler,
    pageName = undefined,
    requirements = {login: true}) {
  routes.push([pattern, handler, requirements, pageName]);
}

// Route page content to the specified path or current location hash.
function route(rawPath = undefined) {
  if (!rawPath) {
    rawPath = Util.getLocationHash();
  }

  const [path, params] = parsePath(rawPath);

  // Form the canonical path we are on.
  const newHash = Core.formHashPath(path, params);

  // Skip routing if already on the target path to avoid infinite loops.
  if (newHash == currentHash) {
    return;
  }

  // Set the hash to the cleaned-up value.
  previousHash = currentHash;
  currentHash = newHash;
  window.location.hash = newHash;

  // Check all known routes.
  for (const [pattern, handler, requirements, pageName] of routes) {
    if (path.match(pattern)) {
      console.debug(`Routing '${path}' to ${handler.name}.`);
      return handlerWrapper(handler, path, params, pageName, requirements);
    }
  }

  // Fallback to the default route.
  console.log(`Unknown path '${path}'. Falling back to default route.`);
  return handlerWrapper(DEFAULT_HANDLER, path, params, undefined, {});
}

// Parse a raw path into a clean path and query parameters.
function parsePath(rawPath) {
  rawPath = Core.basicPathClean(rawPath);

  // Parse the raw path as a url to pull out parameters.
  const url = URL.parse(rawPath, 'http://localhost');

  // Pull out the path (which comes with a leading slash).
  const path = url.pathname.replace(/^\//, '');

  // Pull out the params into a simple object.
  const rawParams = url.searchParams;
  rawParams.sort();

  const params = {};
  for (const [key, value] of rawParams.entries()) {
    params[key] = value;
  }

  return [path, params];
}

// Do any setup for a handler, call the handler, then do any teardown.
function handlerWrapper(
    handler, path, params, pageName, requirements, context = {}) {
  const loggedIn = SharkMatcher.hasCredentials();

  // Redirect to a login if required.
  if (requirements.login && !loggedIn) {
    console.log('Redirecting to login.');
    return Core.redirectLogin();
  }

  // Anything that needs a login also needs a context user.
  context.user = Core.getContextUser();
  if (requirements.login && !context.user) {
    return fetchContextUser()
        .then(function(result) {
          // Call this function again to complete all checks.
          context.user = result;
          return handlerWrapper(
              handler, path, params, pageName, requirements, context);
        })
        .catch(function(result) {
          // Session is gone; clear cached data the same way an explicit logout does.
          SharkMatcher.clearAllLocal();
          return Core.redirectLogin();
        });
  }

  // Set the page inforamtion on the body for CSS targeting.
  if (pageName) {
    document.body.setAttribute('data-page', pageName);
  } else {
    document.body.setAttribute('data-page', '');
  }

  // Call the handler.
  return handler(path, params, context);
}

function fetchContextUser() {
  const contextUser = SharkMatcher.getCredentials();
  Core.setContextUser(contextUser);
  return Promise.resolve(Core.getContextUser());
}

function handlerNotFound(path, params) {
  document.querySelector('.content').innerHTML = `
        <h2>Location Not Found</h2>
        <p>We could not find the location '${Util.escapeHtml(path)}'.</p>
    `;
}

function getPreviousHash() {
  return previousHash;
}

export {
  init,
  addRoute,
  route,
  parsePath,
  getPreviousHash,
};
