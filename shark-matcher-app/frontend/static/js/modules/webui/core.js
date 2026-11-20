const roleDict = {
  'viewer': 0,
  'collaborator': 1,
  'admin': 2,
  'dev': 3,
  'owner': 4,
};

// Stores user context including profile details, role, and authentication tokens.
let contextUser = undefined;

function getContextUser() {
  return contextUser;
}

function setContextUser(user) {
  contextUser = user;
}

function clearContextUser() {
  contextUser = undefined;
}

// Function to get current user mode for other modules
function getCurrentUserMode() {
  return window.sharkMatcher?.config?.userMode || localStorage.getItem('userMode') || 'viewing';
}

function canEdit() {
  return getCurrentUserMode() === 'editing';
}

function canUnmatch() {
  const mode = getCurrentUserMode();
  const role = roleDict[getContextUser()?.user_role] ?? 0;
  return (mode === 'editing' && role >= 2) || (mode === 'demo' && role >= 1);
}

function canMerge() {
  const mode = getCurrentUserMode();
  const role = roleDict[getContextUser()?.user_role] ?? 0;
  return (mode === 'editing' && role >= 2) || (mode === 'demo' && role >= 1);
}

function canManageDemo() {
  return (roleDict[getContextUser()?.user_role] ?? 0) >= 1;
}

function isReadOnlyMode() {
  const mode = getCurrentUserMode();
  return mode === 'viewing' || mode === 'demo';
}

// Return a standardized URL hash path with sorted query parameters.
function formHashPath(path, params = {}) {
  path = basicPathClean(path);

  // Create a fake URL for parsing and formatting.
  const url = URL.parse(path, 'http://localhost');

  // Add the params.
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.append(key, value);
  }

  // Sort the params.
  url.searchParams.sort();

  // Pull out the path (which comes with a leading slash).
  path = url.pathname.replace(/^\//, '');

  // Append query parameters including any leading question mark.
  path += url.search;

  return '#' + path;
}

function basicPathClean(path) {
  path = path.trim();

  // Remove leading hashes, slashes, and spaces.
  path = path.replace(/^[\s/#]+/, '');

  // Remove tailing slashes and spaces.
  path = path.replace(/[/\s]+$/, '');

  return path;
}

// Loader.
function loading() {
  // TODO
  console.log('Loading');
  document.querySelector('.content').innerHTML = '<h1>LOADING ...</h1>';
}

// Redirection.
function redirect(path = '') {
  window.location.hash = path;
}

function redirectHome() {
  redirect('');
}

function redirectLogin() {
  redirect('login');
}

function redirectLogout() {
  redirect('logout');
}

export {
  canEdit,
  canUnmatch,
  canMerge,
  canManageDemo,
  isReadOnlyMode,
  basicPathClean,
  formHashPath,

  clearContextUser,
  getContextUser,
  setContextUser,
  getCurrentUserMode,

  redirect,
  redirectHome,
  redirectLogin,
  redirectLogout,

  loading,

  roleDict,
};
