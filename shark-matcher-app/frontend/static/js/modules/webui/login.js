import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Log from './log.js';
import * as Routes from './routes.js';
import * as Util from './util.js';


function init() {
  Routes.addRoute(/^login$/, handlerLogin, 'login', {login: false});
}


function handlerLogin(path, params, context) {
  // Header content.
  let html = `
        <div class="tab-content"></div>
        </div>
    `;
  document.querySelector('.header-content').innerHTML = html;

  // Remove global controls that might persist from home page
  const contrastContainer = document.querySelector('.contrast-toggle-container');
  if (contrastContainer) contrastContainer.remove();
  const modeContainer = document.querySelector('.mode-selector-container');
  if (modeContainer) modeContainer.remove();

  // Results content.
  html = `
        <div class="tab-content login">
            <h3>Login</h3>
            <form class="input-section login">
                <div class="input-item">
                    <label for="login-username-input">Username:</label>
                    <input
                        class="login username-input"
                        id="login-username-input"
                        type="text"
                        name="username"
                        autocomplete="username"
                        placeholder="username"/>
                    </div>
                <div>
                <label for="login-password-input">Password:</label>
                <input
                    class="login password-input"
                    id="login-password-input"
                    type="password"
                    name="password"
                    autocomplete="current-password"
                    placeholder="password"/>
                </div>
                <div class="input-item">
                    <input
                        class="login submit"
                        type="submit"
                        value="Sign In"/>
                </div>
            </form>
        </div>
    `;
  let doc = new DOMParser().parseFromString(html, 'text/html');
  // The form submit covers both the button and the Enter key.
  doc.querySelector('form.login').addEventListener('submit', function(event) {
    event.preventDefault();
    executeLogin(context);
  });
  document.querySelector('.tab-content')
      .replaceWith(doc.querySelector('.tab-content.login'));

  // Add results content.
  html = `
    <div class='content'>
        <div class="tab-results"></div>
    </div>
    `;
  doc = new DOMParser().parseFromString(html, 'text/html');
  document.querySelector('.content').replaceWith(doc.querySelector('.content'));
}

// Path update functions.

// API execution functions.
function executeLogout(context) {
  // Clear local state whether or not the server call succeeds — a failed logout
  // must still leave nothing behind on this machine.
  function clearAndRedirect() {
    Core.clearContextUser();
    SharkMatcher.clearAllLocal();
    return Core.redirectLogin();
  }

  SharkMatcher.User.logout_user()
      .then(clearAndRedirect)
      .catch(function(result) {
        console.log('Warning Error logging out.');
        Log.warn(result, context);
        return clearAndRedirect();
      });
}

async function executeLogin(context) {
  if (SharkMatcher.hasCredentials()) {
    alert('Already logged in. Logout first to login again.', context);
    return;
  }

  const username = document.querySelector('.login.username-input').value;
  const cleartext = document.querySelector('.login.password-input').value;

  if (username.length < 1) {
    alert('No username provided for login.', context);
    return;
  }

  if (cleartext.length < 1) {
    alert('No password provided for login.', context);
    return;
  }

  try {
    // Encode password.
    const encoded = await Util.sha256(cleartext);
    const result = await SharkMatcher.User.login_user(username, encoded);

    if (result === false) {
      alert('Incorrect username/password combination.');
      return;
    }

    // Set credientials in local browser storage.
    SharkMatcher.setCredentials(
        result['email'], result['id'], result['name'], result['role'],
        result['token'], result['username']);
    const contextUser = SharkMatcher.getCredentials();

    // Set the context user, which is used to pass user data with
    // all API calls.
    Core.setContextUser(contextUser);

    Util.refreshCache('users', context);

    // Go to homepage.
    Core.redirectHome();
  } catch (error) {
    console.log('Error logging in.');
    const message = Util.handleAPIError(error);
    if (message) {
      alert(`Error logging in: ${message}`);
    }
  }
}

// HTML render functions.

export {
  init,
  executeLogin,
  executeLogout,

};
