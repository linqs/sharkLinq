import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Home from './home.js';
import * as Routes from './routes.js';
import * as Util from './util.js';

function init() {
  Routes.addRoute(/^settings$/, handlerSettings, 'settings');
}

function handlerSettings(path, params, context) {
  Home.handlerHome(path, params, context);

  // Tab content.
  let html = `
        <div class="tab-content settings">
            <div class="tab-title"></div>
        </div>
    `;
  let doc = new DOMParser().parseFromString(html, 'text/html');
  document.querySelector('.tab-content')
      .replaceWith(doc.querySelector('.tab-content.settings'));

  Home.renderTitle('settings', context);

  // Results content.
  // Get user role value and display permissions text if needed.
  let userRoleValue = 0;
  const currentUser = Core.getContextUser();
  if (currentUser && currentUser.user_role) {
    userRoleValue = Core.roleDict[currentUser.user_role];
  }

  let permissionsText = ``;
  if (userRoleValue < 2) {
    permissionsText = `
        <div class="input-item">
            <p>Updating selector options is only available to admins.</p>
        </div>
        `;
  }

  html = `
        <div class="tab-results">
            <div class="results-container"></div>
        </div>
        ${permissionsText}
        <div class="settings-section user-profile"></div>
        <div class="settings-section selectors-edit"></div>
        <div class="settings-section user-list"></div>
        <div class="settings-section user-creation"></div>
    `;
  doc = new DOMParser().parseFromString(html, 'text/html');
  document.querySelector('.tab-results')
      .replaceWith(doc.querySelector('.tab-results'));
  // Append the sections to the tab content, after tab-results?
  // actually tab-results is replaced. The placeholders need to be appended to .tab-content.settings

  const settingsContent = document.querySelector('.tab-content.settings');
  settingsContent.appendChild(doc.querySelector('.settings-section.user-profile'));
  settingsContent.appendChild(doc.querySelector('.settings-section.selectors-edit'));
  settingsContent.appendChild(doc.querySelector('.settings-section.user-list'));
  settingsContent.appendChild(doc.querySelector('.settings-section.user-creation'));

  renderUserSettings(context, {'selector': '.settings-section.user-profile'});

  // Render the selectors edit area if the user is an admin.
  if (userRoleValue >= 2) {
    // Render the selectors edit area.
    // Get the "utility individual" to see the labels stored to it.
    SharkMatcher.Individual.get_individuals(-1, null, null, true, true)
        .then(function(result) {
          renderSelectorsEdit(
              result, context, {'selector': '.settings-section.selectors-edit'});

          Home.renderObjectList(
              result, 'label', context,
              {'selector': '.label .object-list', 'page': 'settings'});
          Home.renderObjectList(
              result, 'alternate_name', context,
              {'selector': '.location .object-list', 'page': 'settings'});
          Home.renderObjectList(
              result, 'tag', context,
              {'selector': '.tag .object-list', 'page': 'settings'});
        })
        .catch(function(error) {
          console.log('Error with utility individual search.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error with utility ${Util.getDisplayName().toLowerCase()} search: ${message}`);
          }
        });

    // Render user creation independently of the utility individual fetch.
    renderAdminUserCreation(context, {'selector': '.settings-section.user-creation'});
    // Hide by default; revealed by the "Create User" toggle button.
    const userCreationSection = document.querySelector('.settings-section.user-creation');
    if (userCreationSection) {
      userCreationSection.style.display = 'none';
    }
  }

  // Render user list if user is admin or higher.
  if (userRoleValue >= 2) {
    SharkMatcher.User.filter_users(null, null, null)
        .then(function(users) {
          renderUserList(context, users, {'selector': '.settings-section.user-list'});
        })
        .catch(function(error) {
          console.log('Error fetching users.');
          // Non-critical, just log.
        });
  }
}

// Path update functions.

// API execution functions.
async function deriveSettingsPassword(
    passwordSelector, verifySelector, required = false) {
  const password = document.querySelector(passwordSelector).value;
  const verification = document.querySelector(verifySelector).value;

  if (password.length === 0 && verification.length === 0) {
    if (required) {
      alert('Password is required.');
      return undefined;
    }
    return null;
  }

  if (password.length < 2) {
    alert('Password must be at least 2 characters.');
    return undefined;
  }
  if (password !== verification) {
    alert('Passwords must match.');
    return undefined;
  }

  return Util.sha256(password);
}

async function executeUserSave() {
  const nameInput = document.querySelector('.settings.name-input').value.trim();

  if (nameInput.length === 0) {
    alert('Name cannot be empty.');
    return;
  }

  try {
    const encoded = await deriveSettingsPassword(
        '.settings.password-input', '.settings.verify-input');
    if (encoded === undefined) {
      return;
    }

    const currentUser = Core.getContextUser();

    if (nameInput === currentUser.user_display_name && encoded === null) {
      return; // No changes to save
    }

    await SharkMatcher.User.post_user(
        currentUser.user_id, nameInput, currentUser.user_email,
        currentUser.user_username, currentUser.user_role, encoded);
    if (encoded !== null) {
      document.querySelector('.settings.password-input').value = '';
      document.querySelector('.settings.verify-input').value = '';
    }
    alert('Settings updated. Name displays update after login.');
  } catch (error) {
    console.log('Error updating settings.');
    const message = Util.handleAPIError(error);
    if (message) {
      alert(`Error updating settings: ${message}`);
    }
  }
}

// HTML render functions.
// Render the menu to edit a user.
function renderUserSettings(context, kwargs = {}) {
  const currentUser = Core.getContextUser();

  const html = `
    <div class="settings user">
      <div class="settings-user-pill">
        <div class="user profile">
            <h4>${Util.escapeHtml(currentUser.user_display_name)}</h4>
            <div class="info-row"><span class="label">Username:</span> <span class="value">${Util.escapeHtml(currentUser.user_username)}</span></div>
            <div class="info-row"><span class="label">Email:</span> <span class="value">${Util.escapeHtml(currentUser.user_email)}</span></div>
            <div class="info-row"><span class="label">Role:</span> <span class="value">${currentUser.user_role}</span></div>
            <div class="info-row">
              <span class="label" style="line-height: 24px;">Dark Theme</span> 
              <span class="value" style="display: flex; align-items: center;">
                <label class="switch">
                  <input type="checkbox" id="theme-toggle">
                  <span class="slider round"></span>
                </label>
              </span>
            </div>
        </div>

        <div class="settings-btn-row">
            <button class="user properties-button pill-btn">User Settings</button>
        </div>
        <div class="user-properties-area user properties" style="display: none">
            <form class="input-section settings" autocomplete="off">
                <div class="input-item">
                    <label for="settings-name-input">Name:</label>
                    <input
                        class="settings name-input"
                        id="settings-name-input"
                        type="text"
                        value="${Util.escapeHtml(currentUser.user_display_name)}"
                        placeholder="Name"/>
                </div>
                <div class="input-item" style="margin-top: 15px;">
                    <label for="settings-password-input">New Password:</label>
                    <input
                        class="settings password-input"
                        id="settings-password-input"
                        type="password"
                        autocomplete="new-password"
                        placeholder="new password"/>
                    </div>
                <div>
                <label for="settings-verify-input">Verify New Password:</label>
                <input
                    class="settings verify-input"
                    id="settings-verify-input"
                    type="password"
                    autocomplete="new-password"
                    placeholder="new password"/>
                </div>
                <button type="button" class="user properties save pill-btn">Update Settings</button>
            </form>
        </div>
      </div>
    </div>
    `;
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const containerNode = doc.querySelector('.settings.user');
  const propertiesButton = doc.querySelector('.user.properties-button');
  propertiesButton.addEventListener('click', function(event) {
    const isVisible = containerNode.querySelector('.user-properties-area').style.display !== 'none';
    Util.toggleVisibility(
        '', false, false, {'container': containerNode, 'selector': '.user-properties-area'});
    if (!isVisible) {
      event.target.classList.add('active');
    } else {
      event.target.classList.remove('active');
    }
  });
  const userSaveButton = doc.querySelector('.user.properties.save');
  userSaveButton.addEventListener('click', function() {
    executeUserSave();
  });

  // Handle theme toggling.
  const themeToggle = doc.querySelector('#theme-toggle');
  // Check localStorage for user preference
  const currentTheme = SharkMatcher.getTheme();
  if (currentTheme === 'dark') {
    themeToggle.checked = true;
  }

  themeToggle.addEventListener('change', function(e) {
    if (e.target.checked) {
      document.documentElement.dataset.theme = 'dark';
      SharkMatcher.setTheme('dark');
    } else {
      document.documentElement.removeAttribute('data-theme'); // default is light
      SharkMatcher.setTheme('light');
    }
  });

  document.querySelector(kwargs['selector'])
      .appendChild(doc.querySelector('.settings.user'));
}

// Render the menu to edit selector preferences.
// Location options are stored as alternate names so the selector is called
// "alternate_name".
function renderSelectorsEdit(individuals, context, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }

  const html = `
    <div>
    <h4>Edit Selector Options</h4>
    <div class="settings-btn-row">
        <button class="label-button pill-btn">Labels</button>
        <button class="location-button pill-btn">Locations</button>
        <button class="tag-button pill-btn">Tag Types</button>
    </div>

    <div class="objects-area label" style="display: none">
        <h4>Labels:</h4>
        <div class="object-list"></div>
        <div class="input-section" style="display: flex; gap: 10px; align-items: flex-end; justify-content: flex-start;">
            <div class="input-item" style="margin-bottom: 0;">
                <label for="label-name-input">Label: </label>
                <input
                    class="label name-input"
                    id="label-name-input"
                    type="text"
                    placeholder="label"/>
            </div>
            <button class="label save pill-btn">Add Label</button>
        </div>
    </div>

    <div class="objects-area location" style="display: none">
        <h4>Locations:</h4>
        <div class="object-list"></div>
        <div class="input-section" style="display: flex; gap: 10px; align-items: flex-end; justify-content: flex-start;">
            <div class="input-item" style="margin-bottom: 0;">
                <label for="alternate_name-name-input">Location: </label>
                <input
                    class="alternate_name name-input"
                    id="alternate_name-name-input"
                    type="text"
                    placeholder="location"/>
            </div>
            <button class="location save pill-btn">Add Location</button>
        </div>
    </div>

    <div class="objects-area tag" style="display: none">
        <h4>Tag Types:</h4>
        <div class="object-list"></div>
        <div class="input-section" style="display: flex; gap: 10px; align-items: flex-end; justify-content: flex-start;">
            <div class="input-item" style="margin-bottom: 0;">
                <label for="tag-name-input">Tag Type: </label>
                <input
                    class="tag name-input"
                    id="tag-name-input"
                    type="text"
                    placeholder="tag type"/>
            </div>
            <button class="tag save pill-btn">Add Tag Type</button>
        </div>
    </div>
    </div>
    `;
  const doc = new DOMParser().parseFromString(html, 'text/html');

  // Menu toggle buttons.
  const labelButton = doc.querySelector('.label-button');
  const locationButton = doc.querySelector('.location-button');
  const tagButton = doc.querySelector('.tag-button');

  const toggleSelector = function(event, targetSelector) {
    const parentContainer = event.target.parentElement.parentElement.parentElement;
    const allButtons = [labelButton, locationButton, tagButton];
    const targetButton = event.target;

    // Determine current visibility state.
    const targetArea = parentContainer.querySelector(targetSelector);
    const isVisible = targetArea.style.display !== 'none';

    // Reset everything.
    Util.toggleVisibility('None', false, true, {'container': parentContainer, 'selector': '.objects-area.label'});
    Util.toggleVisibility('None', false, true, {'container': parentContainer, 'selector': '.objects-area.tag'});
    Util.toggleVisibility('None', false, true, {'container': parentContainer, 'selector': '.objects-area.location'});
    allButtons.forEach((btn) => btn.classList.remove('active'));

    // If it was hidden, show it and set active.
    if (!isVisible) {
      Util.toggleVisibility('', false, false, {'container': parentContainer, 'selector': targetSelector});
      targetButton.classList.add('active');
    }
  };

  labelButton.addEventListener('click', function(event) {
    toggleSelector(event, '.objects-area.label');
  });

  locationButton.addEventListener('click', function(event) {
    toggleSelector(event, '.objects-area.location');
  });

  tagButton.addEventListener('click', function(event) {
    toggleSelector(event, '.objects-area.tag');
  });

  // Save buttons.
  const labelSaveButton = doc.querySelector('.label.save');
  labelSaveButton.addEventListener('click', function(event) {
    Home.executeObjectSave('.label', individuals[0], context, {'page': 'settings'});
  });
  // Location options are stored as alternate names in the utility individual.
  const locationSaveButton = doc.querySelector('.location.save');
  locationSaveButton.addEventListener('click', function(event) {
    Home.executeObjectSave(
        '.alternate_name', individuals[0], context, {'page': 'settings'});
  });
  const tagSaveButton = doc.querySelector('.tag.save');
  tagSaveButton.addEventListener('click', function(event) {
    Home.executeObjectSave('.tag', individuals[0], context, {'page': 'settings'});
  });

  container.querySelector(kwargs['selector'])
      .appendChild(doc.querySelector('div'));
  return container;
}

// Render the menu to create a new user (Admin only).
function renderAdminUserCreation(context, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }

  const html = `
    <div>
    <h4>Create New User</h4>
    <form class="input-section settings" autocomplete="off">
        <div class="input-item">
            <label for="create-user-name-input">Name: </label>
            <input
                class="create-user name-input"
                id="create-user-name-input"
                type="text"
                placeholder="display name"/>
        </div>
        <div class="input-item">
            <label for="create-user-email-input">Email: </label>
            <input
                class="create-user email-input"
                id="create-user-email-input"
                type="text"
                placeholder="email"/>
        </div>
        <div class="input-item">
            <label for="create-user-username-input">Username: </label>
            <input
                class="create-user username-input"
                id="create-user-username-input"
                type="text"
                placeholder="username"/>
        </div>
        <div class="input-item">
            <label for="create-user-password-input">Password: </label>
            <input
                class="create-user password-input"
                id="create-user-password-input"
                type="password"
                autocomplete="new-password"
                placeholder="password"/>
        </div>
        <div class="input-item">
            <label for="create-user-verify-input">Verify Password: </label>
            <input
                class="create-user verify-input"
                id="create-user-verify-input"
                type="password"
                autocomplete="new-password"
                placeholder="password"/>
        </div>
         <div class="input-item">
            <label for="create-user-role-select">Role: </label>
            <select class="create-user role-select" id="create-user-role-select">
                <option value="viewer">Viewer</option>
                <option value="collaborator">Collaborator</option>
                <option value="admin">Admin</option>
            </select>
        </div>
        <button type="button" class="create-user save pill-btn">Create User</button>
    </form>
    </div>
    `;
  const doc = new DOMParser().parseFromString(html, 'text/html');

  const saveButton = doc.querySelector('.create-user.save');
  saveButton.addEventListener('click', function() {
    executeCreateUser();
  });

  container.querySelector(kwargs['selector'])
  // Append after existing content (e.g. selectors edit area).
      .appendChild(doc.querySelector('div'));
}

async function executeCreateUser() {
  const name = document.querySelector('.create-user.name-input').value;
  const email = document.querySelector('.create-user.email-input').value;
  const username = document.querySelector('.create-user.username-input').value;
  const role = document.querySelector('.create-user.role-select').value;

  if (!name || !email || !username || !role) {
    alert('Please fill in all fields.');
    return;
  }

  try {
    const encoded = await deriveSettingsPassword(
        '.create-user.password-input', '.create-user.verify-input', true);
    if (encoded === undefined) {
      return;
    }

    // Confirm creation.
    if (!confirm(`Create user ${username} with role ${role}?`)) {
      return;
    }

    await SharkMatcher.User.post_user(
        null, name, email, username, role, encoded);
    alert('User created successfully.');
    // Clear inputs
    document.querySelector('.create-user.name-input').value = '';
    document.querySelector('.create-user.email-input').value = '';
    document.querySelector('.create-user.username-input').value = '';
    document.querySelector('.create-user.password-input').value = '';
    document.querySelector('.create-user.verify-input').value = '';
    document.querySelector('.create-user.role-select').value = 'viewer';
  } catch (error) {
    console.log('Error creating user.');
    const message = Util.handleAPIError(error);
    if (message) {
      alert(`Error creating user: ${message}`);
    }
  }
}

function renderUserList(context, users, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }

  const html = `
    <div>
    <h4>Users</h4>
    <div class="settings-btn-row">
        <button class="user-list-button pill-btn">All Users</button>
        <button class="user-creation-toggle-btn pill-btn">Create User</button>
    </div>
    <div class="user-list-wrapper" style="display: none; overflow-x: auto; padding: 10px 0;">
        <div class="user-list" style="display: flex; gap: 20px; white-space: normal; flex-wrap: nowrap; width: fit-content; margin: 0 auto; min-height: 100px;"></div>
    </div>
    </div>
    `;
  const doc = new DOMParser().parseFromString(html, 'text/html');

  const userListButton = doc.querySelector('.user-list-button');
  const creationToggleBtn = doc.querySelector('.user-creation-toggle-btn');

  const toggleUserSection = function(event, type) {
    const parentContainer = event.target.closest('.settings-section.user-list');
    if (!parentContainer) return;

    const allButtons = [userListButton, creationToggleBtn];
    const targetButton = event.target;

    const wrapper = parentContainer.querySelector('.user-list-wrapper');
    const creationSection = document.querySelector('.settings-section.user-creation');

    if (type === 'list') {
      if (!wrapper) return;
      const isVisible = wrapper.style.display !== 'none';
      // Reset.
      wrapper.style.display = 'none';
      if (creationSection) creationSection.style.display = 'none';
      allButtons.forEach((btn) => btn.classList.remove('active'));

      if (!isVisible) {
        wrapper.style.display = 'block';
        targetButton.classList.add('active');
      }
    } else {
      const isVisible = creationSection && creationSection.style.display !== 'none';
      // Reset.
      wrapper.style.display = 'none';
      if (creationSection) creationSection.style.display = 'none';
      allButtons.forEach((btn) => btn.classList.remove('active'));

      if (!isVisible && creationSection) {
        creationSection.style.display = 'block';
        targetButton.classList.add('active');
      }
    }
  };

  userListButton.addEventListener('click', function(event) {
    toggleUserSection(event, 'list');
  });

  // Toggle the create-user form section.
  creationToggleBtn.addEventListener('click', function(event) {
    toggleUserSection(event, 'create');
  });

  // Instead of Home.renderObjectList, render a profile card for each user.
  for (const user of users) {
    const userCardHtml = `
      <div class="results-item" object-id="${user.id}" style="min-width: 250px; flex-shrink: 0; display: inline-block;">
          <div class="user-profile"></div>
      </div>
    `;
    const cardDoc = new DOMParser().parseFromString(userCardHtml, 'text/html');
    renderUserProfile([user], context, {'container': cardDoc, 'selector': '.user-profile'});
    doc.querySelector('.user-list').appendChild(cardDoc.querySelector('.results-item'));
  }

  container.querySelector(kwargs['selector'])
      .appendChild(doc.querySelector('div'));
}

function renderUserProfile(users, context, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }

  const user = users[0];

  const profileString = `
    <div class="card-id-label">
        <span class="card-title-pill">User: ${Util.escapeHtml(user.username)}</span>
    </div>`;

  let cardActions = '';
  // Disable delete button when in viewing mode
  const currentMode = Core.getCurrentUserMode();
  let deleteDisabled = '';
  if (currentMode === 'viewing') {
    deleteDisabled = 'disabled';
  }

  cardActions = `
    <div class="card-actions" style="display: flex; justify-content: center; margin-top: 15px;">
        <button class="object delete pill-btn" object-id="${user.id}" ${deleteDisabled}>
            Delete
        </button>
    </div>`;

  const html = `
  <div>
      ${profileString}

      <div class="info-row">
          <span class="label">Name:</span>
          <span class="value">${Util.displayValue(user.name)}</span>
      </div>
      <div class="info-row">
          <span class="label">Username:</span>
          <span class="value">${Util.displayValue(user.username)}</span>
      </div>
      <div class="info-row">
          <span class="label">Role:</span>
          <span class="value" style="text-transform: capitalize;">${Util.displayValue(user.role)}</span>
      </div>

      ${cardActions}
  </div>
  `;
  const doc = new DOMParser().parseFromString(html, 'text/html');

  const deleteButton = doc.querySelector('.object.delete');
  if (deleteButton) {
    deleteButton.addEventListener('click', function(event) {
      if (!confirm(`Are you sure you want to delete user ${user.username}? This cannot be undone.`)) {
        return;
      }
      SharkMatcher.User.delete_user(user.id)
          .then(function(result) {
            alert('User deleted.');
            const item = event.target.closest('.results-item');
            if (item) {
              item.remove();
            }
          })
          .catch(function(error) {
            console.log('Error deleting user.');
            const message = Util.handleAPIError(error);
            if (message) {
              alert(`Error deleting user: ${message}`);
            }
          });
    });
  }

  container.querySelector(kwargs['selector'])
      .appendChild(doc.querySelector('div'));
  return container;
}

export {
  init,
};
