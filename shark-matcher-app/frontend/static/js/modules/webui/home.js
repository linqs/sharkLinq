import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as History from './history.js';
import * as Login from './login.js';
import * as Match from './match.js';
import * as Merge from './merge.js';
import * as Proposal from './proposal.js';
import * as Review from './review.js';
import * as Routes from './routes.js';
import * as Search from './search.js';
import * as Individual from './individual.js';
import * as Util from './util.js';

const ASSOCIATED_OBJECT_MAP = {
  'alternate_name': 'alternate_names',
  'tag': 'tags',
  'label': 'labels',
};

// The source (reference) individual currently loaded on the individual profile page.
// Set when the gallery renders so that renderMediaProfile can access it for
// constraint coloring without needing to change function signatures across the call chain.
let profileSourceIndividual = null;

function init() {
  Routes.addRoute(/^(home)?$/, handlerHome);

  // Apply high contrast mode if previously enabled
  if (localStorage.getItem('highContrastMode') === 'enabled') {
    document.body.classList.add('high-contrast-mode');
  }
}
function buttonLink(link) {
  window.location.href = link;
}

function disableElement(element, message) {
  element.setAttribute('aria-disabled', 'true');
  if (message) {
    element.title = message;
  }
  element.addEventListener('click', (event) => {
    if (element.getAttribute('aria-disabled') === 'true') {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, {capture: true});
}


function handlerHome(path, params, context) {
  // Fetch label options and add to local storage. This is done after login,
  // it may expire so logging out and logging back in will get the latest
  // labels.
  Util.refreshCache('selectors', context);
  Util.refreshCache('users', context);
  Util.refreshCache('dataset_stats', context);

  // Page setup.
  let html = `
    <div class="header-content">
        <div class="input-section">
            <!-- Tabs -->
            <div class="tab-group">
                <div class="tab-links search" role="button" tabindex="0">Search</div>
                <div class="tab-links match" role="button" tabindex="0">Match</div>
                <!-- REVIEW TAB HIDDEN: tab is incomplete and not ready for use.
                     To restore, uncomment this line and the reviewButton wiring below.
                <div class="tab-links review" role="button" tabindex="0">Review</div>
                -->
                <div class="tab-links upload" role="button" tabindex="0">Upload</div>
                <div class="tab-links proposals" role="button" tabindex="0">Proposals</div>
                <div class="tab-links history" role="button" tabindex="0">History</div>
                <div class="tab-links settings" role="button" tabindex="0">Settings</div>
            </div>

            <div class="menu-section"></div>
            <button class="logout">Logout</button>
        </div>

        <!-- Tab content. -->
        <div class="tab-content"></div>
    </div>
    `;
  let doc = document.createElement('div');
  doc.innerHTML = html;

  // Ensure popup container exists on body (for full screen overlay)
  if (!document.querySelector('.popup-container')) {
    const popupContainer = document.createElement('div');
    popupContainer.className = 'popup-container';
    document.body.appendChild(popupContainer);
  }

  // --- High Contrast Toggle (Moved to Body) ---
  const contrastHtml = `
        <div class="contrast-toggle-container">
            <span>Contrast</span>
            <label class="switch">
                <input type="checkbox" class="contrast-toggle">
                <span class="slider round"></span>
            </label>
        </div>
    `;
  const contrastDoc = document.createElement('div');
  contrastDoc.innerHTML = contrastHtml;
  const newContrastContainer = contrastDoc.querySelector('.contrast-toggle-container');
  const existingContrastContainer = document.querySelector('.contrast-toggle-container');

  if (existingContrastContainer) {
    existingContrastContainer.replaceWith(newContrastContainer);
  } else {
    document.body.appendChild(newContrastContainer);
  }

  // High contrast toggle functionality
  const contrastToggle = newContrastContainer.querySelector('.contrast-toggle');
  if (localStorage.getItem('highContrastMode') === 'enabled') {
    contrastToggle.checked = true;
    // Body class should already be set by init(), but ensure it if not
    if (!document.body.classList.contains('high-contrast-mode')) {
      document.body.classList.add('high-contrast-mode');
    }
  }
  contrastToggle.addEventListener('change', function() {
    document.body.classList.toggle('high-contrast-mode');
    localStorage.setItem(
        'highContrastMode',
      document.body.classList.contains('high-contrast-mode') ? 'enabled' :
        'disabled',
    );
  });

  // --- Mode Selector (Moved to Body) ---
  const modeHtml = `
        <div class="mode-selector-container">
            <span>Mode</span>
            <select class="mode-selector">
                <option value="viewing">Viewing</option>
            </select>
        </div>
    `;
  const modeDoc = document.createElement('div');
  modeDoc.innerHTML = modeHtml;
  const newModeContainer = modeDoc.querySelector('.mode-selector-container');
  const existingModeContainer = document.querySelector('.mode-selector-container');

  if (existingModeContainer) {
    existingModeContainer.replaceWith(newModeContainer);
  } else {
    document.body.appendChild(newModeContainer);
  }

  // Mode selector functionality
  // Dynamically populate mode selector based on user role
  const currentUser = Core.getContextUser();
  const modeSelector = newModeContainer.querySelector('.mode-selector');

  if (currentUser && currentUser.user_role) {
    const userRoleValue = Core.roleDict[currentUser.user_role];

    // Add proposing mode for users with role >= 1
    // (collaborator, admin, dev, owner)
    if (userRoleValue >= 1) {
      const proposingOption = document.createElement('option');
      proposingOption.value = 'proposing';
      proposingOption.textContent = 'Proposing';
      modeSelector.appendChild(proposingOption);
    }
    // Add editing mode for users with role >= 2 (admin, dev, owner)
    if (userRoleValue >= 2) {
      const editingOption = document.createElement('option');
      editingOption.value = 'editing';
      editingOption.textContent = 'Editing';
      modeSelector.appendChild(editingOption);
    }
  }

  if (window.sharkMatcher?.config?.demo_enabled) {
    const option = document.createElement('option');
    option.value = 'demo';
    option.textContent = 'Demo';
    modeSelector.appendChild(option);
  }

  let savedMode = window.sharkMatcher.config.userMode ?? localStorage.getItem('userMode');

  // Check user permissions for different modes
  let canAccessProposing = false;
  let canAccessEditing = false;
  if (currentUser && currentUser.user_role) {
    const userRoleValue = Core.roleDict[currentUser.user_role];
    canAccessProposing = userRoleValue >= 1; // collaborator, admin, dev, owner
    canAccessEditing = userRoleValue >= 2; // admin, dev, owner
  }

  // Determine the mode to use
  let currentMode;

  // Only reset mode if saved mode is invalid for current user
  if (savedMode === 'proposing' && !canAccessProposing) {
    // User can't access proposing mode, reset to highest available
    savedMode = null;
  } else if (savedMode === 'editing' && !canAccessEditing) {
    // User can't access editing mode, reset to highest available
    savedMode = null;
  }

  // If no valid saved mode, set to highest available mode
  if (!savedMode) {
    let highestMode = 'viewing'; // Default fallback
    if (canAccessEditing) {
      highestMode = 'editing';
    } else if (canAccessProposing) {
      highestMode = 'proposing';
    }
    localStorage.setItem('userMode', highestMode);
    currentMode = highestMode;
  } else {
    // Use the valid saved mode
    currentMode = savedMode;
  }

  window.sharkMatcher.config.userMode = currentMode;

  // Apply the mode CSS classes
  if (currentMode === 'proposing') {
    document.body.classList.add('proposing-mode');
    document.body.classList.remove('viewing-mode');
  } else if (Core.isReadOnlyMode()) {
    document.body.classList.add('viewing-mode');
    document.body.classList.remove('proposing-mode');
  } else {
    // In editing mode.
    document.body.classList.remove('proposing-mode', 'viewing-mode');
  }

  // Set mode selector value and add event listener
  modeSelector.value = currentMode;

  // Mode classes are already applied above, no need to apply again
  modeSelector.addEventListener('change', function(event) {
    const selectedMode = event.target.value;
    localStorage.setItem('userMode', selectedMode);

    // Update body classes
    if (selectedMode === 'proposing') {
      document.body.classList.add('proposing-mode');
      document.body.classList.remove('viewing-mode');
    } else if (selectedMode === 'viewing') {
      document.body.classList.add('viewing-mode');
      document.body.classList.remove('proposing-mode');
    } else {
      // editing mode
      document.body.classList.remove('proposing-mode', 'viewing-mode');
    }

    // Crossing datasets starts at Home, avoiding a profile from the other dataset.
    if (selectedMode === 'demo' || currentMode === 'demo') window.location.hash = '';
    // Reload the page to apply the new mode
    location.reload(true);
  });

  // Links for tab buttons.
  const searchButton = doc.querySelector('.tab-links.search');
  const matchButton = doc.querySelector('.tab-links.match');
  const uploadButton = doc.querySelector('.tab-links.upload');
  // REVIEW TAB HIDDEN: tab is incomplete and not ready for use.
  // To restore: uncomment the next line and the reviewButton.addEventListener block below.
  // const reviewButton = doc.querySelector('.tab-links.review');
  const proposalsButton = doc.querySelector('.tab-links.proposals');
  const historyButton = doc.querySelector('.tab-links.history');
  const settingsButton = doc.querySelector('.tab-links.settings');
  const logoutButton = doc.querySelector('.logout');
  if (currentMode === 'demo') {
    for (const tab of [uploadButton, proposalsButton, historyButton, settingsButton]) {
      disableElement(tab, 'Unavailable in Demo mode.');
    }
  }
  for (const tab of doc.querySelectorAll('.tab-links')) {
    tab.addEventListener('keydown', function(event) {
      if (event.key === 'Enter' || event.key === ' ') {
        // The document-level Enter handler would otherwise run a search on the page being opened.
        event.preventDefault();
        event.stopPropagation();
        tab.click();
      }
    });
  }
  searchButton.addEventListener('click', function() {
    buttonLink(Core.formHashPath('search'));
  });
  matchButton.addEventListener('click', function() {
    buttonLink(Core.formHashPath('match'));
  });
  uploadButton.addEventListener('click', function() {
    buttonLink(Core.formHashPath('upload'));
  });
  // REVIEW TAB HIDDEN: restore by uncommenting the block below.
  // reviewButton.addEventListener('click', function () {
  //   buttonLink(Core.formHashPath('review'));
  // });
  proposalsButton.addEventListener('click', function() {
    buttonLink(Core.formHashPath('proposals'));
  });
  historyButton.addEventListener('click', function() {
    buttonLink(Core.formHashPath('history'));
  });
  settingsButton.addEventListener('click', function() {
    buttonLink(Core.formHashPath('settings'));
  });
  logoutButton.addEventListener('click', function() {
    Login.executeLogout(context);
  });

  document.querySelector('.header-content')
      .replaceWith(doc.querySelector('.header-content'));

  // Results content.
  html = `
    <div class="content">
        <div class="tab-results"></div>
    </div>
    `;
  doc = document.createElement('div');
  doc.innerHTML = html;
  document.querySelector('.content').replaceWith(doc.querySelector('.content'));

  // Populate welcome.
  if (path === '' || path === 'home') {
    executeDatasetSearch(context);
  }
}

// API execution functions.
function executeObjectSave(inputSelector, individual, context, kwargs = {}) {
  // Extract the table name from the inputSelector.
  let tableName = inputSelector.replaceAll('media', '');
  tableName = tableName.replaceAll(' ', '');
  tableName = tableName.replaceAll('.', '');

  // Get and validate inputs.
  const nameInput = document.querySelector(`${inputSelector}.name-input`).value;
  let typeInput = null;
  let timeInput = null;
  try {
    if (Util.typedObjects.includes(tableName)) {
      typeInput = document.querySelector(`${inputSelector}.type-input`).value;
    }
    if (Util.timedObjects.includes(tableName)) {
      timeInput = Util.inputToTimestamp(
          document.querySelector(`${inputSelector}.time-input`).value, false);
    }
  } catch {
    console.log('No type or time input found.');
  }


  if (nameInput == '') {
    alert('Please provide a valid input.');
    return;
  } else if (Util.validName(nameInput) === false) {
    alert('Name input contains disallowed characters.');
    return;
  }

  // Check if in proposing mode or editing mode.
  const currentMode = Core.getCurrentUserMode();
  if (currentMode === 'proposing' && kwargs['page'] !== 'settings') {
    // Build the updated object, and post as a proposal.
    const savedObject = {
      'id': null,
      'individual': individual.id,
      'name': nameInput,
      'type': typeInput,
      'time': timeInput,
    };
    SharkMatcher.Proposal
        .post_proposal(
            null, null, savedObject, 'create', tableName, 'open')
        .then(function(result) {
          alert('Successfully saved proposal.');
        })
        .catch(function(error) {
          console.log('Error saving proposal.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error saving proposal: ${message}`);
          }
        });
  } else if (Core.canEdit() || kwargs['page'] === 'settings') {
    // Update database and log activity.
    SharkMatcher.Individual
        .post_object(
            null, individual.id, nameInput, typeInput, timeInput, tableName)
        .then(function(result) {
          // If the settings page was the caller, update the selector options
          // cached in the browser.
          if (kwargs['page'] === 'settings') {
            Util.refreshCache('selectors', context, true).then(() => {
              location.reload(true);
            });
          } else {
            location.reload(true);
          }
        })
        .catch(function(error) {
          console.log('Error updating individual object.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error updating ${Util.getDisplayName().toLowerCase()} object: ${message}`);
          }
        });
  }
}

function executeObjectDelete(event, individual, tableName, context, kwargs = {}) {
  const objectId = parseInt(event.target.getAttribute('object-id'));

  // Check if in proposing mode or editing mode.
  const currentMode = Core.getCurrentUserMode();
  if (currentMode === 'proposing' && kwargs['page'] !== 'settings') {
    // Find the deleted object and remove it from the individual.
    let deletedObject;
    for (let i = 0; i < individual[ASSOCIATED_OBJECT_MAP[tableName]].length; i++) {
      if (individual[ASSOCIATED_OBJECT_MAP[tableName]][i].id === objectId) {
        deletedObject = individual[ASSOCIATED_OBJECT_MAP[tableName]][i];
        break;
      }
    }

    SharkMatcher.Proposal
        .post_proposal(
            null, deletedObject, null, 'delete', tableName, 'open')
        .then(function(result) {
          alert('Successfully saved proposal.');
        })
        .catch(function(error) {
          console.log('Error saving proposal.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error saving proposal: ${message}`);
          }
        });
  } else if (Core.canEdit() || kwargs['page'] === 'settings') {
    // Deletes require confimation.
    const confirmation =
      confirm('Are you sure you want to delete this object?');
    if (confirmation === false) {
      return;
    }

    // Update database and log activity.
    SharkMatcher.Individual.delete_object(objectId, tableName)
        .then(function(result) {
        // Update page.
          if (kwargs['page'] === 'settings') {
            Util.refreshCache('selectors', context, true).then(() => {
              location.reload(true);
            });
          } else {
            location.reload(true);
          }
        })
        .catch(function(error) {
          console.log('Error deleting individual object.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error deleting ${Util.getDisplayName().toLowerCase()} object: ${message}`);
          }
        });
  }
}

function executeMediaSearch(
    path, params, context, renderFunction, kwargs = {}) {
  SharkMatcher.Media.get_medias(params.id, true)
      .then(function(result) {
      // Update page.
        renderFunction(result, context, kwargs);
      })
      .catch(function(error) {
        console.log('Error executing search.');
        const message = Util.handleAPIError(error);
        if (message) {
          alert(`Error executing search: ${message}`);
        }
      });
}

function executeDatasetSearch(context) {
  Util.loadCache('dataset_stats')
      .then(function(result) {
        renderHome(result, context, {});
      })
      .catch(function(error) {
        console.log('Error executing search.');
        const message = Util.handleAPIError(error);
        if (message) {
          alert(`Error executing search: ${message}`);
        }
      });
}

let embeddingsPollInterval = null;

function startEmbeddingsPolling() {
  if (embeddingsPollInterval !== null) {
    return;
  }
  embeddingsPollInterval = setInterval(pollEmbeddingsStatus, 3000);
}

function stopEmbeddingsPolling() {
  if (embeddingsPollInterval !== null) {
    clearInterval(embeddingsPollInterval);
    embeddingsPollInterval = null;
  }
}

function pollEmbeddingsStatus() {
  SharkMatcher.get_embeddings_status()
      .then(function(status) {
        if (!status) return;
        const embeddingsButton = document.querySelector('.embeddings-button');
        if (status.status === 'in_progress') {
          if (embeddingsButton) {
            disableElement(
                embeddingsButton,
                status.progress || 'An embedding update is currently in progress.');
            embeddingsButton.textContent = 'Updating Embeddings...';
          }
        } else if (status.status === 'completed') {
          stopEmbeddingsPolling();
          Util.clearCache('dataset_stats');
          alert('Embeddings update complete.');
          location.reload(true);
        } else if (status.status === 'failed') {
          stopEmbeddingsPolling();
          const err = status.error || 'Unknown error';
          alert(`Embedding update failed: ${err}`);
          location.reload(true);
        } else {
          stopEmbeddingsPolling();
        }
      })
      .catch(function(err) {
        console.warn('Error polling embeddings status:', err);
      });
}

function executeUpdateEmbeddings() {
  const embeddingsButton = document.querySelector('.embeddings-button');
  if (Core.getCurrentUserMode() === 'demo' ||
      embeddingsButton?.getAttribute('aria-disabled') === 'true') {
    return;
  }

  const confirmation = confirm(
      'Updating embeddings will take several minutes. Are you sure you want to update embeddings?');
  if (confirmation === false) {
    return;
  }

  if (embeddingsButton) {
    disableElement(embeddingsButton, 'Embeddings update in progress.');
    embeddingsButton.textContent = 'Updating Embeddings...';
  }

  SharkMatcher.update_embeddings()
      .then(function() {
        startEmbeddingsPolling();
      })
      .catch(function(error) {
        console.log('Error updating embeddings.');
        const message = Util.handleAPIError(error);
        if (message && message.includes('already in progress')) {
          alert('An embedding update is already in progress.');
          if (embeddingsButton) {
            disableElement(embeddingsButton, 'An embedding update is currently in progress.');
            embeddingsButton.textContent = 'Updating Embeddings...';
          }
          startEmbeddingsPolling();
        } else {
          if (message) {
            alert(`Error updating embeddings: ${message}`);
          }
          if (embeddingsButton) {
            if (Core.getCurrentUserMode() === 'demo') {
              disableElement(embeddingsButton, 'Embedding updates are unavailable in Demo mode.');
            } else {
              embeddingsButton.removeAttribute('aria-disabled');
              embeddingsButton.removeAttribute('title');
            }
            embeddingsButton.textContent = 'Update Embeddings';
          }
        }
      });
}

function executeMediaSave(medias, kwargs = {}) {
  const index = kwargs['index'];
  const media = medias[0];

  // Get and validate inputs. TODO, allow like"ish" inputs, validate inputs.
  const nameInput = document.getElementById(`media-name-input-${index}`).value;
  const locationInput =
    document.getElementById(`media-location-input-${index}`).value;
  const timeInput = document.getElementById(`media-time-input-${index}`).value;
  const sexInput = document.getElementById(`media-sex-input-${index}`).value;
  const lengthInput =
    document.getElementById(`media-length-input-${index}`).value;
  const qualityInput =
    document.getElementById(`media-min-quality-input-${index}`).value;
  const typeInput = document.getElementById(`media-type-input-${index}`).value;

  // Check if in proposing mode or editing mode.
  const currentMode = Core.getCurrentUserMode();
  if (currentMode === 'proposing') {
    // Represent the media fully, including the individual.
    media.sex = media.individual.sex;
    media.length = media.individual.length;

    // Make a copy of the media and apply the changes.
    const updatedMedia = structuredClone(media);
    updatedMedia.name = nameInput;
    updatedMedia.location = locationInput;
    updatedMedia.time = Util.inputToTimestamp(timeInput);
    updatedMedia.sex = sexInput;
    updatedMedia.length = parseFloat(lengthInput);
    updatedMedia.quality = qualityInput;

    // Post the proposal.
    SharkMatcher.Proposal
        .post_proposal(
            null, media, updatedMedia, 'update', 'media', 'open')
        .then(function(result) {
          alert('Successfully saved proposal.');
        })
        .catch(function(error) {
          console.log('Error saving proposal.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error saving proposal: ${message}`);
          }
        });
  } else if (Core.canEdit()) {
    // Directly update the media.
    media.name = nameInput;
    media.location = locationInput;
    media.time = Util.inputToTimestamp(timeInput);
    media.sex = sexInput;
    media.length = parseFloat(lengthInput);
    media.quality = qualityInput;
    media.type = typeInput;

    // Update database and log activity.
    SharkMatcher.Media
        .post_media(
            media.id, media.hash, media.name, media.source,
            media.individual.id, media.metadata, media.sex, media.length,
            media.location, media.time, media.type, media.quality)
        .then(function(result) {
        // Update page.
          location.reload(true);
        })
        .catch(function(error) {
          console.log('Error updating media.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error updating media: ${message}`);
          }
        });
  }
}

// Navigate after a media mutation returns the individual that remains active.
function navigateToFetchedIndividual(fetchIndividual) {
  if (!fetchIndividual || !fetchIndividual.name) {
    window.location.hash = Core.formHashPath('search');
    return;
  }

  const path = Routes.parsePath(window.location.hash);
  const params = path[1];
  if (fetchIndividual.name === params.name) {
    location.reload(true);
    return;
  }

  window.location.hash = Core.formHashPath('individual', {
    'name': fetchIndividual.name,
  });
}

function executeMediaDelete(media) {
  // Check if in proposing mode or editing mode.
  const currentMode = Core.getCurrentUserMode();
  if (currentMode === 'proposing') {
    // Post the proposal.
    SharkMatcher.Proposal
        .post_proposal(null, media, null, 'delete', 'media', 'open')
        .then(function(result) {
          alert('Successfully saved proposal.');
        })
        .catch(function(error) {
          console.log('Error saving proposal.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error saving proposal: ${message}`);
          }
        });
  } else if (Core.canEdit()) {
    // Deletes require confimation.
    const confirmation = confirm('Are you sure you want to delete this image?');
    if (confirmation === false) {
      return;
    }

    // Update database and log activity.
    SharkMatcher.Media.delete_media(media.id)
        .then(function(result) {
          const fetchIndividual = result['fetch_individual'];
          navigateToFetchedIndividual(fetchIndividual);
        })
        .catch(function(error) {
          console.log('Error deleting image.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error deleting image: ${message}`);
          }
        });
  }
}

function executeUnmatch(media) {
  const individual = media.individual;

  // Check if in proposing mode or editing mode.
  const currentMode = Core.getCurrentUserMode();
  if (currentMode === 'proposing') {
    // Represent the reference individual fully.
    const referenceIndividual = Util.individualGetReferenceIndividual(individual);
    individual.reference = referenceIndividual.id;

    // Find the individual that is being unmatched.
    const unmatchedIndividual = individual.associated_individuals.find(
        (associatedIndividual) => associatedIndividual.id === individual.id);
    unmatchedIndividual.reference = individual.id;

    // Post the proposal.
    SharkMatcher.Proposal
        .post_proposal(
            null, individual, unmatchedIndividual, 'unmatch', 'individual', 'open')
        .then(function(result) {
          alert('Successfully saved proposal.');
        })
        .catch(function(error) {
          console.log('Error saving proposal.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error saving proposal: ${message}`);
          }
        });
  } else if (Core.canEdit() || (currentMode === 'demo' && Core.canUnmatch())) {
    // Unmatch requires confimation.
    const confirmation =
      confirm('Are you sure you want to unmatch this image?');
    if (confirmation === false) {
      return;
    }

    // Update database and log activity.
    SharkMatcher.Individual.unmatch_individual(individual.id)
        .then(function(result) {
        // Open a new tab containing the unmatched individual.
          const newTabPath = Core.formHashPath('individual', {
            'name': individual.name,
          });
          window.open(newTabPath, '_blank', 'noopener,noreferrer');

          // Update page.
          const fetchIndividual = result['fetch_individual'];
          navigateToFetchedIndividual(fetchIndividual);
        })
        .catch(function(error) {
          console.log('Error updating media.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error updating media: ${message}`);
          }
        });
  }
}

function executeMediaReassign(media) {
  // Check if in proposing mode or editing mode.
  const currentMode = Core.getCurrentUserMode();
  if (currentMode === 'proposing') {
    // Get find the original reference individual id, then switch it to the
    // new reference individual id.
    const referenceIndividual = Util.individualGetReferenceIndividual(media.individual);
    referenceIndividual.reference = referenceIndividual.id;

    // Make a copy of the individual and apply the changes.
    const updatedIndividual = structuredClone(referenceIndividual);
    updatedIndividual.reference = media.individual.id;

    // Post the proposal.
    SharkMatcher.Proposal
        .post_proposal(
            null, referenceIndividual, updatedIndividual, 'update', 'individual', 'open')
        .then(function(result) {
          alert('Successfully saved proposal.');
        })
        .catch(function(error) {
          console.log('Error saving proposal.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error saving proposal: ${message}`);
          }
        });
  } else if (Core.canEdit()) {
    // Update database and log activity.
    SharkMatcher.Individual.update_reference(media.individual.id)
        .then(function(result) {
          // Update page.
          const fetchIndividual = result['fetch_individual'];
          navigateToFetchedIndividual(fetchIndividual);
        })
        .catch(function(error) {
          console.log('Error updating media.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error updating media: ${message}`);
          }
        });
  }
}

// The matcher being unreachable is reported instead of showing zero embeddings.
function renderEmbeddingsRows(result) {
  if (result.inference_available === false) {
    return `
                        <p class="matcher-unavailable">The matching service cannot be reached, so embedding details are unavailable.</p>
                        <div class="info-row"><span class="label">Embeddings Last Updated:</span> <span class="value">Unavailable</span></div>
                        <div class="info-row"><span class="label">Embeddings:</span> <span class="value">Unavailable</span></div>
                        <div class="info-row"><span class="label">Images without Embeddings:</span> <span class="value">Unavailable</span></div>`;
  }
  const lastUpdated = result.start_time ? Util.timestampToString(result.start_time) : 'Unknown';
  return `
                        <div class="info-row"><span class="label">Embeddings Last Updated:</span> <span class="value">${lastUpdated}</span></div>
                        <div class="info-row"><span class="label">Embeddings:</span> <span class="value">${result.embedded_images}</span></div>
                        <div class="info-row"><span class="label">Images without Embeddings:</span> <span class="value">${result.media_without_embedding_names.length}${result.media_without_embedding_names.length > 0 ? ' <button class="expand-btn media-without-expand" style="background: none; border: none; cursor: pointer; color: var(--text-color); font-size: 0.9em; padding: 0 4px;">+</button>' : ''}</span></div>
                        <div class="media-without-details" style="display: none; margin: 5px 20px; padding: 5px 40px; font-size: 0.9em; color: var(--text-secondary);">${Util.escapeHtml(result.media_without_embedding_names.join(', '))}</div>
                        <div class="info-row"><span class="label">Embeddings without Images:</span> <span class="value">${result.embedding_without_media_names.length}${result.embedding_without_media_names.length > 0 ? ' <button class="expand-btn embeddings-without-expand" style="background: none; border: none; cursor: pointer; color: var(--text-color); font-size: 0.9em; padding: 0 4px;">+</button>' : ''}</span></div>
                        <div class="embeddings-without-details" style="display: none; margin: 5px 20px; padding: 5px 40px; font-size: 0.9em; color: var(--text-secondary);">${Util.escapeHtml(result.embedding_without_media_names.join(', '))}</div>`;
}


// HTML render functions.
function renderHome(result, context, kwargs = {}) {
  const currentUser = Core.getContextUser();

  // Stats are cached by Util.loadCache('dataset_stats'), which every other consumer
  // (unembedded image list, model and method selectors) reads from.

  // Add results content.
  const html = `
    <div class="tab-results">
        <div class="results-container">
            <div class="home-pill">
                <div class="welcome">
                    <h3>Welcome, ${Util.escapeHtml(currentUser.user_display_name)}</h3>
                </div>
                <div style="width: fit-content; margin: 0 auto;">
                    <div class="dataset-overview">
                        <h4 style="text-align: center;">Dataset Overview</h4>
                        <div class="info-row"><span class="label">Unique ${Util.getDisplayName()}s:</span> <span class="value">${result.source_individuals}</span></div>
                        <div class="info-row"><span class="label">Images:</span> <span class="value">${result.media_individuals}</span></div>
                    </div>
                    <div class="embeddings-overview">
                        <h4 style="text-align: center;">Embeddings</h4>
                        ${renderEmbeddingsRows(result)}
                    </div>
                </div>
                <button class="embeddings-button pill-btn">Update Embeddings</button>
            </div>
        </div>
    </div>
    `;
  // TODO homepage population.
  // <p><b>Assigned Tasks: </b>placeholder</p>
  // <p><b>TODO Individuals: </b>placeholder</p>
  // <p><b>Match Suggestions: </b>placeholder</p>
  // Future also add information about all the associated objects
  // that exist.

  const doc = document.createElement('div');
  doc.innerHTML = html;
  const embeddingsButton = doc.querySelector('.embeddings-button');
  const cooldownMs = (result.update_cooldown_seconds ?? 60 * 60) * 1000;

  // Only allow updating embeddings to admins/collaborators.
  let userRoleValue = 0;
  if (currentUser && currentUser.user_role) {
    userRoleValue = Core.roleDict[currentUser.user_role];
  }
  if (userRoleValue < 1) {
    disableElement(embeddingsButton, 'Only admins/collaborators can update embeddings.');
  } else if (result.inference_available === false) {
    disableElement(embeddingsButton, 'The matching service cannot be reached.');
  } else if (Core.getCurrentUserMode() === 'demo') {
    disableElement(embeddingsButton, 'Embedding updates are unavailable in Demo mode.');
  } else if (result.update_in_progress) {
    disableElement(embeddingsButton, 'An embedding update is currently in progress.');
    embeddingsButton.textContent = 'Updating Embeddings...';
    startEmbeddingsPolling();
  } else if (result.update_cooldown_started_at &&
    result.update_cooldown_started_at * 1000 > Date.now() - cooldownMs) {
    // The server enforces the same cooldown; this only explains it on the button.
    const timeRemaining =
      result.update_cooldown_started_at * 1000 + cooldownMs - Date.now();
    const hours = Math.floor(timeRemaining / (60 * 60 * 1000));
    const minutes =
      Math.floor((timeRemaining % (60 * 60 * 1000)) / (60 * 1000));
    const seconds = Math.floor((timeRemaining % (60 * 1000)) / 1000);
    disableElement(
        embeddingsButton,
        `Embeddings have been updated in the last hour. ` +
        `Next update allowed in ${hours}h ${minutes}m ${seconds}s.`);
  } else {
    embeddingsButton.addEventListener('click', function() {
      executeUpdateEmbeddings();
    });
  }
  document.querySelector('.tab-results')
      .replaceWith(doc.querySelector('.tab-results'));

  window.sharkMatcher?.demo?.renderHome(document.querySelector('.home-pill'));

  // Expand button toggle logic, reusing Util.toggleVisibility as in
  // settings.js. Attached after replaceWith so elements are in the live DOM.
  const mediaExpandBtn = document.querySelector('.media-without-expand');
  if (mediaExpandBtn) {
    mediaExpandBtn.addEventListener('click', function() {
      const details = document.querySelector('.media-without-details');
      const isVisible = details.style.display !== 'none';
      Util.toggleVisibility(
          '', false, false,
          {'selector': '.media-without-details'});
      mediaExpandBtn.textContent = isVisible ? '+' : '-';
    });
  }
  const embeddingsExpandBtn =
    document.querySelector('.embeddings-without-expand');
  if (embeddingsExpandBtn) {
    embeddingsExpandBtn.addEventListener('click', function() {
      const details = document.querySelector('.embeddings-without-details');
      const isVisible = details.style.display !== 'none';
      Util.toggleVisibility(
          '', false, false,
          {'selector': '.embeddings-without-details'});
      embeddingsExpandBtn.textContent = isVisible ? '+' : '-';
    });
  }
}

// Populate an individual's media gallery, optionally using individuals[1] as the query individual
// in focus mode.
function renderGallery(individuals, context, kwargs = {}) {
  const container = kwargs['container'];

  // A source individual is returned with information for rendering.
  const individual = individuals[0];

  // For individual profile, render same individual twice in popup for image comparison.
  if (kwargs['page'] === 'individual' || kwargs['page'] === 'merge') {
    individuals.push(individual);
  }

  // Track the source individual so renderMediaProfile can use it for constraint coloring.
  if (kwargs['page'] === 'individual') {
    profileSourceIndividual = individual;
  }

  // Clear old object list.
  container.querySelector(kwargs['selector']).innerHTML = '';

  // Add individual names to the media list.
  for (let i = 0; i < individual.associated_medias.length; i++) {
    const media = individual.associated_medias[i];
    // Find the associated individual that points to this media
    const assocInd = individual.associated_individuals.find((ind) => ind && ind.media === media.id);
    if (assocInd) {
      media.individual = assocInd.id;
      if (assocInd.distance !== undefined) {
        media.distance = Math.round(assocInd.distance * 100) / 100;
      }
      media.individual_name = assocInd.name;
    }
  }

  // Sort the media list.
  if (individual.associated_medias.length === 0) {
    container.querySelector(kwargs['selector']).innerHTML = '<p>No images found.</p>';
    return [container, []];
  }

  let mediaList;
  if (kwargs['page'] === 'review') {
    // Review sort: flagged image first (hero), then other flagged images,
    // then the rest by time descending. Reference position is overridden so
    // the user lands directly on the image the retriever struggled on.
    const flaggedId = kwargs['flagged_image_id'];
    const flaggedMap = kwargs['flagged_image_k_map'] || {};
    const byTimeDesc = [...individual.associated_medias].sort(
        (a, b) => (b.time || 0) - (a.time || 0));
    mediaList = [];
    const heroIdx = byTimeDesc.findIndex((m) => m.id === flaggedId);
    if (heroIdx !== -1) {
      mediaList.push(byTimeDesc[heroIdx]);
      byTimeDesc.splice(heroIdx, 1);
    }
    const otherFlagged = byTimeDesc.filter((m) => flaggedMap[m.id] !== undefined);
    const nonFlagged = byTimeDesc.filter((m) => flaggedMap[m.id] === undefined);
    mediaList.push(...otherFlagged, ...nonFlagged);
  } else {
    let attribute = 'time';
    if (kwargs['page'] === 'match') {
      attribute = 'distance';
    }
    mediaList = Util.sortMediaReferenceFirst(
        individual.associated_medias, individual.reference, attribute);
  }

  if (!mediaList || mediaList.length === 0) {
    container.querySelector(kwargs['selector']).innerHTML = '<p>No images found.</p>';
    return [container, []];
  }

  // Store object ID for each gallery, for popup functionality.
  const objectId = individual.source;
  kwargs['object_id'] = objectId;

  // Build info pill for the reference image, displayed BELOW the image.
  const selectorClass = kwargs['selector'].substring(1);
  let refDistString = ``;
  if (kwargs['page'] === 'match') {
    let refDist = mediaList[0].distance;
    if (isNaN(refDist) || refDist === 1000) {
      refDist = 'N/A';
    }
    refDistString = `<span class="hero-distance">${refDist}</span>`;
  } else if (kwargs['page'] === 'review') {
    const flaggedMap = kwargs['flagged_image_k_map'] || {};
    const k = flaggedMap[mediaList[0].id];
    if (k !== undefined) {
      refDistString = `<span class="hero-distance">k=${k}</span>`;
    }
  }

  const unembeddedImages = Util.getUnembeddedImagesCache() || [];
  const firstMedia = mediaList[0];
  const refIndicator = (firstMedia && unembeddedImages.includes(firstMedia.name)) ?
    `<div class="unembedded-indicator" title="Processing or unembedded">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
    </div>` : '';

  let html = `
    <div class=${selectorClass}>
        <div class="gallery-item reference" object-id=${mediaList[0].id}>
            <figure>
                <div class="image-container">
                    ${refIndicator}
                    <img src="${Util.IMAGES_URL}${Util.escapeHtml(mediaList[0].filename.split('/').map(encodeURIComponent).join('/'))}?v=protected-1">
                </div>
                <figcaption class="hero-image-pill">
                    <span class="hero-filename">${Util.escapeHtml(mediaList[0].name || '')}</span>
                    <div class="hero-meta">
                        <span class="hero-name">Ref</span>
                        ${refDistString}
                    </div>
                </figcaption>
            </figure>
        </div>
        <div class="gallery-scroll"></div>
    </div>
    `;
  const doc = document.createElement('div');
  doc.innerHTML = html;
  doc.querySelector('img').addEventListener('click', function(event) {
    selectImage(event, context);
  });
  doc.querySelector('img').addEventListener('dblclick', function(event) {
    kwargs['index'] = 0;
    selectImage(event, context);
    renderPopup(individuals, kwargs);
  });

  // Populate gallery scroll.
  for (let i = 1; i < mediaList.length; i++) {
    const media = mediaList[i];

    let galleryDistString = ``;
    if (kwargs['page'] === 'match') {
      let galleryDist = media.distance;
      if (isNaN(galleryDist) || galleryDist === 1000) {
        galleryDist = 'N/A';
      }
      galleryDistString = `<span class="hero-distance">${galleryDist}</span>`;
    } else if (kwargs['page'] === 'review') {
      const flaggedMap = kwargs['flagged_image_k_map'] || {};
      const k = flaggedMap[media.id];
      if (k !== undefined) {
        galleryDistString = `<span class="hero-distance">k=${k}</span>`;
      }
    }

    const galleryIndicator = unembeddedImages.includes(media.name) ?
      `<div class="unembedded-indicator" title="Processing or unembedded">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"></circle>
              <polyline points="12 6 12 12 16 14"></polyline>
          </svg>
      </div>` : '';

    html = `
        <div class="gallery-item" object-id=${media.id}>
            <figure>
                <div class="image-container">
                    ${galleryIndicator}
                    <img src="${Util.IMAGES_URL}${Util.escapeHtml(media.filename.split('/').map(encodeURIComponent).join('/'))}?v=protected-1" loading="lazy">
                </div>
                <figcaption class="hero-image-pill">
                    <span class="hero-filename">${Util.escapeHtml(media.name || '')}</span>
                    <div class="hero-meta">
                        <span class="hero-name">${Util.escapeHtml(media.individual_name || '')}</span>
                        ${galleryDistString}
                    </div>
                </figcaption>
            </figure>
        </div>
        `;
    const innerDoc = document.createElement('div');
    innerDoc.innerHTML = html;
    innerDoc.querySelector('img').addEventListener('click', function(event) {
      selectImage(event, context);
    });
    innerDoc.querySelector('img').addEventListener('dblclick', function(event) {
      kwargs['index'] = i;
      selectImage(event, context);
      renderPopup(individuals, kwargs);
    });
    doc.querySelector('.gallery-scroll')
        .appendChild(innerDoc.querySelector('.gallery-item'));
  }
  container.querySelector(kwargs['selector'])
      .replaceWith(doc.querySelector(`.${selectorClass}`));
  return [container, mediaList, objectId];
}

function renderMedia(medias, context, kwargs = {}) {
  // Render media profile and media edit.
  renderMediaProfile(medias, context, kwargs);

  // Clear media edit first because this function doesn't clear itself,
  // a dependency when being used for upload files.
  const html = `
    <div class="media-menu">
        <div class="media-edit">
            <h3>Edit Image</h3>
        </div>
        <div class="media-object-edit">
            <h3>Edit Image Objects</h3>
        </div>
    </div>
    </div>
    `;
  const doc = document.createElement('div');
  doc.innerHTML = html;
  document.querySelector('.media-menu')
      .replaceWith(doc.querySelector('.media-menu'));

  // Render the media edit menu to edit media properties and the individual edit menu
  // to edit associated objects for this media individual, not the source individual.
  renderMediaEdit(
      medias, context,
      {'selector': '.media-edit', 'index': 0, 'page': 'individual'});
  Individual.renderIndividualEdit(
      [medias[0].individual], context,
      {'selector': '.media-object-edit', 'page': 'individual-media'});

  renderObjectList(
      [medias[0].individual], 'alternate_name', context,
      {
        'selector': '.media.alternate_name .object-list',
        'page': 'individual-media',
      });
  renderObjectList(
      [medias[0].individual], 'tag', context,
      {'selector': '.media.tag .object-list', 'page': 'individual-media'});
  renderObjectList(
      [medias[0].individual], 'label', context,
      {'selector': '.media.label .object-list', 'page': 'individual-media'});
}

function renderMediaProfile(medias, context, kwargs = {}) {
  // Clear old media profile.
  document.querySelector('.media .media-profile').innerHTML = '';

  const media = medias[0];
  const mediaTime = Util.timestampToString(media.time);
  const typeString = `
        <div class="info-row">
            <span class="label">Type:</span>
            <span class="value">${Util.displayValue(media.type)}</span>
        </div>`;

  const filteredNicknames = (media.individual.alternate_names || []).filter(
      (obj) => obj.individual === media.individual.id);
  const filteredTags = (media.individual.tags || []).filter(
      (obj) => obj.individual === media.individual.id);
  const filteredLabels = (media.individual.labels || []).filter(
      (obj) => obj.individual === media.individual.id);

  const nicknamesLinks = Util.listToLinks(filteredNicknames);
  const tagsLinks = Util.listToLinks(filteredTags);
  const labelsLinks = Util.listToLinks(filteredLabels);

  let nicknamesRow = '';
  if (nicknamesLinks !== '') {
    nicknamesRow = `
            <div class="info-row">
                <span class="label">Nicknames:</span>
                <span class="value">${nicknamesLinks}</span>
            </div>`;
  }

  let tagsRow = '';
  if (tagsLinks !== '') {
    tagsRow = `
            <div class="info-row">
                <span class="label">Tags:</span>
                <span class="value">${tagsLinks}</span>
            </div>`;
  }

  let labelsRow = '';
  if (labelsLinks !== '') {
    labelsRow = `
            <div class="info-row">
                <span class="label">Labels:</span>
                <span class="value">${labelsLinks}</span>
            </div>`;
  }

  // Apply constraint coloring to the associated individual's sex if it disagrees
  // with the source individual currently loaded on the profile page.
  let sexClass = '';
  if (profileSourceIndividual) {
    const sexColor = Util.sexConstraintColor(profileSourceIndividual.sex, media.individual.sex);
    if (sexColor) {
      sexClass = ` ${Util.constraintColorClass(sexColor)}`;
    }
  }

  const html = `
    <div object-id=${media.id}>
        <h3 style="text-align: center;">Properties for ${Util.escapeHtml(media.name)}</h3>
        <div style="width: fit-content; margin: 0 auto;">
            <div class="info-row">
                <span class="label">Image ${Util.getDisplayName()}:</span>
                <span class="value">${Util.escapeHtml(media.individual.name)}</span>
            </div>
            ${typeString}
            <div class="info-row">
                <span class="label">Sex:</span>
                <span class="value${sexClass}">${Util.displayValue(media.individual.sex)}</span>
            </div>
            <div class="info-row">
                <span class="label">Length:</span>
                <span class="value">${Util.displayValue(media.individual.length)}</span>
            </div>
            <div class="info-row">
                <span class="label">Location:</span>
                <span class="value">${Util.displayValue(media.location)}</span>
            </div>
            <div class="info-row">
                <span class="label">Time:</span>
                <span class="value">${mediaTime}</span>
            </div>
            <div class="info-row">
                <span class="label">Quality:</span>
                <span class="value">${Util.displayValue(media.quality)}</span>
            </div>
            ${nicknamesRow}
            ${tagsRow}
            ${labelsRow}
        </div>
        <div class="button-row" style="margin-top: 10px; display: flex; gap: 10px; justify-content: center;">
            <button class="media delete" object-id=${media.id}>Delete</button>
            <button class="media reference" object-id=${media.id}>
                Set as ${Util.getDisplayName()} Reference Image
            </button>
            <button class="media unmatch" object-id=${media.id}>
                Unmatch from ${Util.getDisplayName()}
            </button>
        </div>
    </div>
    `;
  const doc = document.createElement('div');
  doc.innerHTML = html;
  const deleteButton = doc.querySelector('.media.delete');
  const referenceButton = doc.querySelector('.media.reference');
  const unmatchButton = doc.querySelector('.media.unmatch');
  const currentMode = Core.getCurrentUserMode();

  if (currentMode === 'demo') {
    deleteButton.disabled = true;
    deleteButton.title = 'Image deletion is unavailable in Demo mode.';
    referenceButton.disabled = true;
    referenceButton.title = 'Reference assignment is unavailable in Demo mode.';
    if (!Core.canUnmatch() || media.individual.associated_medias.length === 1) {
      unmatchButton.disabled = true;
    } else {
      unmatchButton.disabled = false;
    }
  } else if (Core.isReadOnlyMode()) {
    if (deleteButton) deleteButton.disabled = true;
    if (referenceButton) referenceButton.disabled = true;
    if (unmatchButton) unmatchButton.disabled = true;
  } else {
    // Potentially disable unmatch button based on business logic
    // (only when not in viewing mode)
    if (media.individual.associated_medias.length === 1) {
      unmatchButton.disabled = true;
    } else {
      unmatchButton.disabled = false;
    }

    // Disable reference button if it's already the reference image.
    if (media.id === media.individual.reference) {
      referenceButton.disabled = true;
      referenceButton.title = 'Already reference image.';
    }
  }

  deleteButton.addEventListener('click', function() {
    executeMediaDelete(media);
  });
  referenceButton.addEventListener('click', function() {
    executeMediaReassign(media);
  });
  unmatchButton.addEventListener('click', function() {
    executeUnmatch(media);
  });
  document.querySelector('.media .media-profile')
      .appendChild(doc.querySelector('div'));
}

// Render the menu to edit a media object.
function renderMediaEdit(medias, context, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }
  const index = kwargs['index'];
  const media = medias[index];

  // Shark based attributes are stored with media when uploading.
  let lengthValue = '';
  let selectOptions;
  if (kwargs['page'] === 'upload') {
    selectOptions = [media.sex].concat(Util.sexOptions);
    if (media.length && media.length !== '') {
      lengthValue = `value="${media.length}"`;
    }
  } else {
    selectOptions = [media.individual.sex].concat(Util.sexOptions);
    if (media.individual.length && media.individual.length !== '') {
      lengthValue = `value="${media.individual.length}"`;
    }
  }

  let nameValue = '';
  if (media.name && media.name !== '') {
    nameValue = `value="${Util.escapeHtml(media.name)}"`;
  }
  let timeValue = '';
  if (media.time && media.time !== '') {
    timeValue = `value="${Util.timestampToInput(media.time, true)}"`;
  }

  let saveButton = '';
  if (kwargs['page'] === 'individual' && Core.getCurrentUserMode() !== 'demo') {
    saveButton = `<button class="media save">Update Image</button>`;
  }

  const html = `
    <div class="input-section">
        <div class="input-item">
            <label for="media-name-input-${index}">Name: </label>
            <input
                class="media name-input"
                id="media-name-input-${index}"
                type="text"
                placeholder="name"
                ${nameValue}/>
        </div>
        <div class="input-item">
            <label for="media-type-input-${index}">Type: </label>
            <select
                class="media type-input"
                id="media-type-input-${index}">
            </select>
        </div>
        <div class="input-item">
            <label for="media-min-quality-input-${index}">Quality: </label>
            <select
                class="media min-quality-input"
                id="media-min-quality-input-${index}">
            </select>
        </div>
        <div class="input-item">
            <label for="media-location-input-${index}">Location: </label>
            <select
                class="media location-input"
                id="media-location-input-${index}">
            </select>
        </div>
        <div class="input-item">
            <label for="media-time-input-${index}">Time: </label>
            <input
                class="media time-input"
                id="media-time-input-${index}"
                type="datetime-local"
                ${timeValue}/>
        </div>
        <div class="input-item">
            <label for="media-sex-input-${index}">Sex: </label>
            <select
                class="media sex-input"
                id="media-sex-input-${index}">
            </select>
        </div>
        <div class="input-item">
            <label for="media-length-input-${index}">Length: </label>
            <input
                class="media length-input"
                id="media-length-input-${index}"
                type="number"
                placeholder="length"
                ${lengthValue}/>
        </div>
        ${saveButton}
    </div>
    `;

  let doc = document.createElement('div');
  doc.innerHTML = html;

  if (Core.getCurrentUserMode() === 'demo') {
    for (const input of doc.querySelectorAll('.input-section input, .input-section select')) {
      input.disabled = true;
    }
  }

  // Update button when on the individual page.
  if (kwargs['page'] === 'individual' && Core.getCurrentUserMode() !== 'demo') {
    const saveButtonElement = doc.querySelector('.media.save');

    // Disable save button when in viewing mode
    if (Core.isReadOnlyMode()) {
      if (saveButtonElement) saveButtonElement.disabled = true;
    }

    saveButtonElement.addEventListener('click', function() {
      executeMediaSave(medias, kwargs);
    });
  }

  doc = Util.renderSelectOptions(
      selectOptions, context,
      {'container': doc, 'selector': '.media.sex-input'});
  selectOptions = [media.type].concat(Util.typeOptions);
  doc = Util.renderSelectOptions(
      selectOptions, context,
      {'container': doc, 'selector': '.media.type-input'});
  selectOptions = [media.quality].concat(Util.qualityOptions);
  doc = Util.renderSelectOptions(
      selectOptions, context,
      {'container': doc, 'selector': '.media.min-quality-input'});

  const dynamicOptions = Util.getSelectorOptions();
  selectOptions = [media.location].concat(dynamicOptions[1]);
  doc = Util.renderSelectOptions(
      selectOptions, context,
      {'container': doc, 'selector': '.media.location-input', 'allowNew': true, 'type': 'location'});

  container.querySelector(kwargs['selector'])
      .appendChild(doc.querySelector('.input-section'));
  return container;
}

// Populate an object list given a list of objects.
function renderObjectList(individuals, object, context, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }

  let individual;
  if (kwargs['page'] === 'merge') {
    individual = individuals[2];
  } else if (
    kwargs['page'] === 'individual' || kwargs['page'] === 'individual-media' ||
    kwargs['page'] === 'settings') {
    individual = individuals[0];
  } else if (kwargs['page'] === 'upload') {
    individual = individuals[kwargs['index']];
  }

  // Clear old object list.
  container.querySelector(kwargs['selector']).innerHTML = '';

  // Select object type.
  let objectList;
  if (object === 'alternate_name') {
    objectList = individual.alternate_names;
  } else if (object === 'tag') {
    objectList = individual.tags;
  } else if (object === 'label') {
    objectList = individual.labels;
  } else if (object === 'user') {
    objectList = individuals;
  }

  // Repopulate object list.
  for (let i = 0; i < objectList.length; i++) {
    // Render an individual's associated objects dependent on page type.
    if (object === 'user') {
      // Pass for users.
    } else if (kwargs['page'] === 'individual-media' && objectList[i].individual !== individual.id) {
      continue;
    } else if (
      kwargs['page'] !== 'upload' && kwargs['page'] !== 'merge' &&
      ((kwargs['page'] !== 'individual-media' &&
        objectList[i].individual !== individual.source) ||
        individual.source === null)) {
      continue;
    }

    // Show the type if there is one.
    let typeString = '';
    if (Util.typedObjects.includes(object) &&
      objectList[i].type != undefined &&
      objectList[i].type != '') {
      typeString = `, ${Util.escapeHtml(objectList[i].type)}`;
    }
    let timeString = '';
    if (Util.timedObjects.includes(object) &&
      objectList[i].time != undefined &&
      objectList[i].time != '') {
      timeString = `, ${Util.timestampToString(objectList[i].time)}`;
    }

    const html = `
        <div class="object-list-item" object-id=${objectList[i].id}>
            <div>${Util.escapeHtml(objectList[i].name)}${typeString}${timeString}</div>
            <button class="object delete" object-id=${objectList[i].id}>
                Delete
            </button>
        </div>
        `;
    const doc = document.createElement('div');
    doc.innerHTML = html;
    const deleteButton = doc.querySelector('.object.delete');
    if (Core.getCurrentUserMode() === 'demo' && object === 'label' && objectList[i].name === 'randomized') {
      deleteButton.disabled = true;
      deleteButton.title = 'Retained until the demo round is reset.';
    }

    // Disable delete button when in viewing mode
    if (Core.isReadOnlyMode() && kwargs['page'] !== 'settings') {
      if (deleteButton) deleteButton.disabled = true;
    }

    deleteButton.addEventListener('click', function(event) {
      // Preview does not go to db.
      if (kwargs['page'] === 'merge' || kwargs['page'] === 'upload') {
        Merge.previewExecuteObjectDelete(
            event, individuals, object, context, kwargs);
      } else {
        executeObjectDelete(event, individual, object, context, kwargs);
      }
    });
    container.querySelector(kwargs['selector'])
        .appendChild(doc.querySelector('.object-list-item'));
  }
  return container;
}

function renderTitle(page, context, kwargs = {}) {
  const html = `
    <div class="tab-title-container">
    <div class="tab-title">
        <h3>${Util.titles[page]}</h3>
        <span class="info">&#9432;</span>
    </div>
    <div class="instructions" style="display: none">
        <p>${Util.instructions[page]}</p>
    </div>
    <div>
    `;
  const doc = document.createElement('div');
  doc.innerHTML = html;
  const infoButton = doc.querySelector('.info');
  infoButton.addEventListener('click', function() {
    Util.toggleVisibility(
        'block', false, false, {'selector': '.instructions'});
  });
  document.querySelector('.tab-title')
      .replaceWith(doc.querySelector('.tab-title-container'));
}

// Populate a gallery given an individual and its list of media objects.
function renderPopup(individuals, kwargs = {}) {
  const path = Routes.parsePath(window.location.hash);
  const onMatchPage = path[0] === 'match';
  const existingPopup = document.querySelector('.popup');

  // If on the match page and a popup is already open, only update the right
  // gallery. The right gallery shows the result shark, which is what changes
  // when navigating.
  if (onMatchPage && existingPopup && individuals.length > 1) {
    const rightGallery = document.querySelector('.popup-gallery.right');
    if (rightGallery) {
      // individuals[0] is the result individual.
      // When rendering the result side, the selected image is at
      // kwargs['index'].
      renderSinglePopupGallery(
          rightGallery, individuals[0], 'Result ', kwargs['index'], 'Selected: ',
          kwargs['page'], individuals[1]);
    }

    // The object_id on the popup tracks which gallery item in the main view is
    // open. Update it to the new one.
    existingPopup.setAttribute('object-id', kwargs['object_id']);

    // The left gallery (query) remains untouched.
    return;
  }

  // Clear and re-render the popup container.
  let html = `
    <div class="popup-container">
    <div class="popup" object-id=${kwargs['object_id']}>
        <div class="popup-escape-container">
            <span>Esc</span>
        </div>
        <div class="popup-results"></div>
    </div>
    </div>
    `;
  const doc = document.createElement('div');
  doc.innerHTML = html;
  doc.querySelector('.popup-escape-container').addEventListener('click', function(event) {
    closePopup();
  });
  document.querySelector('.popup-container')
      .replaceWith(doc.querySelector('.popup-container'));

  // Determine the rendering order based on whether this is the match page and
  // whether multiple individuals exist.
  const renderOrder = onMatchPage && individuals.length > 1 ?
    [1, 0] :
    [...Array(individuals.length).keys()].reverse();

  for (const i of renderOrder) {
    const selector =
      individuals.length === 1 ? 'single' : (i === 1 ? 'left' : 'right');
    let headingPrefix = '';
    let selectedText = 'Selected: ';
    let index = kwargs['index'];

    if (individuals.length > 1) {
      if (i === 1) {
        headingPrefix = kwargs['page'] === 'match' ? 'Query ' : '';
        selectedText = 'Reference: ';
        index = 0; // Reference image index
      } else {
        headingPrefix = kwargs['page'] === 'match' ? 'Result ' : '';
      }
    }

    html = `<div class="popup-gallery ${selector}"></div>`;
    const galleryDoc = document.createElement('div');
    galleryDoc.innerHTML = html;
    const gallery = galleryDoc.querySelector('.popup-gallery');

    const compareIndividual =
      (individuals.length > 1 && i === 0) ? individuals[1] : null;
    renderSinglePopupGallery(
        gallery, individuals[i], headingPrefix, index, selectedText, kwargs['page'],
        compareIndividual);
    document.querySelector(`.popup-results`).appendChild(gallery);
  }
}

// Helper function to render the content of a single popup gallery (left or
// right).
function renderSinglePopupGallery(
    container, individual, headingPrefix, selectedIndex, selectedText, page,
    compareIndividual = null) {
  // Clear the container first
  container.innerHTML = '';

  // Get the reference individual given this source individual to diplay.
  const referenceIndividual = Util.individualGetReferenceIndividual(individual);

  let medias = individual.associated_medias;
  // Sort the medias, leaving the reference image first.
  let attribute = 'time';
  if (page === 'match') {
    attribute = 'distance';
  }
  medias = Util.sortMediaReferenceFirst(medias, referenceIndividual.id, attribute);

  let sexClass = '';
  let lengthClass = '';
  if (compareIndividual) {
    const compareRef = Util.individualGetReferenceIndividual(compareIndividual);
    const sexColor = Util.sexConstraintColor(compareRef.sex, referenceIndividual.sex);
    if (sexColor) sexClass = Util.constraintColorClass(sexColor);
    const lengthColor = Util.lengthConstraintColor(compareRef.length, referenceIndividual.length);
    if (lengthColor) lengthClass = Util.constraintColorClass(lengthColor);
  }

  const headingString = `
        <h3>${headingPrefix}${Util.getDisplayName()} ${Util.escapeHtml(referenceIndividual.name)}</h3>
        <p>Sex: <span class="${sexClass}">${Util.displayValue(referenceIndividual.sex)}</span> | Length: <span class="${lengthClass}">${Util.displayValue(referenceIndividual.length)}</span></p>
        `;

  // Render the selected item first, if a valid index is provided
  if (selectedIndex >= 0 && selectedIndex < medias.length) {
    const selectedMedia = medias[selectedIndex];
    let distance = selectedMedia.distance;
    let distString = '';
    if (page === 'match') {
      if (isNaN(distance) || distance === 1000) {
        distance = 'N/A';
      }
      distString = ` | Distance: ${distance}`;
    }

    const selectedIsReference = selectedIndex === 0;
    const selectedItemClass = selectedIsReference ?
      'popup-gallery-item selected reference' : 'popup-gallery-item selected';

    const html = `
        <div class="${selectedItemClass}"
             object-id=${selectedMedia.source}>
            <figure>
                ${headingString}
                <img src="${Util.IMAGES_URL}${Util.escapeHtml(selectedMedia.filename.split('/').map(encodeURIComponent).join('/'))}?v=protected-1">
                <figcaption>
                    ${selectedText}${Util.escapeHtml(selectedMedia.name)} | ${Util.escapeHtml(selectedMedia.individual_name || referenceIndividual.name)}${distString}
                </figcaption>
            </figure>
        </div>
        `;
    const doc = document.createElement('div');
    doc.innerHTML = html;
    doc.querySelector('img').addEventListener('click', function(event) {
      flipImage(event.target);
    });
    container.appendChild(doc.querySelector('.popup-gallery-item'));
  }

  // Fill gallery with remaining images.
  for (let j = 0; j < medias.length; j++) {
    if (j !== selectedIndex) {
      const media = medias[j];

      let distance = media.distance;
      let distString = '';
      if (page === 'match') {
        if (isNaN(distance) || distance === 1000) {
          distance = 'N/A';
        }
        distString = ` | Distance: ${distance}`;
      }

      // After sortMediaReferenceFirst, medias[0] is the reference image.
      const isReference = j === 0;
      const itemClass = isReference ?
        'popup-gallery-item reference' : 'popup-gallery-item';
      const referencePrefix = isReference ? 'Reference: ' : '';

      const html = `
            <div class="${itemClass}" object-id=${media.source}>
                <figure>
                    ${headingString}
                <img src="${Util.IMAGES_URL}${Util.escapeHtml(media.filename.split('/').map(encodeURIComponent).join('/'))}?v=protected-1">
                    <figcaption>
                        ${referencePrefix}${Util.escapeHtml(media.name)} | ${Util.escapeHtml(media.individual_name || referenceIndividual.name)}${distString}
                    </figcaption>
                </figure>
            </div>
            `;
      const innerDoc = document.createElement('div');
      innerDoc.innerHTML = html;
      innerDoc.querySelector('img').addEventListener('click', function(event) {
        flipImage(event.target);
      });
      container.appendChild(innerDoc.querySelector('.popup-gallery-item'));
    }
  }
}

function closePopup() {
  // Close popup and remove all it's event listeners.
  document.querySelector('.popup-container').innerHTML = '';
}

// Helper functions.
// Activate the event listeners for the popup.
function keydownFunctionality() {
  document.addEventListener('keydown', function(event) {
    const path = Routes.parsePath(window.location.hash);

    // Enter functionality
    if (event.code === 'Enter') {
      if (path.includes('search')) {
        Search.triggerSearch();
      } else if (path.includes('match')) {
        Match.triggerMatchSearch();
      } else if (path.includes('review')) {
        Review.triggerReviewSearch();
      } else if (path.includes('history')) {
        History.triggerActionSearch();
      } else if (path.includes('proposals')) {
        Proposal.triggerProposalSearch();
      }
    }

    // Popup functionality.
    const popup = document.querySelector('.popup');
    if (popup != null) {
      // Flip image on KeyF.
      if (event.code === 'ArrowRight' || event.code === 'ArrowLeft') {
        event.preventDefault();
        let scrollableElement = document.querySelector('.popup-gallery.right');
        if (scrollableElement === null) {
          scrollableElement = document.querySelector('.popup-gallery');
        }

        if (event.key === 'ArrowLeft') {
          scrollableElement.scrollLeft -= 200;
        } else if (event.key === 'ArrowRight') {
          scrollableElement.scrollLeft += 200;
        }
      } else if (event.code === 'ArrowUp' || event.code === 'ArrowDown') {
        event.preventDefault();

        const objectId = parseInt(popup.getAttribute('object-id'));

        // Find the current gallery, if it is one of a list.
        const gallery =
          document.querySelector(`.results-item[object-id="${objectId}"]`);
        if (gallery != null) {
          // Select the new gallery to display.
          let newGallery;
          if (event.code === 'ArrowUp') {
            newGallery = gallery.previousElementSibling;
          } else if (event.code === 'ArrowDown') {
            newGallery = gallery.nextElementSibling;
          }

          if (newGallery != null) {
            const classList = newGallery.classList;
            if (classList.contains('results-item')) {
              // Open the new gallery with a simulated click.
              newGallery.querySelector('img').dispatchEvent(
                  new MouseEvent('dblclick'));
            }
          }
        }
      } else if (event.code === 'Escape') {
        closePopup();
      } else if (event.code === 'KeyF') {
        // Find the hovered image within the popup
        const hoveredImage = popup.querySelector('.popup-gallery img:hover');
        if (hoveredImage) {
          flipImage(hoveredImage);
        }
      }
    }
  });
}

function flipImage(image) {
  if (image.style.transform === 'scaleX(-1)') {
    image.style.transform = 'scaleX(1)';
  } else {
    image.style.transform = 'scaleX(-1)';
  }
}

// TODO update to remove green boarder bugs.
function selectImage(event, context) {
  const path = Routes.parsePath(window.location.hash);
  const image = event.target;

  // Remove old selected image and set new one.
  const newSelected = image.closest('.gallery-item');
  if (!newSelected) {
    return;
  }
  const gallery = newSelected.parentElement;
  const previousSelected = gallery.querySelector('.gallery-item.selected');

  // Find the previous selected gallery item and unselect it.
  try {
    const previousSelectedClass = previousSelected.getAttribute('class');
    if (previousSelectedClass.includes('reference')) {
      previousSelected.className = 'gallery-item reference';
    } else {
      previousSelected.className = 'gallery-item';
    }
  } catch {
  }

  // When on the profile page, remove media profile view.
  if (path.includes('individual')) {
    Util.toggleVisibility('block', false, true, {'selector': '.media'});
  }

  // Select new image if a different image is clicked.
  if (previousSelected !== newSelected) {
    const selectedClass = newSelected.getAttribute('class');
    if (selectedClass.includes('reference')) {
      newSelected.className = 'gallery-item selected reference';
    } else {
      newSelected.className = 'gallery-item selected';
    }

    // When on the profile page, add media profile view.
    if (path.includes('individual')) {
      Util.toggleVisibility('block', true, false, {'selector': '.media'});
      const mediaId = parseInt(newSelected.getAttribute('object-id'));
      executeMediaSearch(null, {'id': mediaId}, context, renderMedia);
    }
  }
}

export {
  init,
  handlerHome,
  executeObjectSave,
  renderGallery,
  renderObjectList,
  renderMediaEdit,
  renderTitle,
  keydownFunctionality,
};
