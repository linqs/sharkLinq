import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Home from './home.js';
import * as Merge from './merge.js';
import * as Routes from './routes.js';
import * as Util from './util.js';

function init() {
  Routes.addRoute(/^individual$/, handlerIndividual, 'individual');
}

function handlerIndividual(path, params, context) {
  Home.handlerHome(path, params, context);

  // Tab content.
  let html = `
        <div class="tab-content">
            <div class="tab-title"></div>
        </div>
    `;
  let doc = new DOMParser().parseFromString(html, 'text/html');
  document.querySelector('.tab-content')
      .replaceWith(doc.querySelector('.tab-content'));

  Home.renderTitle('individual', context);

  // Results content.
  html = `
        <div class="tab-results">
            <div class="results-container"></div>
        </div>
    `;
  doc = new DOMParser().parseFromString(html, 'text/html');
  document.querySelector('.tab-results')
      .replaceWith(doc.querySelector('.tab-results'));

  // Populate results.
  if (Object.keys(params).length !== 0 && params.name !== null) {
    executeIndividualSearch(path, params, context, renderIndividual);
  }
}

// API execution functions.
// Search for an individual or list of individuals, as specified by the name param, then
// execute the passed renderFunction.
function executeIndividualSearch(
    path, params, context, renderFunction, kwargs = {}) {
  // Unify name/names parameter and keywords parameter to lists.
  let names = null;
  if (params.name !== undefined) {
    names = params.name;
  } else if (params.names !== undefined) {
    names = params.names;
  }
  let keywords = null;
  if (params.keywords !== undefined) {
    keywords = params.keywords;
  }

  // Individuals are searched by names from the frontend.
  SharkMatcher.Individual.get_individuals(null, names, keywords, true, true)
      .then(function(result) {
        kwargs['load_index'] = 1;
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

function executeIndividualSave(individual, context) {
  // Get and validate inputs. TODO, allow like"ish" inputs, validate inputs.
  const sexInput = document.querySelector('.sex-input').value;
  const lengthInput = document.querySelector('.length-input').value;

  // Check if in proposing mode or editing mode.
  const currentMode = Core.getCurrentUserMode();
  if (currentMode === 'proposing') {
    // Compact the associated lists back into simple IDs to cleanly reduce the
    // JSON payload over the network.
    const compactedOriginal = Util.compactAssociatedObjects(individual);

    // Make a copy of the individual and apply the changes.
    const updatedIndividual = structuredClone(compactedOriginal);
    updatedIndividual.sex = sexInput;
    updatedIndividual.length = parseFloat(lengthInput);

    // Post the proposal.
    SharkMatcher.Proposal.post_proposal(
        null, compactedOriginal, updatedIndividual, 'update', 'individual', 'open')
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
    individual.sex = sexInput;
    individual.length = parseFloat(lengthInput);

    // Update database and log activity. Note we don't pass the discarded associational arrays.
    SharkMatcher.Individual.post_individual(
        individual.id, null, individual.length, individual.sex,
        individual.source, individual.reference,
        individual.media)
        .then(function(result) {
          // Update page.
          location.reload(true);
        })
        .catch(function(error) {
          console.log('Error updating individual.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error updating ${Util.getDisplayName().toLowerCase()}: ${message}`);
          }
        });
  }
}

// HTML render functions.
function renderIndividual(individuals, context, kwargs = {}) {
  const individual = individuals[0];

  // Handle individual error.
  if (individual === undefined || individual.id === -1) {
    const html = `
        <div>
            <h3>${Util.getDisplayName()} not found.</h3>
        </div>
        `;
    const doc = new DOMParser().parseFromString(html, 'text/html');
    document.querySelector('.tab-results')
        .appendChild(doc.querySelector('div'));
    return;
  }

  const html = `
    <div class="tab-results">
    <div class="tab-content individual">
    <div class="row">
        <div class="individual-profile"></div>
        <div class="individual-edit"></div>
    </div>
    <div class="gallery">
        <h4>Associated Images</h4>
        <div class="result-gallery"></div>
        <div class="media" style="display: none">
            <div class="media-profile"></div>
            <div class="media-menu"></div>
        </div>
    </div>
    </div>
    </div>
    `;

  // Add child elements and functionality.
  let doc = new DOMParser().parseFromString(html, 'text/html');
  doc = renderIndividualProfile(
      [individual], context,
      {'container': doc, 'selector': '.individual-profile', 'page': 'individual'});
  const renderResult = Home.renderGallery(
      [individual], context,
      {'container': doc, 'selector': '.result-gallery', 'page': 'individual'});
  doc = renderResult[0];
  document.querySelector('.tab-results')
      .replaceWith(doc.querySelector('.tab-results'));

  window.sharkMatcher?.demo?.renderKeys(document.querySelector('.individual-profile'), individual);

  if (Core.getCurrentUserMode() !== 'demo') {
    renderIndividualEdit(
        [individual], context, {'selector': '.individual-edit', 'page': 'individual'});
    Home.renderObjectList(
        [individual], 'alternate_name', context,
        {'selector': '.alternate_name .object-list', 'page': 'individual'});
    Home.renderObjectList(
        [individual], 'tag', context,
        {'selector': '.tag .object-list', 'page': 'individual'});
    Home.renderObjectList(
        [individual], 'label', context,
        {'selector': '.label .object-list', 'page': 'individual'});
  }
}

// Populate an individual profile area given an individual.
function renderIndividualProfile(individuals, context, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }

  // A source individual is returned with information for rendering.
  const sourceIndividual = individuals[0];

  if (!sourceIndividual) {
    console.error('renderIndividualProfile: sourceIndividual is undefined', individuals);
    return container;
  }

  // The reference field represents the id of the reference individual, which is
  // the individual who's information is displayed to the user. Get a pointer to this
  // individual.
  const referenceIndividual = Util.individualGetReferenceIndividual(sourceIndividual);

  let finalAlternateNames = [];
  let finalTags = [];
  let finalLabels = [];
  // When merging, keep all associated objects for rendering, they have been
  // filtered.
  if (kwargs['page'] === 'merge_output') {
    finalAlternateNames = sourceIndividual.alternate_names;
    finalTags = sourceIndividual.tags;
    finalLabels = sourceIndividual.labels;
  } else {
    // Filter to only the individual objects.
    const objectLists = [
      [sourceIndividual.alternate_names, finalAlternateNames],
      [sourceIndividual.tags, finalTags],
      [sourceIndividual.labels, finalLabels]];
    for (const objectList of objectLists) {
      for (const object of objectList[0]) {
        if (object.individual == sourceIndividual.id) {
          objectList[1].push(object);
        }
      }
    }
  }

  const nicknamesLinks =
        Util.listToLinks(finalAlternateNames);
  const tagsLinks = Util.listToLinks(finalTags);
  const labelsLinks = Util.listToLinks(finalLabels);

  // Optionally add a link to the individual profile.
  const individualName = referenceIndividual ? referenceIndividual.name : sourceIndividual.name;
  const profileLink = Core.formHashPath('individual', {
    'name': individualName,
  });

  // Build the profile header: compact ID label for result cards, full h3 elsewhere.
  let profileString = '';
  if (kwargs['page'] === 'match_result' || kwargs['page'] === 'individual' || kwargs['page'] === 'match_query' || kwargs['page'] === 'review_unmatch') {
    const displayName = referenceIndividual ? referenceIndividual.name : sourceIndividual.name;
    profileString = `
        <div class="card-id-label">
            <a href="${profileLink}" class="profile-link card-title-pill">${Util.getDisplayName()} ${Util.escapeHtml(displayName)}</a>
        </div>`;
  } else if (kwargs['page'] === 'merge_output') {
    const name = referenceIndividual ? referenceIndividual.name : sourceIndividual.name;
    profileString = `<h3>${Util.getDisplayName()} ${Util.escapeHtml(name)} Preview</h3>`;
  } else {
    const name = referenceIndividual ? referenceIndividual.name : sourceIndividual.name;
    profileString = `<h3>${Util.getDisplayName()} ${Util.escapeHtml(name)}</h3>`;
  }

  // Build the compact icon action button group for result cards.
  let cardActions = '';
  if (kwargs['page'] === 'match_result') {
    const matchLink = Core.formHashPath(
        'match', {'name': referenceIndividual.name, 'method': 'nearest neighbor'});
    const query = individuals[1];
    const queryReferenceIndividual = Util.individualGetReferenceIndividual(query);
    const mergeLink = Core.formHashPath(
        'merge',
        {'name1': queryReferenceIndividual.name, 'name2': referenceIndividual.name});
    cardActions = `
        <div class="card-actions">
            <a href="${matchLink}" class="card-action-btn" title="Match Against This ${Util.getDisplayName()}">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round"
                    stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                <span>Match</span>
            </a>
            <a href="${mergeLink}" class="card-action-btn" title="Merge with Query ${Util.getDisplayName()}">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round"
                    stroke-linejoin="round">
                    <path d="M8 6H5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h3"></path>
                    <path d="M16 6h3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-3"></path>
                    <line x1="12" y1="2" x2="12" y2="22"></line>
                </svg>
                <span>Merge</span>
            </a>
        </div>`;
  } else if (kwargs['page'] === 'individual') {
    const matchLink = Core.formHashPath(
        'match', {'name': referenceIndividual.name, 'method': 'nearest neighbor'});
    cardActions = `
        <div class="card-actions">
            <a href="${matchLink}" class="card-action-btn" title="Match Against This ${Util.getDisplayName()}">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round"
                    stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                <span>Match</span>
            </a>
        </div>`;
  } else if (kwargs['page'] === 'review_unmatch') {
    // Prefer the flagged image's name so the Match button investigates the
    // specific image the retriever struggled on. Fall back to the individual
    // name if the flagged media isn't found (shouldn't happen in practice).
    let matchName = referenceIndividual ? referenceIndividual.name : sourceIndividual.name;
    const flaggedId = kwargs['flagged_image_id'];
    if (flaggedId !== undefined && sourceIndividual.associated_medias) {
      const flaggedMedia = sourceIndividual.associated_medias.find((m) => m.id === flaggedId);
      if (flaggedMedia && flaggedMedia.name) {
        matchName = flaggedMedia.name;
      }
    }
    const matchLink = Core.formHashPath(
        'match', {'name': matchName, 'method': 'nearest neighbor'});
    cardActions = `
        <div class="card-actions">
            <a href="${matchLink}" class="card-action-btn" title="Match the flagged image">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round"
                    stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                <span>Match</span>
            </a>
        </div>`;
  }

  // Conditionally generate info rows.
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

  // Find the most recent sighting date across all associated media.
  let lastSeenString = 'U';
  if (sourceIndividual.associated_medias && sourceIndividual.associated_medias.length > 0) {
    let maxTime = null;
    for (const media of sourceIndividual.associated_medias) {
      if (media.time && (maxTime === null || media.time > maxTime)) {
        maxTime = media.time;
      }
    }
    if (maxTime !== null) {
      lastSeenString = Util.timestampToString(maxTime, false);
    }
  }

  // Apply constraint coloring to property values when rendering a match result,
  // so the user can see at a glance how the result agrees with the query.
  let sexClass = '';
  let lengthClass = '';
  if (kwargs['page'] === 'match_result') {
    const queryIndividual = individuals[1];
    const sexColor = Util.sexConstraintColor(queryIndividual.sex, sourceIndividual.sex);
    if (sexColor) {
      sexClass = ` ${Util.constraintColorClass(sexColor)}`;
    }
    const lengthColor = Util.lengthConstraintColor(queryIndividual.length, sourceIndividual.length);
    if (lengthColor) {
      lengthClass = ` ${Util.constraintColorClass(lengthColor)}`;
    }
  }

  // Review-specific row: rank at which the correct label first appeared.
  // k=1 would be a perfect match; higher k means the retriever ranked many
  // wrong individuals ahead of the truth.
  let reviewRow = '';
  if (kwargs['page'] === 'review_unmatch' && kwargs['k_value'] !== undefined) {
    reviewRow = `
        <div class="info-row">
            <span class="label">K at Hit:</span>
            <span class="value">${kwargs['k_value']}</span>
        </div>`;
  }

  const html = `
    <div>
        ${profileString}

        ${reviewRow}
        <div class="info-row">
            <span class="label">Sex:</span>
            <span class="value${sexClass}">${Util.displayValue(sourceIndividual.sex)}</span>
        </div>
        <div class="info-row">
            <span class="label">Length:</span>
            <span class="value${lengthClass}">${Util.displayValue(sourceIndividual.length)}</span>
        </div>
        <div class="info-row">
            <span class="label">Last Seen:</span>
            <span class="value">${lastSeenString}</span>
        </div>
        ${nicknamesRow}
        ${tagsRow}
        ${labelsRow}

        ${cardActions}
    </div>
    `;
  const doc = new DOMParser().parseFromString(html, 'text/html');

  container.querySelector(kwargs['selector'])
      .appendChild(doc.querySelector('div'));
  return container;
}

// Render the menu to edit an individual.
function renderIndividualEdit(individuals, context, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }

  let propertiesButtonString = ``;
  let propertiesString = ``;
  let headingString = ``;
  let mediaPrefix = ``;
  if (kwargs['page'] === 'individual-media') {
    mediaPrefix = 'media ';
  }
  // Element ids cannot contain spaces, so they use a hyphenated form of the class prefix.
  const idPrefix = mediaPrefix.trim() ? 'media-' : '';

  // Select the correct individual based on page application.
  let individual = individuals[0];
  if (kwargs['page'] === 'merge_output') {
    individual = individuals[2];
  } else if (kwargs['page'] === 'upload') {
    individual = individuals[kwargs['index']];
  }

  let lengthValue = '';
  if (individual.length != null) {
    lengthValue = `value="${individual.length}"`;
  }

  // Add optional html for merge preview.
  let nameInput = ``;
  if (kwargs['page'] === 'merge_output') {
    nameInput = `
        <div class="input-item">
            <label for="reference-input">Reference: </label>
            <select
                class="reference-input"
                id="reference-input">
            </select>
        </div>
        `;
  }

  // Only display properties edit contidional the page.
  if (kwargs['page'] !== 'upload' && kwargs['page'] !== 'individual-media') {
    headingString = `<h4>Edit ${Util.getDisplayName()}</h4>`;
    propertiesButtonString =
            `<button class="properties-button pill-btn">Properties</button>`;
    propertiesString = `
        <div class="objects-area properties" style="display: none">
            <h4>Properties:</h4>
            <div class="input-section">
                ${nameInput}
                <div class="input-item">
                    <label for="sex-input">Sex: </label>
                    <select
                        class="sex-input"
                        id="sex-input">
                    </select>
                </div>
                <div class="input-item">
                    <label for="length-input">Length: </label>
                    <input
                        class="length-input"
                        id="length-input"
                        type="number"
                        placeholder="length"
                        ${lengthValue}/>
                </div>
                <button class="properties save pill-btn">Update ${Util.getDisplayName()}</button>
            </div>
        </div>
    `;
  }

  const html = `
    <div>
    ${headingString}
    <div class="settings-btn-row">
        ${propertiesButtonString}
        <button class="alternate_name-button pill-btn">Nicknames</button>
        <button class="tag-button pill-btn">Tags</button>
        <button class="label-button pill-btn">Labels</button>
    </div>

    ${propertiesString}

    <div class="objects-area ${mediaPrefix}alternate_name"
     style="display: none">
        <h4>Nicknames:</h4>
        <div class="object-list"></div>
        <div class="input-section">
            <div class="input-item">
                <label for="${idPrefix}alternate_name-name-input">
                  Nickname:
                </label>
                <input
                    class="${mediaPrefix}alternate_name name-input"
                    id="${idPrefix}alternate_name-name-input"
                    type="text"
                    placeholder="name"/>
            </div>
            <button class="alternate_name save pill-btn">Add Nickname</button>
        </div>
    </div>

    <div class="objects-area ${mediaPrefix}tag" style="display: none">
        <h4>Tags:</h4>
        <div class="object-list"></div>
        <div class="input-section">
            <div class="input-item">
                <label for="${idPrefix}tag-name-input">Tag: </label>
                <input
                    class="${mediaPrefix}tag name-input"
                    id="${idPrefix}tag-name-input"
                    type="text"
                    placeholder="name"/>
            </div>
            <div class="input-item">
                <label for="${idPrefix}tag-type-input">
                  Tag Type:
                </label>
                <select
                    class="${mediaPrefix}tag type-input"
                    id="${idPrefix}tag-type-input">
                </select>
            </div>
            <div class="input-item">
                <label for="${idPrefix}tag-time-input">Deploy Time: </label>
                <input
                    class="${mediaPrefix}tag time-input"
                    id="${idPrefix}tag-time-input"
                    type="date"/>
            </div>
            <button class="tag save pill-btn">Add Tag</button>
        </div>
    </div>

    <div class="objects-area ${mediaPrefix}label" style="display: none">
        <h4>Labels:</h4>
        <div class="object-list"></div>
        <div class="input-section">
            <div class="input-item">
                <label for="${idPrefix}label-name-input">Label: </label>
                <select
                    class="${mediaPrefix}label name-input"
                    id="${idPrefix}label-name-input">
                </select>
            </div>
            <button class="label save">Add Label</button>
        </div>
    </div>
    </div>
    `;
  const doc = new DOMParser().parseFromString(html, 'text/html');

  // Media prefix to select the correct elements for selector population.
  if (kwargs['page'] === 'individual-media') {
    mediaPrefix = '.media ';
  }

  // Menu toggle buttons.
  if (kwargs['page'] !== 'upload' && kwargs['page'] !== 'individual-media') {
    const propertiesButton = doc.querySelector('.properties-button');
    propertiesButton.addEventListener('click', function(event) {
      const container = event.target.parentElement.parentElement.parentElement;
      Util.toggleVisibility(
          'None', false, true,
          {'container': container, 'selector': '.objects-area.alternate_name'});
      Util.toggleVisibility(
          'None', false, true, {'container': container, 'selector': '.objects-area.tag'});
      Util.toggleVisibility(
          'None', false, true, {'container': container, 'selector': '.objects-area.label'});
      Util.toggleVisibility(
          '', false, false,
          {'container': container, 'selector': '.objects-area.properties'});
    });
    const propertiesSaveButton = doc.querySelector('.properties.save');

    // Disable properties save button when in viewing mode
    if (Core.isReadOnlyMode()) {
      if (propertiesSaveButton) propertiesSaveButton.disabled = true;
    }

    propertiesSaveButton.addEventListener('click', function() {
      if (kwargs['page'] === 'merge_output') {
        Merge.previewExecuteIndividualSave(individuals, context);
      } else if (kwargs['page'] === 'individual') {
        executeIndividualSave(individual, context);
      }
    });
  }
  const alternateNameButton = doc.querySelector('.alternate_name-button');
  const tagButton = doc.querySelector('.tag-button');
  const labelButton = doc.querySelector('.label-button');
  alternateNameButton.addEventListener('click', function(event) {
    const container = event.target.parentElement.parentElement.parentElement;
    try {
      Util.toggleVisibility(
          'None', false, true,
          {'container': container, 'selector': '.objects-area.properties'});
    } catch { }
    Util.toggleVisibility(
        'None', false, true, {'container': container, 'selector': '.objects-area.tag'});
    Util.toggleVisibility(
        'None', false, true, {'container': container, 'selector': '.objects-area.label'});
    Util.toggleVisibility(
        '', false, false,
        {'container': container, 'selector': '.objects-area.alternate_name'});
  });
  tagButton.addEventListener('click', function(event) {
    const container = event.target.parentElement.parentElement.parentElement;
    Util.toggleVisibility(
        'None', false, true,
        {'container': container, 'selector': '.objects-area.alternate_name'});
    try {
      Util.toggleVisibility(
          'None', false, true,
          {'container': container, 'selector': '.objects-area.properties'});
    } catch { }
    Util.toggleVisibility(
        'None', false, true, {'container': container, 'selector': '.objects-area.label'});
    Util.toggleVisibility(
        '', false, false, {'container': container, 'selector': '.objects-area.tag'});
  });
  labelButton.addEventListener('click', function(event) {
    const container = event.target.parentElement.parentElement.parentElement;
    Util.toggleVisibility(
        'None', false, true,
        {'container': container, 'selector': '.objects-area.alternate_name'});
    try {
      Util.toggleVisibility(
          'None', false, true,
          {'container': container, 'selector': '.objects-area.properties'});
    } catch { }
    Util.toggleVisibility(
        'None', false, true, {'container': container, 'selector': '.objects-area.tag'});
    Util.toggleVisibility(
        '', false, false, {'container': container, 'selector': '.objects-area.label'});
  });

  // Save buttons. Need to pass the container in case of upload since there are
  // lists of elements in that page.
  const alternateNameSaveButton = doc.querySelector('.alternate_name.save');
  const tagSaveButton = doc.querySelector('.tag.save');
  const labelSaveButton = doc.querySelector('.label.save');

  // Disable save buttons when in viewing mode
  if (Core.isReadOnlyMode()) {
    if (alternateNameSaveButton) alternateNameSaveButton.disabled = true;
    if (tagSaveButton) tagSaveButton.disabled = true;
    if (labelSaveButton) labelSaveButton.disabled = true;
  }
  alternateNameSaveButton.addEventListener('click', function(event) {
    if (kwargs['page'] === 'merge_output') {
      Merge.previewExecuteObjectSave(
          'alternate_name', individuals, context, {'page': 'merge'});
    } else if (kwargs['page'] === 'individual' || kwargs['page'] === 'individual-media') {
      Home.executeObjectSave(`${mediaPrefix}.alternate_name`, individual, context);
    } else if (kwargs['page'] === 'upload') {
      kwargs['container'] =
                event.target.parentElement.parentElement.parentElement;
      Merge.previewExecuteObjectSave(
          'alternate_name', individuals, context, kwargs);
    }
  });
  tagSaveButton.addEventListener('click', function(event) {
    if (kwargs['page'] === 'merge_output') {
      Merge.previewExecuteObjectSave(
          'tag', individuals, context, {'page': 'merge'});
    } else if (kwargs['page'] === 'individual' || kwargs['page'] === 'individual-media') {
      Home.executeObjectSave(`${mediaPrefix}.tag`, individual, context);
    } else if (kwargs['page'] === 'upload') {
      kwargs['container'] =
                event.target.parentElement.parentElement.parentElement;
      Merge.previewExecuteObjectSave('tag', individuals, context, kwargs);
    }
  });
  labelSaveButton.addEventListener('click', function(event) {
    if (kwargs['page'] === 'merge_output') {
      Merge.previewExecuteObjectSave(
          'label', individuals, context, {'page': 'merge'});
    } else if (kwargs['page'] === 'individual' || kwargs['page'] === 'individual-media') {
      Home.executeObjectSave(`${mediaPrefix}.label`, individual, context);
    } else if (kwargs['page'] === 'upload') {
      kwargs['container'] =
                event.target.parentElement.parentElement.parentElement;
      Merge.previewExecuteObjectSave('label', individuals, context, kwargs);
    }
  });

  container.querySelector(kwargs['selector'])
      .appendChild(doc.querySelector('div'));

  // Populate selector options.
  if (kwargs['page'] !== 'upload') {
    const selectOptions = [individual.sex].concat(Util.sexOptions);
    Util.renderSelectOptions(
        selectOptions, context,
        {'container': container, 'selector': '.sex-input'});
  }
  if (kwargs['page'] === 'merge_output') {
    // Populate reference individual options with the current reference individual
    // followed by all other options.
    const referenceIndividual = Util.individualGetReferenceIndividual(individual);
    const selectOptions = [referenceIndividual.name];
    for (const associatedIndividual of individual.associated_individuals) {
      if (associatedIndividual != null) {
        selectOptions.push(associatedIndividual.name);
      }
    }
    Util.renderSelectOptions(
        selectOptions, context,
        {'container': container, 'selector': '.reference-input'});
  }

  const dynamicOptions = Util.getSelectorOptions();

  let selectOptions = dynamicOptions[2];
  Util.renderSelectOptions(
      selectOptions, context,
      {'container': container, 'selector': `${mediaPrefix}.tag.type-input`, 'allowNew': true, 'type': 'tag'});
  selectOptions = dynamicOptions[0];
  Util.renderSelectOptions(
      selectOptions, context,
      {'container': container, 'selector': `${mediaPrefix}.label.name-input`, 'allowNew': true, 'type': 'label'});

  return container;
}


export {
  init,
  executeIndividualSearch,
  renderIndividualProfile,
  renderIndividualEdit,
};
