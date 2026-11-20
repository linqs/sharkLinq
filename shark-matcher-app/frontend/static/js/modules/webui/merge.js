import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Home from './home.js';
import * as Routes from './routes.js';
import * as Individual from './individual.js';
import * as Util from './util.js';
import * as Upload from './upload.js';

function init() {
  Routes.addRoute(/^merge$/, handlerMerge, 'merge');
}

function handlerMerge(path, params, context) {
  // Check path to not allow self merging.
  if (params['name1'] === params['name2']) {
    alert(`Cannot merge an ${Util.getDisplayName().toLowerCase()} with itself.`);
    history.back();
    return;
  }

  Home.handlerHome(path, params, context);

  // Tab content.
  let html = `
        <div class="tab-content merge">
            <div class="tab-title"></div>
        </div>
    `;
  let doc = document.createElement('div');
  doc.innerHTML = html;
  document.querySelector('.tab-content').replaceWith(doc.querySelector('.tab-content.merge'));

  Home.renderTitle('merge', context);

  // Results content.
  html = `
        <div class="tab-results">
            <div class="merge input-column left"></div>
            <div class="merge input-column right"></div>
            <div class="merge output-column"></div>
        </div>
    `;
  doc = document.createElement('div');
  doc.innerHTML = html;
  document.querySelector('.tab-results').replaceWith(doc.querySelector('.tab-results'));

  // Populate results.
  if (Object.keys(params).length !== 0) {
    Individual.executeIndividualSearch(
        path, {'name': [params['name1'], params['name2']]}, context,
        renderMerge);
  }
}

// API execution functions.
function executeMerge(individuals, context) {
  const previewIndividual = individuals[2];
  const referenceId = previewIndividual.reference;

  // Pick the major and minor individual.
  let majorIndividual = individuals[0];
  let minorIndividual = individuals[1];
  for (const associatedIndividual of individuals[1].associated_individuals) {
    if (referenceId === associatedIndividual?.id) {
      majorIndividual = individuals[1];
      minorIndividual = individuals[0];
    }
  }

  // Compact the objects for transit. Preserve tags/labels because Merge natively reconstructs them!
  const compactedPreview = Util.compactAssociatedObjects(previewIndividual, true);
  const compactedMajor = Util.compactAssociatedObjects(majorIndividual, true);
  const compactedMinor = Util.compactAssociatedObjects(minorIndividual, true);

  // Check if in proposing mode or editing mode.
  const currentMode = Core.getCurrentUserMode();
  if (currentMode === 'proposing') {
    // Represent the individuals individual fully.
    compactedMajor.id = compactedMajor.reference;
    compactedMajor.name = compactedMajor.reference;
    compactedPreview.id = compactedPreview.reference;

    // Post the proposal.
    SharkMatcher.Proposal
        .post_proposal(
            null, compactedMajor, compactedPreview, 'merge', 'individual', 'open')
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
  } else if (Core.canEdit() || (currentMode === 'demo' && Core.canMerge())) {
    // Update database and log activity.
    SharkMatcher.Individual
        .merge_individuals(compactedMajor.id, compactedMinor.id, compactedPreview)
        .then(function(result) {
          window.sharkMatcher?.demo?.showMergeResult(result);
          const fetchIndividual = result['fetch_individual'];
          // Navigate to profile page of newly merged individual, may be better to
          // navigate to previous search page?
          const newPath = Core.formHashPath('individual', {
            'name': fetchIndividual.name,
          });
          window.location.hash = newPath;
        })
        .catch(function(error) {
          console.log('Error merging individual.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error merging ${Util.getDisplayName().toLowerCase()}: ${message}`);
          }
        });
  }
}

function previewExecuteIndividualSave(individuals, context) {
  let previewIndividual = individuals[2];
  // Get and validate inputs. TODO, allow like"ish" inputs, validate inputs.
  const referenceInput = document.querySelector('.reference-input').value;
  const sexInput = document.querySelector('.sex-input').value;
  const lengthInput = document.querySelector('.length-input').value;

  // Update merge individual preview and refresh page.
  if (referenceInput != '' && referenceInput != previewIndividual.reference) {
    // Rebuild the merged individual if merging into a different source individual.
    previewIndividual = buildPreviewIndividual(
        individuals[0], individuals[1], referenceInput, previewIndividual);
    individuals = [individuals[0], individuals[1], previewIndividual];
  }
  if (sexInput != '' && sexInput != previewIndividual.sex) {
    previewIndividual.sex = sexInput;
  }
  if (lengthInput != '' && parseFloat(lengthInput) != previewIndividual.length) {
    previewIndividual.length = parseFloat(lengthInput);
  }

  renderMergeOutput(individuals, context);
}

function previewExecuteObjectSave(object, individuals, context, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }

  let previewIndividual;
  if (kwargs['page'] === 'merge') {
    previewIndividual = individuals[2];
  } else if (kwargs['page'] === 'individual') {
    previewIndividual = individuals[0];
  } else if (kwargs['page'] === 'upload') {
    previewIndividual = individuals[kwargs['index']];
  }

  // Get and validate inputs.
  const nameInput = container.querySelector(`.${object}.name-input`).value;
  let typeInput = null;
  let timeInput = null;
  try {
    if (Util.typedObjects.includes(object)) {
      typeInput = container.querySelector(`.${object}.type-input`).value;
    }
    if (Util.timedObjects.includes(object)) {
      timeInput = Util.inputToTimestamp(
          document.querySelector(`.${object}.time-input`).value, false);
    }
  } catch {
    console.log('No type or time input found.');
  }

  if (nameInput == '') {
    alert('No name provided.');
    return;
  } else if (Util.validName(nameInput) === false) {
    alert('Name input contains disallowed characters.');
    return;
  }

  // Update merge individual preview and refresh page.
  if (object === 'alternate_name') {
    const tempObjectId = -1 * (previewIndividual.alternate_names.length + 1);
    const newObject = {
      'id': tempObjectId,
      'individual': previewIndividual.id,
      'name': nameInput,
      'type': typeInput,
      'time': timeInput,
    };
    previewIndividual.alternate_names.push(newObject);
  } else if (object === 'tag') {
    const tempObjectId = -1 * (previewIndividual.tags.length + 1);
    const newObject = {
      'id': tempObjectId,
      'individual': previewIndividual.id,
      'name': nameInput,
      'type': typeInput,
      'time': timeInput,
    };
    previewIndividual.tags.push(newObject);
  } else if (object === 'label') {
    const tempObjectId = -1 * (previewIndividual.labels.length + 1);
    const newObject = {
      'id': tempObjectId,
      'individual': previewIndividual.id,
      'name': nameInput,
      'type': typeInput,
      'time': timeInput,
    };
    previewIndividual.labels.push(newObject);
  }

  if (kwargs['page'] === 'merge') {
    renderMergeOutput(individuals, context);
  } else if (kwargs['page'] === 'upload') {
    // Save all changes made to the upload item, then re-render.
    Upload.renderUploadResult(individuals, context, kwargs);
  }
}

function previewExecuteObjectDelete(
    event, individuals, object, context, kwargs = {}) {
  const objectId = parseInt(event.target.getAttribute('object-id'));

  let previewIndividual;
  if (kwargs['page'] === 'merge') {
    previewIndividual = individuals[2];
  } else if (kwargs['page'] === 'individual') {
    previewIndividual = individuals[0];
  } else if (kwargs['page'] === 'upload') {
    previewIndividual = individuals[kwargs['index']];
  }

  // Update merge individual preview and refresh page.
  const newArray = [];
  if (object === 'alternate_name') {
    for (const index in previewIndividual.alternate_names) {
      if (previewIndividual.alternate_names[index].id !== objectId) {
        newArray.push(previewIndividual.alternate_names[index]);
      }
    }
    previewIndividual.alternate_names = newArray;
  } else if (object === 'tag') {
    for (const index in previewIndividual.tags) {
      if (previewIndividual.tags[index].id !== objectId) {
        newArray.push(previewIndividual.tags[index]);
      }
    }
    previewIndividual.tags = newArray;
  } else if (object === 'label') {
    for (const index in previewIndividual.labels) {
      if (previewIndividual.labels[index].id !== objectId) {
        newArray.push(previewIndividual.labels[index]);
      }
    }
    previewIndividual.labels = newArray;
  }

  if (kwargs['page'] === 'merge') {
    renderMergeOutput(individuals, context);
  } else if (kwargs['page'] === 'upload') {
    Upload.renderUploadResult(individuals, context, kwargs);
  }
}

// HTML render functions.
function renderMerge(individuals, context, kwargs = {}) {
  // Build merged individual preview.
  const majorReferenceIndividual = Util.individualGetReferenceIndividual(individuals[0]);
  const previewIndividual =
    buildPreviewIndividual(individuals[0], individuals[1], majorReferenceIndividual.name);
  individuals.push(previewIndividual);

  // Populate input individual section.
  renderMergeInput(individuals, context);

  // Populate output individual section.
  renderMergeOutput(individuals, context);
}

function renderMergeInput(individuals, context, kwargs = {}) {
  for (let i = 0; i < individuals.length - 1; i++) {
    let selector = 'left';
    if (i === 1) {
      selector = 'right';
    }
    // Clear old result.
    document.querySelector(`.merge.input-column.${selector}`).innerHTML = '';

    const individual = individuals[i];

    // Handle error.
    if (individual === undefined) {
      const html = `
            <div>
                <h3>${Util.getDisplayName()}s not found.</h3>
            </div>
            `;
      const doc = document.createElement('div');
      doc.innerHTML = html;
      document.querySelector(`.merge.input-column.${selector}`)
          .appendChild(doc.querySelector('div'));
      return;
    }

    const html = `
        <div class="individual">
            <div class="input profile"></div>

            <div class="input gallery">
                <h4>Associated Images</h4>
                <div class="result-gallery"></div>
            </div>
        </div>
        `;

    // Add child elements and functinality.
    let doc = document.createElement('div');
    doc.innerHTML = html;
    doc = Individual.renderIndividualProfile(
        [individual], context,
        {'container': doc, 'selector': '.input.profile', 'page': 'individual'});
    const renderResult = Home.renderGallery(
        [individual], context,
        {'container': doc, 'selector': '.result-gallery', 'page': 'merge'});
    doc = renderResult[0];
    document.querySelector(`.merge.input-column.${selector}`)
        .appendChild(doc.querySelector('.individual'));
  }
}

function renderMergeOutput(individuals, context, kwargs = {}) {
  // Clear old result.
  document.querySelector('.merge.output-column').innerHTML = '';

  const previewIndividual = individuals[2];

  // Handle error.
  if (previewIndividual === undefined) {
    const html = `
        <div>
            <h3>Error merging ${Util.getDisplayName().toLowerCase()}.</h3>
        </div>
        `;
    const doc = document.createElement('div');
    doc.innerHTML = html;
    document.querySelector(`.merge.output-column`)
        .appendChild(doc.querySelector('div'));
    return;
  }

  const html = `
    <div class="individual">
        <div class="output profile"></div>

        <button class="output save pill-btn" style="align-self: center; margin: 10px 0;">Save Merge</button>

        <div class="output edit"></div>

        <div class="output gallery">
            <h4>Associated Images</h4>
            <div class="result-gallery"></div>
        </div>
    </div>
    `;

  // Add child elements and functinality.
  let doc = document.createElement('div');
  doc.innerHTML = html;
  const saveButton = doc.querySelector('.output.save');

  // Disable save button when in viewing mode
  const currentMode = Core.getCurrentUserMode();
  if (currentMode !== 'proposing' && !Core.canMerge()) {
    if (saveButton) saveButton.disabled = true;
  }

  saveButton.addEventListener('click', function() {
    // Merge individuals.
    executeMerge(individuals, context);
  });
  doc = Individual.renderIndividualProfile(
      [previewIndividual], context,
      {'container': doc, 'selector': '.output.profile', 'page': 'merge_output'});
  const renderResult = Home.renderGallery(
      [previewIndividual], context,
      {'container': doc, 'selector': '.result-gallery', 'page': 'merge'});
  doc = renderResult[0];
  document.querySelector('.merge.output-column')
      .appendChild(doc.querySelector('.individual'));

  if (currentMode !== 'demo') {
    Individual.renderIndividualEdit(
        individuals, context, {'selector': '.output.edit', 'page': 'merge_output'});
    Home.renderObjectList(
        individuals, 'alternate_name', context,
        {'selector': '.alternate_name .object-list', 'page': 'merge'});
    Home.renderObjectList(
        individuals, 'tag', context, {'selector': '.tag .object-list', 'page': 'merge'});
    Home.renderObjectList(
        individuals, 'label', context,
        {'selector': '.label .object-list', 'page': 'merge'});
  }
}

function buildPreviewIndividual(
    individual1, individual2, referenceInput, prevPreview = null) {
  // Find which individual is the major and which is the minor individual.
  // Major individual is the who the reference ID is associated with.
  let majorIndividual = individual1;
  let minorIndividual = individual2;
  const sourceIds = [individual1.source, individual2.source];
  let referenceId;

  for (const associatedIndividual of individual1.associated_individuals) {
    if (referenceInput.toString() === associatedIndividual.name) {
      referenceId = associatedIndividual.id;
    }
  }
  for (const associatedIndividual of individual2.associated_individuals) {
    if (referenceInput.toString() === associatedIndividual.name) {
      majorIndividual = individual2;
      minorIndividual = individual1;
      referenceId = associatedIndividual.id;
    }
  }

  const previewIndividual = {};
  previewIndividual.id = null;
  previewIndividual.name = referenceInput.toString();

  // Carry over the major individual's properties, if unknown, use the minor individual's
  // properties. NOTE, this is overridden by the input parameters in
  // previewExecuteIndividualSave!
  previewIndividual.length = majorIndividual.length;
  if (previewIndividual.length === Util.UNKNOWN_VALUE || previewIndividual.length === null) {
    previewIndividual.length = minorIndividual.length;
  }
  previewIndividual.sex = majorIndividual.sex;
  if (previewIndividual.sex === Util.UNKNOWN_VALUE || previewIndividual.sex === null) {
    previewIndividual.sex = minorIndividual.sex;
  }
  previewIndividual.source = null;
  previewIndividual.reference = referenceId;

  // Populate associated individual list for the preview individual, which needs to be:
  // A placeholder for the preview individual itself.
  // All associated media individuals from the major and minor individual.
  const majorMediaIndividuals = majorIndividual.associated_individuals.slice(1);
  const minorMediaIndividuals = minorIndividual.associated_individuals.slice(1);
  previewIndividual.associated_individuals = [null];
  previewIndividual.associated_individuals =
    previewIndividual.associated_individuals.concat(majorMediaIndividuals);
  previewIndividual.associated_individuals =
    previewIndividual.associated_individuals.concat(minorMediaIndividuals);

  previewIndividual.associated_medias =
    majorIndividual.associated_medias.concat(minorIndividual.associated_medias);

  // Carry over all associated objects.
  let allAlternateNames =
    majorIndividual.alternate_names.concat(minorIndividual.alternate_names);
  let allTags = majorIndividual.tags.concat(minorIndividual.tags);
  let allLabels = majorIndividual.labels.concat(minorIndividual.labels);

  if (prevPreview !== null) {
    // Determine which original items the user kept (not deleted) and which were newly added.
    const keptAlternateNameIds = new Set(
        prevPreview.alternate_names.filter((i) => i.id >= 0).map((i) => i.id));
    const keptTagIds = new Set(
        prevPreview.tags.filter((i) => i.id >= 0).map((i) => i.id));
    const keptLabelIds = new Set(
        prevPreview.labels.filter((i) => i.id >= 0).map((i) => i.id));

    // Only keep source items the user didn't delete.
    allAlternateNames = allAlternateNames.filter((item) => keptAlternateNameIds.has(item.id));
    allTags = allTags.filter((item) => keptTagIds.has(item.id));
    allLabels = allLabels.filter((item) => keptLabelIds.has(item.id));

    // Re-add newly user-added items (identified by temporary negative IDs).
    allAlternateNames = allAlternateNames.concat(
        prevPreview.alternate_names.filter((i) => i.id < 0));
    allTags = allTags.concat(prevPreview.tags.filter((i) => i.id < 0));
    allLabels = allLabels.concat(prevPreview.labels.filter((i) => i.id < 0));
  }

  // Keep only one copy of each name/tag/label.
  // Unmatched label is removed after merge and duplicate labels are kept once.
  previewIndividual.alternate_names = [];
  previewIndividual.labels = [];
  previewIndividual.tags = [];

  const alternateNameTracker = [];
  const tagTracker = [];
  const labelTracker = [];
  for (const item of allAlternateNames) {
    if (!alternateNameTracker.includes(item.name) &&
      (sourceIds.includes(item.individual) || item.id < 0)) {
      previewIndividual.alternate_names.push(item);
      alternateNameTracker.push(item.name);
    }
  }
  for (const item of allLabels) {
    if ((item.name !== 'unmatched') &&
      (!labelTracker.includes(item.name) &&
        (sourceIds.includes(item.individual) || item.id < 0))) {
      previewIndividual.labels.push(item);
      labelTracker.push(item.name);
    }
  }
  for (const item of allTags) {
    if (!tagTracker.includes(item.name) &&
      (sourceIds.includes(item.individual) || item.id < 0)) {
      previewIndividual.tags.push(item);
      tagTracker.push(item.name);
    }
  }

  return previewIndividual;
}

export {
  init,
  previewExecuteObjectSave,
  previewExecuteIndividualSave,
  previewExecuteObjectDelete,
};
