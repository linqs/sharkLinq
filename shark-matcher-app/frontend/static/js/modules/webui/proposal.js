import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Home from './home.js';
import * as Routes from './routes.js';
import * as Search from './search.js';
import * as Util from './util.js';

function init() {
  Routes.addRoute(/^proposals$/, handlerProposals, 'proposals');
}

// TODO dynamically populate user input section.
// ADD min and max time filters, original/final filters.
function handlerProposals(path, params, context) {
  Home.handlerHome(path, params, context);

  // Open the sort display on startup.
  const sortDisplay = '';

  // Populate input fields by default with what's already on the current path.
  let keywordsValue = '';
  if (params.keywords && params.keywords !== '') {
    keywordsValue = `value="${Util.escapeHtml(params.keywords)}"`;
  }

  // Tab content.
  let html = `
        <div class="tab-content proposals">
            <div class="tab-title"></div>
            ${Util.buildSmartSearchBar('proposals', 'Search proposals by keywords...', keywordsValue, 'keywords-input', sortDisplay)}
        </div>
    `;
  let doc = new DOMParser().parseFromString(html, 'text/html');

  // Search bar logic
  Util.bindSmartSearchEvents(doc, 'proposals');

  const searchInput = doc.querySelector('.smart-search-input');
  searchInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      const currentPopover = document.getElementById('proposals-filter-popover');
      if (currentPopover) {
        currentPopover.classList.remove('active');
      }
      triggerProposalSearch();
    }
  });

  document.querySelector('.tab-content')
      .replaceWith(doc.querySelector('.tab-content.proposals'));

  Home.renderTitle('proposals', context);
  renderProposalFilters(
      path, params, context,
      {'selector': '.filter-popover', 'page': 'proposals'});
  Search.renderSortInput(
      path, params, context,
      {'selector': '.input-section.sort', 'page': 'proposals', 'display': sortDisplay});

  // Results content.
  html = `
        <div class="tab-results">
            <div class="results-container"></div>
            <button class="load-more-button" style="display: none;">
              Load More
            </button>
        </div>
    `;
  doc = new DOMParser().parseFromString(html, 'text/html');
  document.querySelector('.tab-results')
      .replaceWith(doc.querySelector('.tab-results'));

  // Populate results.
  if (Object.keys(params).length !== 0) {
    executeProposalSearch(path, params, context, renderProposalResults);
  }
}

// Path update functions.
function triggerProposalSearch() {
  // Search and filter inputs.
  const keywordInput =
    document.querySelector('.smart-search-input.proposals').value;
  const objectInput = document.querySelector('.proposals.object-input').value;
  const actionInput = document.querySelector('.proposals.action-input').value;
  const authorInput = document.querySelector('.proposals.author-input').value;
  let minTimeInput = document.querySelector('.proposals.min_time-input').value;
  minTimeInput = Util.inputToTimestamp(minTimeInput, false);
  let maxTimeInput = document.querySelector('.proposals.max_time-input').value;
  maxTimeInput = Util.inputToTimestamp(maxTimeInput, false);
  let statusInput = document.querySelector('.proposals.status-input').value;
  if (statusInput === '') {
    statusInput = 'all';
  }

  // Sorting inputs.
  let attributeInput = document.querySelector('.sort.attribute-input').value;
  let orderInput = document.querySelector('.sort.order-input').value;

  // Default sort input
  if (attributeInput === '') {
    attributeInput = 'time';
  }
  if (orderInput === '') {
    orderInput = 'descending';
  }

  const newPath = Core.formHashPath('proposals', {
    'min_time': minTimeInput,
    'max_time': maxTimeInput,
    'object': objectInput,
    'action': actionInput,
    'status': statusInput,
    'original': '',
    'final': '',
    'author': authorInput,
    'ids': '',
    'keywords': keywordInput,
    'attribute': attributeInput,
    'order': orderInput,
  });
  window.location.hash = newPath;
}

// API execution functions.
function executeProposalSearch(
    path, params, context, renderFunction, kwargs = {}) {
  let apiStatus = params['status'];
  if (apiStatus === 'all') {
    apiStatus = '';
  } else if (!apiStatus) {
    apiStatus = 'open';
  }

  if (Util.PAGINATE) {
    const page = kwargs['page'] || 1;
    const attribute = params['attribute'] || 'id';
    const order = params['order'] || 'descending';
    SharkMatcher.Proposal.filter_proposals(
        params['min_time'], params['max_time'], params['object'],
        params['action'], params['original'], params['final'], apiStatus,
        params['author'], params['ids'], params['keywords'],
        true, page, Util.PAGE_SIZE, attribute, order)
        .then(function(result) {
          kwargs['page'] = page;
          renderFunction(result, context, kwargs);
        })
        .catch(function(error) {
          console.log('Error executing proposal search.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error executing proposal search: ${message}`);
          }
          return Core.redirectHome();
        });
  } else {
    SharkMatcher.Proposal.filter_proposals(
        params['min_time'], params['max_time'], params['object'],
        params['action'], params['original'], params['final'], apiStatus,
        params['author'], params['ids'], params['keywords'])
        .then(function(result) {
          kwargs['load_index'] = 1;
          renderFunction(result, context, kwargs);
        })
        .catch(function(error) {
          console.log('Error executing proposal search.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error executing proposal search: ${message}`);
          }
          return Core.redirectHome();
        });
  }
}

function executeProposalArchive(proposalId, currentStatus, context) {
  // Toggle the proposal status between "open" and "archive"
  const newStatus = currentStatus === 'archive' ? 'open' : 'archive';

  // Update the proposal in the database
  SharkMatcher.Proposal
      .post_proposal(proposalId, null, null, null, null, newStatus)
      .then(function(result) {
        console.log(
            `Successfully updated proposal ${proposalId} status to ${newStatus}`);
        // Refresh the page to show the updated status.
        window.location.reload();
      })
      .catch(function(error) {
        console.log('Error updating proposal status.');
        const message = Util.handleAPIError(error);
        if (message) {
          alert(`Error updating proposal status: ${message}`);
        }
      });
}

function executeProposalDelete(proposalId, context) {
  // Deletes require confirmation
  const confirmation =
    confirm('Are you sure you want to delete this proposal?');
  if (confirmation === false) {
    return;
  }

  // Delete the proposal from the database
  SharkMatcher.Proposal.delete_proposal(proposalId)
      .then(function(result) {
        // Refresh the page to show the updated proposals list
        window.location.reload();
      })
      .catch(function(error) {
        console.log('Error deleting proposal.');
        const message = Util.handleAPIError(error);
        if (message) {
          alert(`Error deleting proposal: ${message}`);
        }
      });
}

// HTML render functions.
// Render the filter section of the search area.
function renderProposalFilters(path, params, context, kwargs = {}) {
  // Populate input fields by default with what's already on the current path.
  let objectPlaceholder = 'all';
  if (params.object && params.object !== '') {
    objectPlaceholder = params.object;
  }
  if (objectPlaceholder === 'individual') {
    objectPlaceholder = {value: 'individual', text: Util.getDisplayName().toLowerCase() + 's'};
  }
  let statusPlaceholder = 'open';
  if (params.status && params.status !== '') {
    statusPlaceholder = params.status;
  }
  let actionPlaceholder = 'all';
  if (params.action && params.action !== '') {
    actionPlaceholder = params.action;
  }
  let authorPlaceholder = 'all';
  if (params.author && params.author !== '') {
    authorPlaceholder = params.author;
  }
  let minTimeValue = '';
  if (params.min_time && params.min_time !== '') {
    minTimeValue = `value="${Util.timestampToInput(params.min_time)}"`;
  }
  let maxTimeValue = '';
  if (params.max_time && params.max_time !== '') {
    maxTimeValue = `value="${Util.timestampToInput(params.max_time)}"`;
  }

  const html = `
    <div class="filter-popover" id="${kwargs['page']}-filter-popover">
        <div class="input-item">
            <label for="proposals-status-input">Status: </label>
            <select
                class="proposals status-input"
                id="proposals-status-input">
            </select>
        </div>
        <div class="input-item">
            <label for="proposals-object-input">Object: </label>
            <select
                class="proposals object-input"
                id="proposals-object-input">
            </select>
        </div>
        <div class="input-item">
            <label for="proposals-action-input">Action: </label>
            <select
                class="proposals action-input"
                id="proposals-action-input">
            </select>
        </div>
        <div class="input-item">
            <label for="proposals-author-input">Author: </label>
            <select
                class="proposals author-input"
                id="proposals-author-input">
            </select>
        </div>
        <div class="input-item">
        <label for="proposals-min_time-input">Start Date: </label>
            <input
                class="proposals min_time-input"
                id="proposals-min_time-input"
                type="date"
                ${minTimeValue}/>
            </div>
        <div class="input-item">
            <label for="proposals-max_time-input">End Date: </label>
            <input
                class="proposals max_time-input"
                id="proposals-max_time-input"
                type="date"
                ${maxTimeValue}/>
        </div>
        <button class="clear-filters-button" style="grid-column: 1 / -1; justify-self: center; margin-top: 15px;">Clear Filters</button>
    </div>
    `;

  const doc = new DOMParser().parseFromString(html, 'text/html');

  const clearButton = doc.querySelector('.clear-filters-button');
  clearButton.addEventListener('click', function() {
    Util.clearUrlParams();
  });

  document.querySelector(kwargs['selector'])
      .replaceWith(doc.querySelector('.filter-popover'));

  // Populate selector options
  let selectOptions = [
    objectPlaceholder, `all`, {value: 'individual', text: Util.getDisplayName().toLowerCase() + 's'}, `media`, `nicknames`, `tags`,
    `labels`,
  ];
  Util.renderSelectOptions(
      selectOptions, context, {'selector': '.proposals.object-input'});
  selectOptions = [statusPlaceholder, `all`, `open`, `archive`];
  Util.renderSelectOptions(
      selectOptions, context, {'selector': '.proposals.status-input'});
  selectOptions = [
    actionPlaceholder, `all`, `create`, `update`, `merge`, `unmatch`,
    `delete`,
  ];
  Util.renderSelectOptions(
      selectOptions, context, {'selector': '.proposals.action-input'});

  // Determine the text value of the pre-selected author parameter for the placeholder.
  const authorPlaceholderText = Util.getAuthorUsername(params.author);
  Util.renderSelectOptions(
      [authorPlaceholder, 'all', ...Util.getLocalUsersFormatted()],
      context,
      {'selector': '.proposals.author-input', 'placeholderText': authorPlaceholderText});

  Util.updateFilterIndicator(kwargs['page']);
}

function renderProposalResults(result, context, kwargs = {}) {
  if (Util.PAGINATE) {
    const page = kwargs['page'];
    const isFirstPage = page === 1;
    const items = result.results;

    if (isFirstPage) {
      document.querySelector('.results-container').innerHTML =
        Util.renderResultCount(result.total_count);
    }

    // Reset load-more button on every page to bind the correct next-page handler.
    const loadMoreButton = document.querySelector('.load-more-button');
    const newLoadMoreButton = loadMoreButton.cloneNode(true);
    loadMoreButton.replaceWith(newLoadMoreButton);
    newLoadMoreButton.style.display = result.has_more ? 'block' : 'none';
    newLoadMoreButton.addEventListener('click', function() {
      const [currentPath, currentParams] = Routes.parsePath(window.location.hash);
      kwargs['page'] = page + 1;
      executeProposalSearch(currentPath, currentParams, context, renderProposalResults, kwargs);
    });

    for (const proposal of items) {
      document.querySelector('.results-container')
          .appendChild(buildProposalItem(proposal, context));
    }
    return;
  }

  // Non-paginated path: result is a flat array, client-side slicing.
  let results = result;
  const path = Routes.parsePath(window.location.hash);
  const params = path[1];
  let loadIndex = kwargs['load_index'];

  if (loadIndex === 1) {
    const effectiveOrder = (params.order && params.order !== '') ? params.order : 'descending';
    const effectiveAttribute = (params.attribute && params.attribute !== '') ? params.attribute : 'id';
    const order = effectiveOrder !== 'descending';
    results = Util.sortByAttribute(results, effectiveAttribute, order);
    kwargs['processed_results'] = results;

    document.querySelector('.results-container').innerHTML =
      Util.renderResultCount(results.length);

    const loadMoreButton = document.querySelector('.load-more-button');
    const newLoadMoreButton = loadMoreButton.cloneNode(true);
    loadMoreButton.replaceWith(newLoadMoreButton);
    newLoadMoreButton.addEventListener('click', function() {
      kwargs['load_index'] = loadIndex += 1;
      renderProposalResults(results, context, kwargs);
    });
  } else {
    results = kwargs['processed_results'];
  }

  const bounds = Util.computeResultBounds(results.length, loadIndex);
  for (let i = bounds[0]; i < bounds[1]; i++) {
    document.querySelector('.results-container')
        .appendChild(buildProposalItem(results[i], context));
  }
}

function buildProposalItem(proposal, context) {
  const proposalTime = Util.timestampToString(proposal.time);
  const proposalIDs = {};
  for (const type of ['original', 'final']) {
    const value = proposal[type];
    proposalIDs[type] = (value !== -1 && value !== null && value !== undefined) ?
      value : 'Empty';
  }
  const {original: proposalOriginal, final: proposalFinal} = proposalIDs;
  const resultDescription = Util.renderResultDescription(proposal.notes);
  const statusClass = proposal.status === 'archive' ? 'proposal-archived' : 'proposal-open';
  const buttonText = proposal.status === 'archive' ? 'Open' : 'Archive';
  const buttonClass = proposal.status === 'archive' ? 'proposal-open-btn' : 'proposal-archive-btn';

  const html = `
      <div class="results-item ${statusClass}">
          <div class="result-profile">
              <h3>Proposal ID: ${proposal.id}</h3>
              <div class="info-row"><span class="label">Author:</span> <span class="value">${Util.getAuthorUsername(proposal.author)}</span></div>
              <div class="info-row"><span class="label">Time:</span> <span class="value">${proposalTime}</span></div>
              <div class="info-row"><span class="label">Proposal Type:</span> <span class="value">${Util.displayValue(proposal.action)}</span></div>
              <div class="info-row"><span class="label">Proposal Object:</span> <span class="value">${proposal.object === 'individual' ? Util.getDisplayName().toLowerCase() : Util.displayValue(proposal.object)}</span></div>
              <div class="info-row"><span class="label">Proposal Status:</span> <span class="value">${Util.displayValue(proposal.status)}</span></div>
              <div class="info-row"><span class="label">Source Object ID:</span> <span class="value">${proposalOriginal}</span></div>
              <div class="info-row"><span class="label">Target Object ID:</span> <span class="value">${proposalFinal}</span></div>
              <div style="display: flex; gap: 10px; margin-top: 15px; justify-content: center;">
                  <button class="${buttonClass} pill-btn"
                   data-proposal-id="${proposal.id}"
                   data-current-status="${proposal.status}">${buttonText}
                  </button>
                  <button class="proposal-delete-btn pill-btn"
                   data-proposal-id="${proposal.id}">Delete</button>
              </div>
          </div>
          ${resultDescription}
      </div>
      `;
  const doc = new DOMParser().parseFromString(html, 'text/html');

  const archiveButton = doc.querySelector(`.${buttonClass}`);
  let userRoleValue = 0;
  const currentUser = Core.getContextUser();
  if (currentUser && currentUser.user_role) {
    userRoleValue = Core.roleDict[currentUser.user_role];
  }
  if (userRoleValue < 2) {
    archiveButton.disabled = true;
    archiveButton.title = 'Only admins can update proposals.';
  }
  archiveButton.addEventListener('click', function() {
    executeProposalArchive(proposal.id, proposal.status, context);
  });

  const deleteButton = doc.querySelector('.proposal-delete-btn');
  const canDelete = userRoleValue >= 1 &&
    Number(proposal.author) === Number(currentUser.user_id);
  if (!canDelete) {
    deleteButton.disabled = true;
    deleteButton.title = 'Can only delete one\'s own proposals.';
  }
  deleteButton.addEventListener('click', function() {
    executeProposalDelete(proposal.id, context);
  });

  return doc.querySelector('.results-item');
}

export {
  init,
  triggerProposalSearch,
};
