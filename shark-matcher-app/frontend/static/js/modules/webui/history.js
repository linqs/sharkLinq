import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Home from './home.js';
import * as Routes from './routes.js';
import * as Search from './search.js';
import * as Util from './util.js';

function init() {
  Routes.addRoute(/^history$/, handlerHistory, 'history');
}

// TODO dynamically populate user input section.

function handlerHistory(path, params, context) {
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
        <div class="tab-content history">
            <div class="tab-title"></div>
            ${Util.buildSmartSearchBar('history', 'Search history by keywords...', keywordsValue, 'keywords-input', sortDisplay)}
        </div>
    `;
  let doc = new DOMParser().parseFromString(html, 'text/html');

  // Search bar logic
  Util.bindSmartSearchEvents(doc, 'history');

  const searchInput = doc.querySelector('.smart-search-input');
  searchInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      const currentPopover = document.getElementById('history-filter-popover');
      if (currentPopover) {
        currentPopover.classList.remove('active');
      }
      triggerActionSearch();
    }
  });

  document.querySelector('.tab-content')
      .replaceWith(doc.querySelector('.tab-content.history'));

  Home.renderTitle('history', context);
  renderActionFilters(
      path, params, context,
      {'selector': '.filter-popover', 'page': 'history'});
  Search.renderSortInput(
      path, params, context,
      {
        'selector': '.input-section.sort', 'page': 'history',
        'display': sortDisplay,
      });

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
    executeActionSearch(path, params, context, renderActionResults);
  }
}

// Path update functions.
function triggerActionSearch() {
  // Search and filter inputs.
  const keywordInput = document.querySelector('.smart-search-input.history').value;
  const objectInput = document.querySelector('.history.object-input').value;
  const actionInput = document.querySelector('.history.action-input').value;
  const authorInput = document.querySelector('.history.author-input').value;
  let minTimeInput = document.querySelector('.history.min_time-input').value;
  minTimeInput = Util.inputToTimestamp(minTimeInput, false);
  let maxTimeInput = document.querySelector('.history.max_time-input').value;
  maxTimeInput = Util.inputToTimestamp(maxTimeInput, false);

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

  const newPath = Core.formHashPath('history', {
    'min_time': minTimeInput,
    'max_time': maxTimeInput,
    'object': objectInput,
    'action': actionInput,
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
// Search for result individuals that meet a specified filter condition.
function executeActionSearch(
    path, params, context, renderFunction, kwargs = {}) {
  if (Util.PAGINATE) {
    const page = kwargs['page'] || 1;
    const attribute = params['attribute'] || 'id';
    const order = params['order'] || 'descending';
    SharkMatcher.Action.filter_actions(
        params['min_time'], params['max_time'], params['object'],
        params['action'], params['original'], params['final'], params['author'],
        params['ids'], params['keywords'],
        true, page, Util.PAGE_SIZE, attribute, order)
        .then(function(result) {
          kwargs['page'] = page;
          renderFunction(result, context, kwargs);
        })
        .catch(function(error) {
          console.log('Error executing action search.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error executing action search: ${message}`);
          }
          return Core.redirectHome();
        });
  } else {
    SharkMatcher.Action.filter_actions(
        params['min_time'], params['max_time'], params['object'],
        params['action'], params['original'], params['final'], params['author'],
        params['ids'], params['keywords'])
        .then(function(result) {
          kwargs['load_index'] = 1;
          renderFunction(result, context, kwargs);
        })
        .catch(function(error) {
          console.log('Error executing action search.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error executing action search: ${message}`);
          }
          return Core.redirectHome();
        });
  }
}

// HTML render functions.
// Render the filter section of the search area.
function renderActionFilters(path, params, context, kwargs = {}) {
  // Populate input fields by default with what's already on the current path.
  let objectPlaceholder = 'all';
  if (params.object && params.object !== '') {
    objectPlaceholder = params.object;
  }
  if (objectPlaceholder === 'individual') {
    objectPlaceholder = {value: 'individual', text: Util.getDisplayName().toLowerCase() + 's'};
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
            <label for="history-object-input">Object: </label>
            <select
                class="history object-input"
                id="history-object-input">
            </select>
        </div>
        <div class="input-item">
            <label for="history-action-input">Action: </label>
            <select
                class="history action-input"
                id="history-action-input">
            </select>
        </div>
        <div class="input-item">
            <label for="history-author-input">Author: </label>
            <select
                class="history author-input"
                id="history-author-input">
            </select>
        </div>
        <div class="input-item">
        <label for="history-min_time-input">Start Date: </label>
            <input
                class="history min_time-input"
                id="history-min_time-input"
                type="date"
                ${minTimeValue}/>
            </div>
        <div class="input-item">
            <label for="history-max_time-input">End Date: </label>
            <input
                class="history max_time-input"
                id="history-max_time-input"
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
    objectPlaceholder, `all`, {value: 'individual', text: Util.getDisplayName().toLowerCase() + 's'}, `media`, `nicknames`, `tags`, `labels`];
  Util.renderSelectOptions(
      selectOptions, context, {'selector': '.history.object-input'});
  selectOptions = [
    actionPlaceholder, `all`, `create`, `update`, `merge`, `unmatch`, `delete`];
  Util.renderSelectOptions(
      selectOptions, context, {'selector': '.history.action-input'});

  if (params.author === '' || params.author === 'all') {
    Util.renderSelectOptions(
        [authorPlaceholder, 'all', ...Util.getLocalUsersFormatted()],
        context,
        {'selector': '.history.author-input'},
    );
  } else {
    // Determine the text value of the pre-selected author parameter for the placeholder.
    const authorPlaceholderText = Util.getAuthorUsername(params.author);
    Util.renderSelectOptions(
        [authorPlaceholder, 'all', ...Util.getLocalUsersFormatted()],
        context,
        {'selector': '.history.author-input', 'placeholderText': authorPlaceholderText},
    );
  }

  Util.updateFilterIndicator(kwargs['page']);
}

function renderActionResults(result, context, kwargs = {}) {
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
      executeActionSearch(currentPath, currentParams, context, renderActionResults, kwargs);
    });

    for (const action of items) {
      document.querySelector('.results-container')
          .appendChild(buildActionItem(action));
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
      renderActionResults(results, context, kwargs);
    });
  } else {
    results = kwargs['processed_results'];
  }

  const bounds = Util.computeResultBounds(results.length, loadIndex);
  for (let i = bounds[0]; i < bounds[1]; i++) {
    document.querySelector('.results-container')
        .appendChild(buildActionItem(results[i]));
  }
}

function buildActionItem(action) {
  const actionTime = Util.timestampToString(action.time);
  const actionIds = {};
  for (const type of ['original', 'final']) {
    actionIds[type] = (action[type] !== -1 && action[type] !== null && action[type] !== undefined)
      ? action[type] : 'Empty';
  }
  const {original: actionOriginal, final: actionFinal} = actionIds;
  const resultDescription = Util.renderResultDescription(action.notes);
  const html = `
      <div class="results-item">
          <div class="result-profile">
              <h3>Action ID: ${action.id}</h3>
              <div class="info-row"><span class="label">Author:</span> <span class="value">${Util.getAuthorUsername(action.author)}</span></div>
              <div class="info-row"><span class="label">Time:</span> <span class="value">${actionTime}</span></div>
              <div class="info-row"><span class="label">Action Type:</span> <span class="value">${Util.displayValue(action.action)}</span></div>
              <div class="info-row"><span class="label">Action Object:</span> <span class="value">${action.object === 'individual' ? Util.getDisplayName().toLowerCase() : Util.displayValue(action.object)}</span></div>
              <div class="info-row"><span class="label">Source Object ID:</span> <span class="value">${actionOriginal}</span></div>
              <div class="info-row"><span class="label">Target Object ID:</span> <span class="value">${actionFinal}</span></div>
          </div>
          ${resultDescription}
      </div>
      `;
  return new DOMParser().parseFromString(html, 'text/html').querySelector('.results-item');
}

export {
  init,
  triggerActionSearch,
};
