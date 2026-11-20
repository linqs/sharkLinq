import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Home from './home.js';
import * as Individual from './individual.js';
// import * as Log from './log.js';
import * as Routes from './routes.js';
import * as Search from './search.js';
import * as Util from './util.js';

function init() {
  Routes.addRoute(/^match$/, handlerMatch, 'match');
}


function handlerMatch(path, params, context) {
  Home.handlerHome(path, params, context);

  // Method and model: use the URL param if explicitly provided, otherwise
  // default to the first available option. Stored params are not used —
  // they can be stale if the server has been reconfigured.
  if (!params.method) {
    const matchMethods = Util.getCachedMatchMethods();
    if (matchMethods.length > 0) params.method = matchMethods[0];
  }
  if (!params.model) {
    const loadedModels = Util.getCachedLoadedModels();
    if (loadedModels.length > 0) params.model = loadedModels[0];
  }

  let nameValue = '';
  if (params.name && params.name !== '') {
    nameValue = `value="${Util.escapeHtml(params.name)}"`;
  }
  // Tab content.
  let html = `
    <div class="tab-content match">
        <div class="tab-title"></div>
        ${Util.buildSmartSearchBar('match', 'Search by individual/image name...', nameValue, 'name-input', '')}
    </div>
    `;
  let doc = new DOMParser().parseFromString(html, 'text/html');

  // Search bar logic
  Util.bindSmartSearchEvents(doc, 'match');

  const searchInput = doc.querySelector('.smart-search-input');
  searchInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      const currentPopover = document.getElementById('match-filter-popover');
      if (currentPopover) {
        currentPopover.classList.remove('active');
      }
      triggerMatchSearch();
    }
  });

  document.querySelector('.tab-content')
      .replaceWith(doc.querySelector('.tab-content.match'));

  Home.renderTitle('match', context);
  Search.renderSearchFilters(
      path, params, context,
      {
        'selector': '.filter-popover', 'page': 'match',
      });

  // Both selectors are rendered by shared util helpers so method names and
  // model names are each defined in exactly one place.
  Util.renderModelSelector('match', params, context);
  Util.renderMethodSelector('match', params, context);

  Util.updateFilterIndicator('match');

  // Results content.
  html = `
        <div class="tab-results">
            <div class="match query-column"></div>
            <div class="match results-column"></div>
            <button class="load-more-button" style="display: none;">
              Load More
            </div>
        </div>
    `;
  doc = new DOMParser().parseFromString(html, 'text/html');
  document.querySelector('.tab-results')
      .replaceWith(doc.querySelector('.tab-results'));

  // Populate results.
  if (Object.keys(params).length !== 0 && params.name) {
    executeMatchSearch(path, params, context);
  }
}

function triggerMatchSearch() {
  const nameInput = document.querySelector('.smart-search-input.match').value;
  const methodInput = document.querySelector('.match.method-input').value;
  const modelInput = document.querySelector('.match.model-input').value;

  const filters = Util.collectSearchFilters();

  const newPath = Core.formHashPath('match', {
    ...filters,
    'name': nameInput,
    'method': methodInput,
    'model': modelInput,
    'keywords': '',
    'names': '',
    'medias': '',
  });
  window.location.hash = newPath;

  // Update the locally stored filter values for this tab. The match tab has no
  // keyword filter, so it is cleared rather than carried over.
  SharkMatcher.setFilters('match', {...filters, 'keywords': ''});
}

// API execution functions.
function executeMatchSearch(
    path, params, context, renderFunction, kwargs = {}) {
  // Always use the filter matches endpoint to unify logic.
  if (Object.keys(params).length >= 2 && params.name !== undefined) {
    SharkMatcher.Match.filter_matches(
        params.name, params.method, params.model, params.min_length, params.max_length,
        params.min_time, params.max_time, params.min_associated,
        params.max_associated, params.sex, params.locations, params.qualities, params.max_qualities,
        params.names, params.medias, params.alternate_names, params.labels,
        params.tags, params.allow_unknown, params.keywords)
        .then(function(matchResult) {
          renderQueryIndividual(matchResult, context);
          renderMatchResults(
              matchResult.query_individual, matchResult, context,
              {'load_index': 1, 'model': params.model});
        })
        .catch(function(error) {
          console.log('Error executing filter match search.');
          const message = Util.handleAPIError(error);
          if (message) {
            renderMatchError(`Error executing filter match search: ${message}`);
          }
        });
  } else {
    alert('Please provide a valid input.');
    return;
  }
}

// HTML render functions.
function renderMatchError(message) {
  const column = document.querySelector('.match.results-column');
  if (!column) return;
  const note = document.createElement('p');
  note.className = 'match-error';
  note.setAttribute('role', 'alert');
  note.textContent = message;
  column.replaceChildren(note);
}

function renderMatchResults(queryIndividual, results, context, kwargs = {}) {
  const resultList = results.match_suggestions;

  // Things to only do on the first load, setup container.
  if (kwargs['load_index'] === 1) {
    // Clear old result.
    const numResults = resultList.length;
    const resultMatches = results.matches;
    let customString = '';
    if (resultMatches === false) {
      customString = ' (No embeddings for query, showing all ' + Util.getDisplayName() + 's.)';
    }
    if (kwargs['model']) {
      customString += ` using ${kwargs['model']} model`;
    }
    const html = Util.renderResultCount(numResults, customString);
    document.querySelector('.match.results-column').innerHTML = html;
    const loadMoreButton = document.querySelector('.load-more-button');
    loadMoreButton.addEventListener('click', function() {
      kwargs['load_index'] += 1;
      renderMatchResults(queryIndividual, results, context, kwargs);
    });
  }

  // Handle search result limit.
  const bounds =
    Util.computeResultBounds(resultList.length, kwargs['load_index']);

  for (let i = bounds[0]; i < bounds[1]; i++) {
    const individual = resultList[i];

    if (individual === undefined) {
      const html = `
            <div class="results-item" object-id=${individual.id}>
                <h3>${Util.getDisplayName()} not found.</h3>
            </div>
            `;
      const doc = new DOMParser().parseFromString(html, 'text/html');
      document.querySelector('.match.results-column')
          .appendChild(doc.querySelector('.results-item'));
      return;
    }

    // Don't show query individual again.
    if (queryIndividual.source !== individual.source) {
      const html = `
            <div class="results-item" object-id=${individual.id}>
                <div class="result-profile"></div>
                <div class="result-gallery"></div>
            </div>
            `;
      let doc = new DOMParser().parseFromString(html, 'text/html');
      doc = Individual.renderIndividualProfile(
          [individual, queryIndividual], context,
          {
            'container': doc, 'selector': '.result-profile',
            'page': 'match_result',
          });
      const renderResult = Home.renderGallery(
          [individual, queryIndividual], context,
          {'container': doc, 'selector': '.result-gallery', 'page': 'match'});
      doc = renderResult[0];
      document.querySelector('.match.results-column')
          .appendChild(doc.querySelector('.results-item'));
    }
  }
}

function renderQueryIndividual(matchResult, context, kwargs = {}) {
  // Clear old result.
  let html = `<h3>Query Source ${Util.getDisplayName()}</h3>`;
  document.querySelector('.match.query-column').innerHTML = html;

  if (matchResult.query_individual === null) {
    const html = `
        <div>
            <h3>Query ${Util.getDisplayName()} not found.</h3>
        </div>
        `;
    const doc = new DOMParser().parseFromString(html, 'text/html');
    document.querySelector('.match.query-column')
        .appendChild(doc.querySelector('div'));
    return;
  }

  html = `
    <div class="individual">
        <div class="profile"></div>

        <div class="gallery">
            <div class="result-gallery"></div>
        </div>
    </div>
    `;

  // Add child elements and functinality.
  let doc = new DOMParser().parseFromString(html, 'text/html');
  doc = Individual.renderIndividualProfile(
      [matchResult.query_individual], context,
      {'container': doc, 'selector': '.profile', 'page': 'match_query'});
  const renderResult = Home.renderGallery(
      [matchResult.query_individual], context,
      {'container': doc, 'selector': '.result-gallery', 'page': 'match'});
  doc = renderResult[0];
  document.querySelector('.match.query-column')
      .appendChild(doc.querySelector('.individual'));
}

export {
  init,
  triggerMatchSearch,
};
