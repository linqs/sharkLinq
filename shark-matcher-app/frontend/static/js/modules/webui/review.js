import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Home from './home.js';
import * as Individual from './individual.js';
import * as Routes from './routes.js';
import * as Search from './search.js';
import * as Util from './util.js';


const REVIEW_TYPES = [
  {value: 'unmatch_suggestions', label: 'Unmatch Suggestions'},
  {value: 'merge_suggestions', label: 'Merge Suggestions'},
];

const DEFAULT_REVIEW_TYPE = 'unmatch_suggestions';
const REVIEW_ENABLED = false;

function init() {
  Routes.addRoute(/^review$/, handlerReview, 'review');
}

function handlerReview(path, params, context) {
  Home.handlerHome(path, params, context);

  // Keep direct navigation unavailable until the review tab is complete.
  if (!REVIEW_ENABLED) {
    document.querySelector('.content').innerHTML = `
      <h2>Page Not Found</h2>
      <p>The review page is not currently available.</p>
    `;
    return;
  }

  const sortDisplay = '';

  let keywordsValue = '';
  if (params.keywords && params.keywords !== '') {
    keywordsValue = `value="${Util.escapeHtml(params.keywords)}"`;
  }

  let html = `
        <div class="tab-content review">
            <div class="tab-title"></div>
            ${Util.buildSmartSearchBar('review', 'Search by keywords...', keywordsValue, '', sortDisplay)}
        </div>
    `;
  let doc = document.createElement('div');
  doc.innerHTML = html;

  Util.bindSmartSearchEvents(doc, 'review');

  const searchInput = doc.querySelector('.smart-search-input');
  searchInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      const currentPopover = document.getElementById('review-filter-popover');
      if (currentPopover) {
        currentPopover.classList.remove('active');
      }
      triggerReviewSearch();
    }
  });

  document.querySelector('.tab-content')
      .replaceWith(doc.querySelector('.tab-content.review'));

  Home.renderTitle('review', context);
  Search.renderSearchFilters(
      path, params, context,
      {
        'selector': '.filter-popover', 'page': 'review',
      });

  // Reuse the match tab's model selector so the active model is visible and
  // switchable from the review filters. Populates/refreshes from the inference
  // service's loaded_models list.
  Util.renderModelSelector('review', params, context);

  // Review-type pill selector, rendered into the same .input-section.sort slot
  // used by the search tab's sort pill so the two tabs line up visually.
  renderReviewTypeSelector(params);

  // Results content.
  html = `
        <div class="tab-results">
            <div class="results-container"></div>
            <button class="load-more-button" style="display: none;">
              Load More
            </button>
        </div>
    `;
  doc = document.createElement('div');
  doc.innerHTML = html;
  document.querySelector('.tab-results')
      .replaceWith(doc.querySelector('.tab-results'));

  if (Object.keys(params).length !== 0) {
    executeReviewSearch(path, params, context, renderReviewResults);
  }
}

// Path update functions.
function triggerReviewSearch() {
  // Home's global key handler can run on a directly entered review URL.
  if (!REVIEW_ENABLED) return;

  const filters = Util.collectSearchFilters();
  filters.keywords = document.querySelector('.smart-search-input.review').value;
  const model = document.querySelector('.review.model-input').value;
  const checkedReviewType = document.querySelector(
      '.review-type-container input[name="review-type"]:checked');
  const reviewType = checkedReviewType ?
    checkedReviewType.value : DEFAULT_REVIEW_TYPE;

  const newPath = Core.formHashPath('review', {
    ...filters,
    'names': '',
    'medias': '',
    'model': model,
    'review_type': reviewType,
  });
  window.location.hash = newPath;

  // Persist filter state so the indicator dot stays lit and values survive
  // tab switches — same contract as triggerSearch/triggerMatch.
  SharkMatcher.setFilters('review', filters);
}

// API execution functions.
function executeReviewSearch(path, params, context, renderFunction, kwargs = {}) {
  if (Util.PAGINATE) {
    const page = kwargs['page'] || 1;
    SharkMatcher.Review.filter_reviews(
        params.min_length, params.max_length, params.min_time, params.max_time,
        params.min_associated, params.max_associated, params.sex, params.locations,
        params.qualities, params.max_qualities, params.alternate_names,
        params.labels, params.tags, params.allow_unknown, params.keywords,
        params.model, params.review_type, page, Util.PAGE_SIZE)
        .then(function(result) {
          kwargs['page'] = page;
          kwargs['model'] = params.model;
          kwargs['review_type'] = params.review_type;
          renderFunction(result, context, kwargs);
        })
        .catch(function(error) {
          console.log('Error executing review.');
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error executing review: ${message}`);
          }
        });
  } else {
    // Review only ships paginated — the inference cache is sliced server-side.
    throw new Error('Review tab requires Util.PAGINATE = true.');
  }
}

// HTML render functions.
function renderReviewResults(result, context, kwargs = {}) {
  if (Util.PAGINATE) {
    const page = kwargs['page'];
    const isFirstPage = page === 1;
    const modelName = kwargs['model'] || '';
    const reviewType = kwargs['review_type'] || DEFAULT_REVIEW_TYPE;

    if (isFirstPage) {
      document.querySelector('.results-container').innerHTML = '';
    }

    // Rebind load-more each page so it carries the right next-page number.
    // Re-parse the hash at click time so pagination always reflects the URL
    // that's currently live (filters, model, review_type) — matches search.js.
    const loadMoreButton = document.querySelector('.load-more-button');
    const newLoadMoreButton = loadMoreButton.cloneNode(true);
    loadMoreButton.replaceWith(newLoadMoreButton);
    newLoadMoreButton.style.display = result.has_more ? 'block' : 'none';
    newLoadMoreButton.addEventListener('click', function() {
      const [currentPath, currentParams] = Routes.parsePath(window.location.hash);
      kwargs['page'] = page + 1;
      executeReviewSearch(currentPath, currentParams, context, renderReviewResults, kwargs);
    });

    // Distinguish "retriever has no cached suggestions for this type" from
    // "filters narrowed the result set to zero" — otherwise an empty filtered
    // result misleadingly looks like broken/missing review data.
    if (isFirstPage && result.suggestions.length === 0) {
      const emptyMessage = result.suggestions_available ?
        'No individuals match your search.' :
        'Missing review data, re-generating embeddings produces review.';
      document.querySelector('.results-container').innerHTML =
        `<h3>${emptyMessage}</h3>`;
      return;
    }

    if (isFirstPage && modelName !== '') {
      document.querySelector('.results-container').innerHTML =
        `<h3>Review results using ${Util.escapeHtml(modelName)} model.</h3>`;
    }

    const builder = reviewType === 'merge_suggestions' ?
      buildReviewMergeItem :
      buildReviewUnmatchItem;
    for (const suggestion of result.suggestions) {
      document.querySelector('.results-container')
          .appendChild(builder(suggestion, context));
    }
    return;
  }

  // Review only ships paginated — the inference cache is sliced server-side.
  throw new Error('Review tab requires Util.PAGINATE = true.');
}

function buildReviewUnmatchItem(suggestion, context) {
  const individual = suggestion.individual;
  const html = `
      <div class="results-item" object-id=${individual.id}>
          <div class="result-profile"></div>
          <div class="result-gallery"></div>
      </div>
      `;
  let doc = document.createElement('div');
  doc.innerHTML = html;
  doc = Individual.renderIndividualProfile(
      [individual], context,
      {
        'container': doc,
        'selector': '.result-profile',
        'page': 'review_unmatch',
        'k_value': suggestion.k_value,
        'flagged_image_id': suggestion.flagged_image_id,
      });
  const renderResult = Home.renderGallery(
      [individual], context,
      {
        'container': doc,
        'selector': '.result-gallery',
        'page': 'review',
        'flagged_image_id': suggestion.flagged_image_id,
        'flagged_image_k_map': suggestion.flagged_image_k_map,
      });
  doc = renderResult[0];
  return doc.querySelector('.results-item');
}

function buildReviewMergeItem(suggestion, context) {
  // Render a candidate pair row with a distance card on the left and two candidate panes.
  // Each pane uses simple class selectors because the gallery wrapper cannot parse compound
  // selectors.
  const indA = suggestion.individual_a;
  const indB = suggestion.individual_b;

  const refA = Util.individualGetReferenceIndividual(indA);
  const refB = Util.individualGetReferenceIndividual(indB);
  const nameA = refA ? refA.name : indA.name;
  const nameB = refB ? refB.name : indB.name;
  const mergeLink = Core.formHashPath(
      'merge', {'name1': nameA, 'name2': nameB});

  const distance = suggestion.distance;
  const distanceDisplay =
    distance === null || distance === undefined || isNaN(distance) ?
      'U' :
      Number(distance).toFixed(2);

  const html = `
      <div class="results-item merge-pair">
          <div class="merge-pair-distance">
              <div class="info-row">
                  <span class="label">Distance:</span>
                  <span class="value">${distanceDisplay}</span>
              </div>
              <div class="card-actions">
                  <a href="${mergeLink}" class="card-action-btn" title="Merge these ${Util.getDisplayName()}s">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                          stroke="currentColor" stroke-width="2" stroke-linecap="round"
                          stroke-linejoin="round">
                          <path d="M8 6H5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h3"></path>
                          <path d="M16 6h3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-3"></path>
                          <line x1="12" y1="2" x2="12" y2="22"></line>
                      </svg>
                      <span>Merge</span>
                  </a>
              </div>
          </div>
      </div>
      `;
  const doc = document.createElement('div');
  doc.innerHTML = html;
  const item = doc.querySelector('.results-item');

  item.appendChild(buildReviewMergePairSide(indA, context));
  item.appendChild(buildReviewMergePairSide(indB, context));

  return item;
}

function buildReviewMergePairSide(individual, context) {
  // Mirrors search.js:buildSearchResultItem — profile-left, gallery-right —
  // so the existing .result-profile / .result-gallery CSS takes over and the
  // side renders identically to a search result row.
  const html = `
      <div class="merge-pair-side">
          <div class="result-profile"></div>
          <div class="result-gallery"></div>
      </div>
      `;
  let doc = document.createElement('div');
  doc.innerHTML = html;
  doc = Individual.renderIndividualProfile(
      [individual], context,
      {'container': doc, 'selector': '.result-profile', 'page': 'individual'});
  const renderResult = Home.renderGallery(
      [individual], context,
      {'container': doc, 'selector': '.result-gallery', 'page': 'search'});
  doc = renderResult[0];
  return doc.querySelector('.merge-pair-side');
}

function renderReviewTypeSelector(params) {
  let selected = DEFAULT_REVIEW_TYPE;
  if (params.review_type &&
    REVIEW_TYPES.some((t) => t.value === params.review_type)) {
    selected = params.review_type;
  }
  const selectedLabel =
    REVIEW_TYPES.find((t) => t.value === selected).label;

  const optionsHtml = REVIEW_TYPES.map((t) => `
    <div class="input-item">
      <label>
        <input type="radio" name="review-type" value="${t.value}"
          ${t.value === selected ? 'checked' : ''}/>
        ${t.label}
      </label>
    </div>
  `).join('');

  const {container, popover} = Util.renderPillSelector({
    selector: '.input-section.sort',
    containerClass: 'review-type-container',
    icon: '',
    prefix: 'Review:',
    labelText: selectedLabel,
    bodyHtml: optionsHtml,
  });

  container.querySelectorAll('input[name="review-type"]').forEach((radio) => {
    radio.addEventListener('change', () => {
      if (!radio.checked) return;
      const label = REVIEW_TYPES.find((t) => t.value === radio.value).label;
      container.querySelector('.sort-label').textContent = label;
      popover.classList.remove('active');
    });
  });
}

export {
  init,
  triggerReviewSearch,
};
