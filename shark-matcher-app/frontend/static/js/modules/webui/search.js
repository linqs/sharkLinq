import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Home from './home.js';
import * as Individual from './individual.js';
import * as Routes from './routes.js';
import * as Util from './util.js';


function init() {
  Routes.addRoute(/^search$/, handlerSearch, 'search');
}

function handlerSearch(path, params, context) {
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
        <div class="tab-content search">
            <div class="tab-title"></div>
            ${Util.buildSmartSearchBar('search', 'Search by keywords...', keywordsValue, '', sortDisplay)}
        </div>
    `;
  let doc = document.createElement('div');
  doc.innerHTML = html;

  // Search bar logic
  Util.bindSmartSearchEvents(doc, 'search');

  const searchInput = doc.querySelector('.smart-search-input');
  searchInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      const currentPopover = document.getElementById('search-filter-popover');
      if (currentPopover) {
        currentPopover.classList.remove('active');
      }
      triggerSearch();
    }
  });

  document.querySelector('.tab-content')
      .replaceWith(doc.querySelector('.tab-content.search'));

  Home.renderTitle('search', context);
  renderSearchFilters(
      path, params, context,
      {
        'selector': '.filter-popover', 'page': 'search',
      });
  renderSortInput(
      path, params, context,
      {
        'selector': '.input-section.sort', 'page': 'search',
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
  doc = document.createElement('div');
  doc.innerHTML = html;
  document.querySelector('.tab-results')
      .replaceWith(doc.querySelector('.tab-results'));

  // Populate results.
  if (Object.keys(params).length !== 0) {
    executeSearch(path, params, context, renderSearchResults);
  }
}

// Path update functions.
function triggerSearch() {
  // Sorting inputs.
  const attributeInput = document.querySelector('.sort.attribute-input').value;
  const orderInput = document.querySelector('.sort.order-input').value;

  const filters = Util.collectSearchFilters();
  filters.keywords = document.querySelector('.smart-search-input').value;

  const newPath = Core.formHashPath('search', {
    ...filters,
    'names': '',
    'medias': '',
    'attribute': attributeInput,
    'order': orderInput,
  });
  window.location.hash = newPath;

  // Update the locally stored filter values for this tab.
  SharkMatcher.setFilters('search', filters);
}

function triggerExport() {
  const includeImagesCheckbox = document.querySelector('.export-include-images-checkbox');
  const includeImages = includeImagesCheckbox ? includeImagesCheckbox.checked : false;

  const filters = {
    ...Util.collectSearchFilters(),
    'names': '',
    'medias': '',
    'keywords': document.querySelector('.smart-search-input').value,
    'include_images': includeImages,
  };

  const exportButton = document.querySelector('.export-filtered-button');
  const exportButtonText = document.querySelector('.export-button-text');
  if (exportButton) {
    exportButton.style.pointerEvents = 'none';
    exportButton.style.opacity = '0.7';
  }
  if (exportButtonText) {
    exportButtonText.textContent = ' Exporting...';
  }

  SharkMatcher.export_data('all', 'annotations', filters)
      .then(function(result) {
        if (exportButton) {
          exportButton.style.pointerEvents = '';
          exportButton.style.opacity = '';
        }
        if (exportButtonText) {
          exportButtonText.textContent = ' Export Results';
        }
        if (!result) return;
        downloadFile(result, 'annotations');
      })
      .catch(function(error) {
        if (exportButton) {
          exportButton.style.pointerEvents = '';
          exportButton.style.opacity = '';
        }
        if (exportButtonText) {
          exportButtonText.textContent = ' Export Results';
        }
        console.log('Error exporting dataset.');
        const message = Util.handleAPIError(error);
        if (message) {
          alert(`Error exporting dataset: ${message}`);
        }
      });
}

function downloadFile(base64Data, format) {
  const mimeType = 'application/zip';
  const fileExtension = 'zip';

  // Decode the Base64 string to a binary string
  const binaryString = atob(base64Data);

  // Convert binary string to a Uint8Array
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  // Create a Blob from the Uint8Array
  const blob = new Blob([bytes], {type: mimeType});
  const url = URL.createObjectURL(blob);

  // Create a temporary anchor element and trigger download
  const a = document.createElement('a');
  a.href = url;
  a.download = `shark_matcher_${format}.${fileExtension}`; // e.g., dataset.csv

  document.body.appendChild(a);
  a.click();

  // Cleanup
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// API execution functions.
function executeSearch(path, params, context, renderFunction, kwargs = {}) {
  if (Object.keys(params).length === 0) {
    alert('Please provide a valid input.');
    return;
  }

  if (Util.PAGINATE) {
    const page = kwargs['page'] || 1;
    const attribute = params['attribute'] || 'name';
    const order = params['order'] || 'descending';
    SharkMatcher.Individual.filter_individuals(
        params.min_length, params.max_length, params.min_time,
        params.max_time, params.min_associated, params.max_associated,
        params.sex, params.locations, params.qualities, params.max_qualities, null, params.names,
        params.medias, params.alternate_names, params.labels, params.tags,
        params.allow_unknown, params.keywords,
        true, page, Util.PAGE_SIZE, attribute, order)
        .then(function(result) {
          kwargs['page'] = page;
          renderFunction(result, context, kwargs);
        })
        .catch(function(error) {
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error executing filter search: ${message}`);
          }
        });
  } else {
    SharkMatcher.Individual.filter_individuals(
        params.min_length, params.max_length, params.min_time,
        params.max_time, params.min_associated, params.max_associated,
        params.sex, params.locations, params.qualities, params.max_qualities, null, params.names,
        params.medias, params.alternate_names, params.labels, params.tags,
        params.allow_unknown, params.keywords)
        .then(function(result) {
          kwargs['load_index'] = 1;
          renderFunction(result, context, kwargs);
        })
        .catch(function(error) {
          const message = Util.handleAPIError(error);
          if (message) {
            alert(`Error executing filter search: ${message}`);
          }
        });
  }
}

// Render the filter section of the search area.
function renderSearchFilters(path, params, context, kwargs = {}) {
  const dynamicOptions = Util.getSelectorOptions();

  // Fall back to stored filters only when the URL path specifies none.
  const page = kwargs['page'] || 'search';
  if (!Util.hasFilterParams(params)) {
    params = SharkMatcher.getFilters(page) || params;
  }
  let alternateNamesValue = '';
  if (params.alternate_names && params.alternate_names !== '') {
    alternateNamesValue = `value="${Util.escapeHtml(params.alternate_names)}"`;
  }
  let tagsValue = '';
  if (params.tags && params.tags !== '') {
    tagsValue = `value="${Util.escapeHtml(params.tags)}"`;
  }
  let minTimeValue = '';
  if (params.min_time && params.min_time !== '') {
    minTimeValue = `value="${Util.timestampToInput(params.min_time)}"`;
  }
  let maxTimeValue = '';
  if (params.max_time && params.max_time !== '') {
    maxTimeValue = `value="${Util.timestampToInput(params.max_time)}"`;
  }
  let minAssociatedValue = '';
  if (params.min_associated && params.min_associated !== '') {
    minAssociatedValue = `value="${params.min_associated}"`;
  }
  let maxAssociatedValue = '';
  if (params.max_associated && params.max_associated !== '') {
    maxAssociatedValue = `value="${params.max_associated}"`;
  }
  let minLengthValue = '';
  if (params.min_length && params.min_length !== '') {
    minLengthValue = `value="${params.min_length}"`;
  }
  let maxLengthValue = '';
  if (params.max_length && params.max_length !== '') {
    maxLengthValue = `value="${params.max_length}"`;
  }
  let allowUnknownChecked = 'checked';
  if (params.allow_unknown !== undefined &&
    String(params.allow_unknown) === 'false') {
    allowUnknownChecked = '';
  }

  // Selector pre_population.
  let labelsPlaceholder = 'all';
  if (params.labels && params.labels !== '') {
    labelsPlaceholder = params.labels;
  }
  let sexPlaceholder = 'all';
  if (params.sex && params.sex !== '') {
    sexPlaceholder = params.sex;
  }
  let selectedLocations = [];
  if (params.locations && params.locations !== '') {
    selectedLocations = params.locations.split(',');
  }
  let qualityPlaceholder = 'all';
  if (params.qualities && params.qualities !== '') {
    qualityPlaceholder = params.qualities;
  }
  let maxQualityPlaceholder = 'all';
  if (params.max_qualities && params.max_qualities !== '') {
    maxQualityPlaceholder = params.max_qualities;
  }

  const html = `
    <div class="filter-popover" id="${kwargs['page']}-filter-popover">
    <div class="input-item">
        <label for="search-alternate_name-input">Nicknames: </label>
        <input
            class="search alternate_name-input"
            id="search-alternate_name-input"
            type="text"
            placeholder="nicknames"
            ${alternateNamesValue}/>
    </div>
    <div class="input-item">
        <label for="search-tag-input">Tags: </label>
        <input
            class="search tag-input"
            id="search-tag-input"
            type="text"
            placeholder="tags"
            ${tagsValue}/>
    </div>
    <div class="input-item">
        <label for="search-label-input">Labels: </label>
        <select
            class="search label-input"
            id="search-label-input">
        </select>
    </div>
    <div class="input-item">
        <label for="search-min_associated-input">Min Associated Images: </label>
        <input
            class="search min_associated-input"
            id="search-min_associated-input"
            type="number"
            placeholder="min associated"
            ${minAssociatedValue}/>
    </div>
    <div class="input-item">
        <label for="search-max_associated-input">Max Associated Images: </label>
        <input
            class="search max_associated-input"
            id="search-max_associated-input"
            type="number"
            placeholder="max associated"
            ${maxAssociatedValue}/>
    </div>
    <div class="input-item">
        <label for="search-sex-input">Sex: </label>
        <select
            class="search sex-input"
            id="search-sex-input">
        </select>
    </div>
    <div class="input-item">
        <label for="search-min_length-input">Min Length: </label>
        <input
            class="search min_length-input"
            id="search-min_length-input"
            type="number"
            placeholder="min length"
            ${minLengthValue}/>
    </div>
    <div class="input-item">
        <label for="search-max_length-input">Max Length: </label>
        <input
            class="search max_length-input"
            id="search-max_length-input"
            type="number"
            placeholder="max length"
            ${maxLengthValue}/>
    </div>
    <div class="input-item">
        <label>Locations: </label>
        <div class="search-location-dropdown" id="search-location-dropdown">
            <div class="dropdown-toggle">Select Locations</div>
            <div class="dropdown-menu">
                ${dynamicOptions[1].map((location) => `
                    <label>
                      <input type="checkbox" value="${Util.escapeHtml(location)}">${Util.escapeHtml(location)}
                    </label>
                `).join('')}
            </div>
        </div>
    </div>
    <div class="input-item">
        <label for="search-min_time-input">Start Date: </label>
        <input
            class="search min_time-input"
            id="search-min_time-input"
            type="date"
            ${minTimeValue}/>
    </div>
    <div class="input-item">
        <label for="search-max_time-input">End Date: </label>
        <input
            class="search max_time-input"
            id="search-max_time-input"
            type="date"
            ${maxTimeValue}/>
    </div>
    <div class="input-item">
        <label for="search-min-quality-input">Min Quality: </label>
        <select
            class="search min-quality-input"
            id="search-min-quality-input">
        </select>
    </div>
    <div class="input-item">
        <label for="search-max-quality-input">Max Quality: </label>
        <select
            class="search max-quality-input"
            id="search-max-quality-input">
        </select>
    </div>
    <div class="input-item" style="border-bottom: none; display: flex; flex-direction: row; align-items: center; gap: 10px; margin-top: 5px;">
        <label for="search-allow_unknown-input" style="margin: 0; min-width: auto;">Allow Unknown: </label>
        <input
            class="search allow_unknown-input"
            id="search-allow_unknown-input"
            type="checkbox"
            style="width: 20px; height: 20px; margin: 0;"
            ${allowUnknownChecked}/>
    </div>
    <div style="display: flex; gap: 10px; margin-top: 10px;">
        <button class="clear-filters-button">Clear Filters</button>
    </div>
    </div>
    `;

  const doc = document.createElement('div');
  doc.innerHTML = html;

  const clearButton = doc.querySelector('.clear-filters-button');
  clearButton.addEventListener('click', function() {
    Util.clearSearchFilters();
  });

  document.querySelector(kwargs['selector'])
      .replaceWith(doc.querySelector('.filter-popover'));

  // Populate selector options.
  let selectOptions = [sexPlaceholder, `all`].concat(Util.sexOptions);
  Util.renderSelectOptions(
      selectOptions, context, {'selector': '.search.sex-input'});
  selectOptions = [labelsPlaceholder, `all`].concat(dynamicOptions[0]);
  Util.renderSelectOptions(
      selectOptions, context, {'selector': '.search.label-input'});
  selectOptions = dynamicOptions[1];

  // Populate checkbox.
  createCheckboxDropdown({
    containerClass: '.search-location-dropdown',
    options: selectOptions,
    selectedValues: selectedLocations,
    toggleText: 'Select Locations',
  });

  selectOptions = [qualityPlaceholder, `all`].concat(Util.qualityOptions);
  Util.renderSelectOptions(
      selectOptions, context, {'selector': '.search.min-quality-input'});

  selectOptions = [maxQualityPlaceholder, `all`].concat(Util.qualityOptions);
  Util.renderSelectOptions(
      selectOptions, context, {'selector': '.search.max-quality-input'});

  Util.updateFilterIndicator(kwargs['page']);
}

function createCheckboxDropdown(
    {containerClass, options, selectedValues, toggleText}) {
  // Dropdown functionality.
  const dropdown = document.querySelector(containerClass);
  const dropdownToggle = dropdown.querySelector('.dropdown-toggle');
  const dropdownMenu = dropdown.querySelector('.dropdown-menu');
  const checkboxes = dropdownMenu.querySelectorAll('input[type="checkbox"]');

  // Pre-populate selected values.
  checkboxes.forEach((checkbox) => {
    if (selectedValues.includes(checkbox.value)) {
      checkbox.checked = true;
    }
  });

  // Update dropdown toggle text based on selections.
  const updateDropdownText = () => {
    const selected = Array.from(checkboxes)
        .filter((cb) => cb.checked).map((cb) => cb.value);
    dropdownToggle.textContent =
      selected.length > 0 ? selected.join(', ') : toggleText;
  };

  updateDropdownText(); // Initial update.

  dropdownToggle.addEventListener('click', () => {
    dropdown.classList.toggle('active');
  });

  checkboxes.forEach((checkbox) => {
    checkbox.addEventListener('change', updateDropdownText);
  });

  updateDropdownText();

  // Close dropdown when clicking outside.
  document.addEventListener('click', (event) => {
    if (!dropdown.contains(event.target)) {
      dropdown.classList.remove('active');
    }
  });
}

function renderSearchResults(result, context, kwargs = {}) {
  if (Util.PAGINATE) {
    const page = kwargs['page'];
    const isFirstPage = page === 1;
    const items = result.results;

    if (isFirstPage) {
      document.querySelector('.results-container').innerHTML =
        Util.renderResultCount(result.total_count);
    }

    // Rebind the load-more button on every page so its handler carries the
    // correct next-page number.
    const loadMoreButton = document.querySelector('.load-more-button');
    const newLoadMoreButton = loadMoreButton.cloneNode(true);
    loadMoreButton.replaceWith(newLoadMoreButton);
    newLoadMoreButton.style.display = result.has_more ? 'block' : 'none';
    newLoadMoreButton.addEventListener('click', function() {
      const [currentPath, currentParams] = Routes.parsePath(window.location.hash);
      kwargs['page'] = page + 1;
      executeSearch(currentPath, currentParams, context, renderSearchResults, kwargs);
    });

    for (const individual of items) {
      document.querySelector('.results-container')
          .appendChild(buildSearchResultItem(individual, context));
    }
    return;
  }

  // Non-paginated path: results is a flat array, client-side slicing.
  let results = result;
  const path = Routes.parsePath(window.location.hash);
  const params = path[1];
  let loadIndex = kwargs['load_index'];

  if (loadIndex === 1) {
    // Filter out the utility individual (ID = -1)
    results = results.filter((ind) => ind && ind.id !== -1 && ind.id !== '-1');

    // Add reference individual names to the results list.
    for (let i = 0; i < results.length; i++) {
      if (!results[i]) continue;
      const referenceIndividual = Util.individualGetReferenceIndividual(results[i]);
      let refName = -1;
      if (referenceIndividual && referenceIndividual.name) {
        refName = parseInt(referenceIndividual.name) || -1;
      } else if (results[i].name) {
        refName = parseInt(results[i].name) || -1;
      }
      results[i].reference_name = refName;
    }

    // Default sorting for results, or override with given parameters.
    // Defaults: attribute 'name', order 'descending'.
    let sortAttribute = 'name';
    if (params.attribute && params.attribute !== '') {
      sortAttribute = params.attribute;
    }
    const effectiveOrder = (params.order && params.order !== '') ? params.order : 'descending';
    const sortOrder = effectiveOrder !== 'descending';

    results = Util.sortByAttribute(results, sortAttribute, sortOrder);
    kwargs['processed_results'] = results;

    document.querySelector('.results-container').innerHTML =
      Util.renderResultCount(results.length);

    const loadMoreButton = document.querySelector('.load-more-button');
    const newLoadMoreButton = loadMoreButton.cloneNode(true);
    loadMoreButton.replaceWith(newLoadMoreButton);
    newLoadMoreButton.addEventListener('click', function() {
      kwargs['load_index'] = loadIndex += 1;
      renderSearchResults(results, context, kwargs);
    });
  } else {
    results = kwargs['processed_results'];
  }

  const bounds = Util.computeResultBounds(results.length, loadIndex);
  for (let i = bounds[0]; i < bounds[1]; i++) {
    const individual = results[i];

    if (individual === undefined) {
      const html = `
            <div class="results-item">
                <h3>${Util.getDisplayName()} not found.</h3>
            </div>
            `;
      const doc = document.createElement('div');
      doc.innerHTML = html;
      document.querySelector('.results-container')
          .appendChild(doc.querySelector('.results-item'));
      return;
    }
    document.querySelector('.results-container')
        .appendChild(buildSearchResultItem(individual, context));
  }
}

function buildSearchResultItem(individual, context) {
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
      {'container': doc, 'selector': '.result-profile', 'page': 'individual'});
  const renderResult = Home.renderGallery(
      [individual], context,
      {'container': doc, 'selector': '.result-gallery', 'page': 'search'});
  doc = renderResult[0];
  return doc.querySelector('.results-item');
}

function renderSortInput(path, params, context, kwargs = {}) {
  // Explicit defaults per page — no ambiguous placeholders.
  const defaultAttribute = kwargs['page'] === 'search' ? 'name' : 'id';
  const defaultOrder = 'descending';

  // Populate input fields by default with what's already on the current path.
  let selectedAttribute = defaultAttribute;
  if (params.attribute && params.attribute !== '') {
    selectedAttribute = params.attribute;
  }
  let selectedOrder = defaultOrder;
  if (params.order && params.order !== '') {
    selectedOrder = params.order;
  }

  const bodyHtml = `
    <div class="input-item">
        <label for="sort-attribute-input">Attribute: </label>
        <select class="sort attribute-input" id="sort-attribute-input"></select>
    </div>
    <div class="input-item">
        <label for="sort-order-input">Order: </label>
        <select class="sort order-input" id="sort-order-input"></select>
    </div>
  `;

  const {wrapper, container} = Util.renderPillSelector({
    selector: kwargs['selector'],
    icon: '',
    prefix: 'Sort:',
    labelText: selectedAttribute,
    bodyHtml: bodyHtml,
  });

  const attributeSelect = container.querySelector('.sort.attribute-input');

  // Update label live when attribute changes.
  attributeSelect.addEventListener('change', () => {
    if (attributeSelect.value) {
      container.querySelector('.sort-label').textContent = attributeSelect.value;
    }
  });

  // The export button lives as a sibling of the pill inside the wrapper, so
  // it sits next to "Sort:" on the search tab only. Mounted after the pill
  // helper so it doesn't bleed into other tabs.
  if (kwargs['page'] === 'search') {
    const currentUser = Core.getContextUser();
    let canExport = false;
    if (currentUser && currentUser.user_role) {
      canExport = Core.roleDict[currentUser.user_role] >= 1;
    }
    if (canExport) {
      const exportButtonHtml = `
        <div class="export-filtered-button sort-pill-btn">
          <span class="export-button-text">Export Results</span>
          <span style="margin: 0 4px; opacity: 0.3;">|</span>
          <label class="export-images-label" style="font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 4px; font-weight: normal;">
            <input type="checkbox" class="export-include-images-checkbox"> Include Images
          </label>
        </div>`;
      wrapper.insertAdjacentHTML('beforeend', exportButtonHtml);
      wrapper.querySelector('.export-filtered-button')
          .addEventListener('click', function(e) {
            if (!e.target.closest('.export-images-label')) {
              triggerExport();
            }
          });
    }
  }

  // Populate selector options. The selected value is set afterwards so the
  // dropdown reflects the current (or default) choice explicitly.
  let selectOptions;
  if (kwargs['page'] === 'search') {
    // TODO have an option to sort by last modified individual, this would rely on
    // the actions information.
    // TODO Sorting options: `associated images`, `oldest image`, `newest image`
    selectOptions = [`name`, `associated images`];
  }

  if (kwargs['page'] === 'history' || kwargs['page'] === 'proposals') {
    selectOptions = [`id`, `time`, `author`, `object`, `action`];
  }

  Util.renderSelectOptions(
      selectOptions, context, {'selector': '.sort.attribute-input'});
  attributeSelect.value = selectedAttribute;

  const orderSelect = container.querySelector('.sort.order-input');
  Util.renderSelectOptions(
      [`ascending`, `descending`], context, {'selector': '.sort.order-input'});
  orderSelect.value = selectedOrder;
}

export {
  init,
  renderSearchFilters,
  renderSortInput,
  triggerSearch,
  createCheckboxDropdown,
};
