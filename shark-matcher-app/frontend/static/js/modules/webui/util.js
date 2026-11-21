import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Log from './log.js';
import * as Routes from './routes.js';

// JS util functions
function init() {
  registerHandler('onKeyEvent', onKeyEvent);

  // Update titles and instructions with dynamic display name.
  const displayName = getDisplayName();
  titles['individual'] = displayName;

  instructions['match'] = `Query an ${displayName.toLowerCase()} and see the match results provided by Shark Matcher with the option to filter match results. View images in focus mode by double clicking on any query or result image.`;
  instructions['merge'] = `View the 2 input ${displayName.toLowerCase()}s (leftmost columns) and a preview of the merged output ${displayName.toLowerCase()} that will result. Modify properties or associated objects of the output ${displayName.toLowerCase()} before saving the merge.`;
  instructions['proposals'] = `View and modify proposed updates to ${displayName.toLowerCase()}s and images. Admin users can approve, decline, or modify proposed udpates.`;
  instructions['search'] = `Search ${displayName.toLowerCase()}s by keywords with query search. Keywords may include ${displayName.toLowerCase()} id numbers, nicknames, tags, labels, or image names. Advanced search allows searching by a multitude of associated objects or properties.`;
  instructions['individual'] = `View and edit an ${displayName.toLowerCase()}'s properties, associated objects, and associated images. Select images from the gallery to view or edit the image's properties. Images may also be set as the new reference image for this ${displayName.toLowerCase()} or unmatched to be their own ${displayName.toLowerCase()} that may be rematched.`;
  instructions['upload'] = `Upload new images to Shark Matcher. Drop image files and update image properties before uploading. All uploaded images become new unmatched ${displayName.toLowerCase()}s that may then be matched.`;
}

// Register a handler and return it's global name.
function registerHandler(name, handler) {
  window.sharkMatcher = window.sharkMatcher || {};
  window.sharkMatcher.handlers = window.sharkMatcher.handlers || {};

  window.sharkMatcher.handlers[name] = handler;

  return `window.sharkMatcher.handlers.${name}`;
}

function onKeyEvent(event, context, keys, handler) {
  if (!keys.includes(event.key)) {
    return;
  }

  handler(event, context);
}

// Look through parents until one matches the specified query.
function queryAncestor(element, query) {
  if (!element) {
    return undefined;
  }

  if (element.parentElement.matches(query)) {
    return element.parentElement;
  }

  return queryAncestor(element.parentElement, query);
}


// Encode text as UTF-8 bytes for browser-native consumers.
function textToBytes(text) {
  return new TextEncoder().encode(text);
}

// Return a lowercase hexadecimal SHA-256 digest for text when subtle crypto is unavailable.
function sha256Fallback(text) {
  function rightRotate(value, amount) {
    return (value >>> amount) | (value << (32 - amount));
  }

  const bytes = textToBytes(text);
  const byteLength = bytes.length;
  const words = [];
  for (let i = 0; i < byteLength; i++) {
    words[i >> 2] |= bytes[i] << ((3 - (i % 4)) * 8);
  }
  words[byteLength >> 2] |= 128 << ((3 - (byteLength % 4)) * 8);
  words[(((byteLength + 8) >> 6) << 4) + 14] = Math.floor((byteLength * 8) / 0x100000000);
  words[(((byteLength + 8) >> 6) << 4) + 15] = (byteLength * 8) >>> 0;

  const k = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Array(64);
  for (let i = 0; i < words.length; i += 16) {
    for (let j = 0; j < 16; j++) {
      w[j] = words[i + j] | 0;
    }
    for (let j = 16; j < 64; j++) {
      const s0 = rightRotate(w[j - 15], 7) ^
                 rightRotate(w[j - 15], 18) ^
                 (w[j - 15] >>> 3);
      const s1 = rightRotate(w[j - 2], 17) ^
                 rightRotate(w[j - 2], 19) ^
                 (w[j - 2] >>> 10);
      w[j] = (w[j - 16] + s0 + w[j - 7] + s1) | 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let j = 0; j < 64; j++) {
      const s1 = rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + s1 + ch + k[j] + w[j]) | 0;
      const s0 = rightRotate(a, 2) ^ rightRotate(a, 13) ^ rightRotate(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (s0 + maj) | 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) | 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) | 0;
    }

    h0 = (h0 + a) | 0;
    h1 = (h1 + b) | 0;
    h2 = (h2 + c) | 0;
    h3 = (h3 + d) | 0;
    h4 = (h4 + e) | 0;
    h5 = (h5 + f) | 0;
    h6 = (h6 + g) | 0;
    h7 = (h7 + h) | 0;
  }

  const toHex = (word) => (word >>> 0).toString(16).padStart(8, '0');
  return toHex(h0) + toHex(h1) + toHex(h2) + toHex(h3) +
         toHex(h4) + toHex(h5) + toHex(h6) + toHex(h7);
}

// Return a lowercase hexadecimal SHA-256 digest for text.
async function sha256(text) {
  if (typeof window !== 'undefined' && window.isSecureContext &&
      window.crypto && window.crypto.subtle) {
    const digest = await window.crypto.subtle.digest(
        'SHA-256', textToBytes(text));
    return Array.from(new Uint8Array(digest), function(byte) {
      return byte.toString(16).padStart(2, '0');
    }).join('');
  }

  return sha256Fallback(text);
}

// Show and hide loader.
let loaderCalls = 0;
function loadingStart() {
  loaderCalls += 1;
  if (loaderCalls >= 0) {
    document.querySelector('.loader').style.display = 'block';
  }
}
function loadingStop() {
  loaderCalls -= 1;
  if (loaderCalls == 0) {
    document.querySelector('.loader').style.display = 'none';
  }
}

function caseInsensitiveStringCompare(a, b) {
  return a.localeCompare(b, undefined, {sensitivity: 'base'});
}

// Return UNKNOWN_VALUE if null or undefined, otherwise return the HTML-escaped value.
// Do not use for plain-text contexts like alert() or DOM value assignments where entities would
// show literally.
function displayValue(value) {
  if (value === null || value === undefined) {
    return UNKNOWN_VALUE;
  }
  return escapeHtml(value);
}

// Get the display name from the global config.
function getDisplayName() {
  if (window.sharkMatcher && window.sharkMatcher.config && window.sharkMatcher.config.displayName) {
    return window.sharkMatcher.config.displayName;
  }
  return 'Individual';
}

// Return a standardized location that always has as leading hash.
function getLocationHash() {
  const hash = window.location.hash.trim();

  if (hash.length === 0) {
    return '#';
  }

  return hash;
}

// Given a list of items, return a string concatenation, ordered alphanumerically.
function listToString(inputList, object) {
  inputList.sort((a, b) => a.name.localeCompare(b.name));

  let outputString = '';
  for (let i = 0; i < inputList.length; i++) {
    if (object === 'alternate_name') {
      outputString = outputString + inputList[i].name + ', ';
    } else if (object === 'tag') {
      outputString = outputString + inputList[i].name + ', ';
    } else if (object === 'label') {
      outputString = outputString + inputList[i].name + ', ';
    }
  }
  outputString = outputString.slice(0, -2);

  return outputString;
}

// Given a list of items, return a string concatenation, ordered alphanumerically.
function listToLinks(inputList) {
  let html = '';

  if (inputList.length === 0) {
    return html;
  }

  // Alphabetically sort the list.
  inputList.sort((a, b) => a.name.localeCompare(b.name));

  // Build the object string with links for each object.
  html = `<div class="object-string"></div>`;
  const doc = document.createElement('div');
  doc.innerHTML = html;

  for (let i = 0; i < inputList.length; i++) {
    const itemLink = Core.formHashPath('search', {
      'keywords': inputList[i].name,
      'attribute': '',
      'order': '',
    });
    const itemHtml = `<p class="object-link"><a href="${itemLink}">${escapeHtml(inputList[i].name)}</a></p>`;
    const innerDoc = document.createElement('div');
    innerDoc.innerHTML = itemHtml;
    doc.querySelector('div').appendChild(innerDoc.querySelector('p'));
  }

  html = doc.querySelector('div').outerHTML;
  return html;
}

function sortMediaReferenceFirst(inputList, reference, attribute = 'time') {
  // Sort the rest of the array.
  const sortedList = inputList.toSorted((a, b) => a[attribute] - b[attribute]);

  // If the list is longer than 1, Get the reference element and remove it, then place it first.
  if (sortedList.length > 1) {
    let removalObject;
    for (let i = 0; i < sortedList.length; i++) {
      if (sortedList[i].individual === reference) {
        removalObject = sortedList[i];
        sortedList.splice(i, 1);
        break;
      }
    }
    if (!removalObject) {
      console.log('Error sorting medias.');
    }
    sortedList.unshift(removalObject);
  }

  return sortedList;
}

function sortList(inputList, attribute = 'time') {
  // Sort the rest of the array.
  const sortedList = inputList.toSorted((a, b) => a[attribute] - b[attribute]);

  return sortedList;
}

// Return a copy of arr sorted by attribute; empty arrays and invalid attributes return arr itself.
function sortByAttribute(arr, attribute, ascending = true) {
  // Return original array if input is invalid.
  if (!arr || arr.length === 0 || !attribute || typeof attribute !== 'string') {
    return arr;
  }

  // Change display attribute names to backend attribute names.
  if (attribute === 'associated images') {
    attribute = 'associated_medias';
  }
  if (attribute === 'name') {
    attribute = 'reference_name';
  }

  // Create a copy to avoid modifying the original array.
  const sortedArr = [...arr];

  sortedArr.sort((a, b) => {
    const aValue = a[attribute];
    const bValue = b[attribute];

    // Handle undefined or null values.
    if (aValue === undefined || aValue === null) return -1;
    if (bValue === undefined || bValue === null) return 1;

    if (typeof aValue === 'string' && typeof bValue === 'string') {
      const comparison = aValue.localeCompare(bValue); // Use localeCompare for strings.
      return ascending ? comparison : -comparison;
    } else if (Array.isArray(aValue) && Array.isArray(bValue)) {
      // Sort by array length: larger arrays first if descending, smaller first if ascending.
      return ascending ? aValue.length - bValue.length : bValue.length - aValue.length;
    } else if (typeof aValue === 'number' && typeof bValue === 'number') {
      return ascending ? aValue - bValue : bValue - aValue;
    } else if (typeof aValue === 'boolean' && typeof bValue === 'boolean') {
      return ascending ?
        (aValue ? -1 : 1) - (bValue ? -1 : 1) :
        (bValue ? -1 : 1) - (aValue ? -1 : 1);
    } else if (typeof aValue === 'object' && typeof bValue === 'object') {
      // Handle object comparison (e.g., by stringifying).
      const aString = JSON.stringify(aValue);
      const bString = JSON.stringify(bValue);
      const comparison = aString.localeCompare(bString);
      return ascending ? comparison : -comparison;
    } else {
      // Handle mixed types or other cases as needed (e.g., by converting to string).
      const aString = String(aValue);
      const bString = String(bValue);
      const comparison = aString.localeCompare(bString);
      return ascending ? comparison : -comparison;
    }
  });

  return sortedArr;
}

// Given a timestamp, return a string representation in mm-dd-yyyy format.
// Pass time = false to omit the hh:mm:ss portion (e.g. for "last seen" readouts).
function timestampToString(unixSeconds, time = true) {
  if (!unixSeconds) {
    return 'U';
  }
  // Shift the epoch by the local timezone offset so UTC accessors read local time.
  const offsetDate = new Date(
      (unixSeconds * 1000) +
      (new Date(unixSeconds * 1000).getTimezoneOffset() * -60 * 1000));
  const month = String(offsetDate.getUTCMonth() + 1).padStart(2, '0');
  const day = String(offsetDate.getUTCDate()).padStart(2, '0');
  const year = offsetDate.getUTCFullYear();
  const dateString = `${month}-${day}-${year}`;
  if (!time) {
    return dateString;
  }
  const hours = String(offsetDate.getUTCHours()).padStart(2, '0');
  const minutes = String(offsetDate.getUTCMinutes()).padStart(2, '0');
  const seconds = String(offsetDate.getUTCSeconds()).padStart(2, '0');
  return `${dateString} ${hours}:${minutes}:${seconds}`;
}

// Convert Unix seconds to an ISO date or datetime-local input string in the local timezone.
function timestampToInput(unixSeconds, time = false) {
  if (!unixSeconds) {
    return undefined;
  }

  if (time === true) {
    // Adjust the local time with the timezone offset.
    const offsetDate = new Date(
        (unixSeconds * 1000) +
        (new Date(unixSeconds * 1000).getTimezoneOffset() * -60 * 1000));
    return offsetDate.toISOString().slice(0, 19);
  } else if (time === false) {
    // Strip off the timezone portion and, when time is false, the time portion.
    const offsetDate = new Date(unixSeconds * 1000);
    return offsetDate.toISOString().slice(0, 10);
  }
}

function inputToTimestamp(inputString, time = true) {
  if (inputString === '') {
    return inputString;
  }

  let localTimestamp = Date.parse(inputString) / 1000;

  // If using just date, manually handle the timezone as well. Not needed with datetime-local input.
  if (time === false) {
    const offset = (new Date(localTimestamp * 1000).getTimezoneOffset() * 60 * 1000) / 1000;
    localTimestamp += offset;
  }

  return localTimestamp;
}

function toggleVisibility(style = 'block', visibleOverride = false, invisibleOverride = false, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }

  const element = container.querySelector(kwargs['selector']);
  // Override behavior.
  if (visibleOverride === true) {
    element.style.display = style;
    return;
  }
  if (invisibleOverride === true) {
    element.style.display = 'none';
    return;
  }

  // Regular toggle behavior.
  if (element.style.display === 'none') {
    element.style.display = style;
  } else {
    element.style.display = 'none';
  }
}

// Render the description of a history or proposal item's description field.
function renderResultDescription(actionNotes) {
  let parsedNotes = {};
  if (typeof actionNotes === 'string' && actionNotes.trim() !== '') {
    try {
      const parsed = JSON.parse(actionNotes);
      if (parsed && typeof parsed === 'object') {
        parsedNotes = parsed;
      }
    } catch {
      parsedNotes = {};
    }
  } else if (actionNotes && typeof actionNotes === 'object') {
    parsedNotes = actionNotes;
  }

  const original = (parsedNotes.original && typeof parsedNotes.original === 'object') ?
    parsedNotes.original : null;
  const final = (parsedNotes.final && typeof parsedNotes.final === 'object') ?
    parsedNotes.final : null;
  const skipKeys = ['source', 'media', 'individual'];
  const descriptions = {};

  const isEmptyObject = (obj) => !obj ||
      Object.keys(obj).length === 0 ||
      (Object.keys(obj).length === 1 && obj.id === -1);
  const isOriginalEmpty = isEmptyObject(original);
  const isFinalEmpty = isEmptyObject(final);

  for (const type of ['original', 'final']) {
    const data = (type === 'original') ? original : final;
    let descriptionString = '';

    // Render empty placeholder when the object is missing, empty, or the id: -1 sentinel.
    if (isEmptyObject(data)) {
      descriptionString = `<div class="info-row" style="justify-content: center; font-style: italic; opacity: 0.6;">Empty</div>`;
    } else {
      // Build the description string, preserving key order.
      for (const key in data) {
        if (skipKeys.includes(key)) {
          continue;
        }

        // If this section is populated and not one that should not be displayed, render it.
        if (!Object.prototype.hasOwnProperty.call(data, key) || hiddenKeys.includes(key)) {
          continue;
        }

        // Helper function to check if a value is null-like (null, undefined, empty string, etc.)
        function isNullLike(value) {
          return value === null || value === undefined || value === '' || value === 'null' || value === 'NULL';
        }

        // Check if only one of original or final contains this key
        const originalHasKey = Boolean(original) &&
            Object.prototype.hasOwnProperty.call(original, key);
        const finalHasKey = Boolean(final) && Object.prototype.hasOwnProperty.call(final, key);
        const onlyOneHasKey = (originalHasKey && !finalHasKey) || (!originalHasKey && finalHasKey);

        // If original and final differ, highlight this property.
        let isDifferent = originalHasKey &&
                    finalHasKey &&
                    (JSON.stringify(original[key]) !== JSON.stringify(final[key]));

        // If both values are null-like, they are not different.
        if (isDifferent && isNullLike(original[key]) && isNullLike(final[key])) {
          isDifferent = false;
        }

        // If only one has the key, it's definitely different
        if (onlyOneHasKey) {
          isDifferent = true;
        }

        // If one entire side is empty (i.e. a create or delete action),
        // highlighting every single row is redundant and breaks layout, so skip it.
        if (isOriginalEmpty || isFinalEmpty) {
          isDifferent = false;
        }

        const highlightClass = isDifferent ? ' highlight-diff' : '';

        // Show the data for this property, null-like values go to "".
        descriptionString += `<div class="info-row${highlightClass}"><span class="label">${escapeHtml(key)}:</span> <span class="value object-string">${displayValue(data[key])}</span></div>`;
      }
      // For all other strings, add spacing for better readability.
      // Using a temporary wrapper to avoid replacing HTML tags.
      const tempWrapper = document.createElement('div');
      tempWrapper.innerHTML = descriptionString;
      const values = tempWrapper.querySelectorAll('.value');
      values.forEach((val) => {
        if (val.innerHTML && !val.innerHTML.includes('<') && !val.innerHTML.includes('>')) {
          val.innerHTML = val.innerHTML.replaceAll(',', ', ');
        }
      });
      descriptionString = tempWrapper.innerHTML;
    }
    descriptions[type] = descriptionString;
  }

  const {original: originalString, final: finalString} = descriptions;

  const html = `
    <div class="result-description">
        <div class="result-description-item">
            <h4>Original Object</h4> 
            ${originalString}
        </div>
        <div class="result-description-item">
            <h4>Final Object</h4> 
            ${finalString}
        </div>
    </div>
    `;

  return html;
}

// Render the options for select inputs.
function renderSelectOptions(options, context, kwargs = {}) {
  let container = document;
  if (kwargs['container'] != undefined) {
    container = kwargs['container'];
  }

  const selectElement = container.querySelector(kwargs['selector']);

  // Clear existing options to prevent duplicates on re-render.
  selectElement.innerHTML = '';

  const seen = new Set();
  for (let i = 0; i < options.length; i++) {
    const option = options[i];
    let optionText = '';
    let optionValue = '';

    if (option === null || option === undefined) {
      // Skip or treat as empty
    } else if (typeof option === 'object' && option.value !== undefined && option.text !== undefined) {
      optionValue = option.value;
      optionText = option.text;
    } else {
      optionText = option.toString();
      optionValue = optionText;
    }

    // All or brackets become blank value.
    if (optionValue === 'all' || (typeof optionValue === 'string' && optionValue.includes('['))) {
      optionValue = '';
    }

    if (seen.has(optionValue)) {
      continue;
    }
    seen.add(optionValue);

    const html = `<option value="${escapeHtml(optionValue)}">${escapeHtml(optionText)}</option>`;
    const doc = document.createElement('div');
    doc.innerHTML = html;
    selectElement.appendChild(doc.querySelector('option'));
  }

  // Add "New..." option if requested and user has admin permissions (role >= 2).
  if (kwargs['allowNew'] && kwargs['type']) {
    let userRoleValue = 0;
    const currentUser = Core.getContextUser();
    if (currentUser && currentUser.user_role) {
      userRoleValue = Core.roleDict[currentUser.user_role];
    }

    if (userRoleValue >= 2) {
      const newOptionHtml = `<option value="__NEW__">New...</option>`;
      const newOptionDoc = document.createElement('div');
      newOptionDoc.innerHTML = newOptionHtml;
      selectElement.appendChild(newOptionDoc.querySelector('option'));

      // Remove old listener if present to prevent multiple bindings on re-renders
      if (selectElement._newOptionListener) {
        selectElement.removeEventListener('change', selectElement._newOptionListener);
      }

      selectElement._newOptionListener = function(event) {
        if (event.target.value === '__NEW__') {
          // Reset selection to placeholder or previous value to prevent it staying on "__NEW__"
          event.target.value = '';
          openNewOptionPopup(kwargs['type'], context, event.target);
        }
      };

      selectElement.addEventListener('change', selectElement._newOptionListener);
    }
  }

  return container;
}

// Opens a popup to create a new selector option
function openNewOptionPopup(type, context, selectElement) {
  // Determine the user-facing name for the type
  let displayType = type;
  let tableName = type;
  if (type === 'alternate_name') displayType = 'location';
  // Title case it
  displayType = displayType.charAt(0).toUpperCase() + displayType.slice(1);

  const html = `
        <div class="new-option-modal" style="position: fixed; inset: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 10000;">
            <div class="filter-popover active" style="position: relative; top: auto; left: auto; transform: none; box-shadow: 0 4px 12px rgba(0,0,0,0.2); display: flex; flex-direction: column; width: 480px; max-width: 90vw;">
                <div style="text-align: center; margin-bottom: 20px;">
                    <h4 style="margin: 0; font-size: 1.2rem;">Edit ${displayType}s</h4>
                </div>
                
                <div class="input-section" style="padding: 0; border: none; background: transparent; box-shadow: none; display: flex; flex-direction: column; gap: 15px; margin-bottom: 25px;">
                    <div class="input-item" style="margin-bottom: 0; width: 100%;">
                        <input id="new-option-input" type="text" placeholder="New ${displayType.toLowerCase()}..." autocomplete="off" style="width: 100%; box-sizing: border-box;"/>
                    </div>
                    <div style="display: flex; gap: 10px; width: 100%; flex-wrap: nowrap;">
                        <button class="save-new-option pill-btn" style="flex: 1; margin: 0; padding: 6px 16px; min-height: 36px; display: flex; align-items: center; justify-content: center; white-space: nowrap;">Add</button>
                        <button class="cancel-new-option secondary-btn" style="flex: 1; margin: 0; padding: 6px 16px; min-height: 36px; border-radius: 20px; border: 1px solid var(--border-color); background: var(--bg-secondary); cursor: pointer; display: flex; align-items: center; justify-content: center; white-space: nowrap;">Cancel</button>
                    </div>
                </div>
                
                <div style="border-top: 1px solid var(--border-color); padding-top: 15px;">
                    <h5 style="margin: 0 0 10px 0; text-align: center;">Existing ${displayType}s</h5>
                    <div class="existing-options-list" style="max-height: 250px; overflow-y: auto; overflow-x: hidden; padding: 5px 0;">
                        <div class="object-list popup-object-list" style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; align-items: center; margin: 0;">Loading...</div>
                    </div>
                </div>
            </div>
        </div>
    `;

  const doc = document.createElement('div');
  doc.innerHTML = html;
  const modal = doc.querySelector('.new-option-modal');

  const cleanUp = () => {
    document.removeEventListener('keydown', handleEscape);
    modal.remove();
  };

  const handleEscape = (e) => {
    if (e.key === 'Escape') {
      cleanUp();
    }
  };

  document.addEventListener('keydown', handleEscape);

  modal.querySelector('.cancel-new-option').addEventListener('click', cleanUp);

  modal.querySelector('.save-new-option').addEventListener('click', () => {
    const inputVal = modal.querySelector('#new-option-input').value.trim();
    if (!inputVal) {
      alert('Please provide a valid input.');
      return;
    }
    if (validName(inputVal) === false) {
      alert('Name input contains disallowed characters.');
      return;
    }

    // Save the new option
    saveNewOption(type, inputVal, context, selectElement, cleanUp);
  });

  document.body.appendChild(modal);
  modal.querySelector('#new-option-input').focus();

  // Fetch existing options and populate list
  SharkMatcher.Individual.get_individuals(-1, null, null, true, true)
      .then(function(result) {
        const utilityIndividual = result[0];
        let objectList = [];
        // "alternate_name" is typed as "alternate_names" in the individual object payload
        if (tableName === 'alternate_name' || tableName === 'location') {
          objectList = utilityIndividual.alternate_names;
          tableName = 'alternate_name'; // Ensure deletions hit the right table
        } else if (tableName === 'tag') {
          objectList = utilityIndividual.tags;
        } else if (tableName === 'label') {
          objectList = utilityIndividual.labels;
        }

        const listContainer = modal.querySelector('.object-list');
        listContainer.innerHTML = '';

        if (!objectList || objectList.length === 0) {
          listContainer.innerHTML = '<em>None</em>';
          return;
        }

        for (let i = 0; i < objectList.length; i++) {
          const item = objectList[i];
          let typeString = '';
          if (typedObjects.includes(tableName) && item.type) {
            typeString = `, ${escapeHtml(item.type)}`;
          }
          const itemHtml = `
                    <div class="object-list-item" object-id="${item.id}">
                        <div>${escapeHtml(item.name)}${typeString}</div>
                        <button class="object delete" object-id="${item.id}">Delete</button>
                    </div>
                `;
          const itemDoc = document.createElement('div');
          itemDoc.innerHTML = itemHtml;
          const domItem = itemDoc.querySelector('.object-list-item');

          domItem.querySelector('.delete').addEventListener('click', (e) => {
            if (!confirm(`Are you sure you want to delete ${item.name}? This cannot be undone.`)) return;
            SharkMatcher.Individual.delete_object(item.id, tableName)
                .then(() => {
                  refreshCache(SELECTORS_KEY, context, true).then(() => {
                    alert(`Successfully deleted ${item.name}.`);
                    location.reload(true);
                  });
                })
                .catch((err) => {
                  const message = handleAPIError(err);
                  if (message) alert(`Error deleting: ${message}`);
                });
          });

          listContainer.appendChild(domItem);
        }
      })
      .catch((err) => {
        console.error(err);
        modal.querySelector('.object-list').innerHTML = '<em>Failed to load options.</em>';
      });
}

// Internal function to save the new option to the utility individual (id = -1)
function saveNewOption(tableName, nameValue, context, selectElement, cleanUpCallback) {
  // The util individual id is -1
  SharkMatcher.Individual.post_object(null, -1, nameValue, null, null, tableName)
      .then(function(result) {
        // Success, update local selector cache
        refreshCache(SELECTORS_KEY, context, true).then(() => {
          cleanUpCallback();
          alert(`Successfully added new ${tableName}.`);
          // Reload to reflect new option everywhere
          location.reload(true);
        });
      })
      .catch(function(error) {
        console.log('Error saving new selector option.');
        const message = handleAPIError(error);
        if (message) {
          alert(`Error saving new option: ${message}`);
        }
      });
}

// Reference data cached in the browser. Each entry is a cache key, a TTL, and a
// fetcher; SharkMatcher.getCached owns the storage, expiry, in-flight collapsing
// and corrupt-entry recovery for all of them.
const CACHE_TTL_MS = 5 * 60 * 1000;
const SELECTORS_KEY = 'selectors';
const USERS_KEY = 'users';
const DATASET_STATS_KEY = 'dataset_stats';

// Selector options live on the "utility individual" (id -1). Only the names are
// kept; blank entries are dropped and each list is sorted for display.
function fetchSelectorOptions() {
  return SharkMatcher.Individual.get_individuals(-1, null, null, true, true)
      .then(function(result) {
        const utilityIndividual = (result && result[0]) || {};
        const names = (items) => (items || [])
            .filter((x) => x.name && x.name.trim() !== '')
            .map((x) => x.name)
            .sort();

        return {
          'labels': names(utilityIndividual.labels),
          'locations': names(utilityIndividual.alternate_names),
          'tags': names(utilityIndividual.tags),
        };
      });
}

function fetchUsers() {
  return SharkMatcher.User.filter_users(null, null, null)
      .then(function(result) {
        // Only id and name are ever read; drop the rest rather than persist
        // the staff directory's emails in every browser.
        return (result || [])
            .map((user) => ({'id': user.id, 'name': user.name || user.username}))
            .sort((a, b) => (a.name || '').localeCompare(b.name || ''));
      });
}

// Registry mapping cache keys to their async fetcher functions.
const CACHE_FETCHERS = {
  [SELECTORS_KEY]: () => fetchSelectorOptions(),
  [USERS_KEY]: () => fetchUsers(),
  [DATASET_STATS_KEY]: () => SharkMatcher.get_dataset_stats(),
};

// Validate a name against the registry so a typo throws instead of silently
// caching under a key nothing reads.
function cacheName(name) {
  if (!Object.prototype.hasOwnProperty.call(CACHE_FETCHERS, name)) {
    throw new Error(`Unknown cache '${name}'.`);
  }

  return name;
}

// Foreground read: resolves with the value, fetching if stale, and rejects if the
// fetch fails so the caller can surface the error to the user.
function loadCache(name, forceUpdate = false) {
  if (forceUpdate) {
    SharkMatcher.clearCache(cacheName(name));
  }

  return SharkMatcher.getCached(cacheName(name), CACHE_TTL_MS, CACHE_FETCHERS[name]);
}

// Warm cache in the background without throwing errors on failure.
function refreshCache(name, context, forceUpdate = false) {
  return loadCache(name, forceUpdate)
      .catch(function(error) {
        console.log(`Error refreshing cached '${name}'.`);
        Log.warn(error, context, false);
      });
}

// Invalidate the specified cache key so the next read fetches fresh data.
function clearCache(name) {
  SharkMatcher.clearCache(cacheName(name));
}

// Return cached selector options with fallbacks to defaults.
function getSelectorOptions() {
  const selectors = SharkMatcher.getCachedValue(SELECTORS_KEY) || {};
  return [
    selectors.labels || labelOptions,
    selectors.locations || locationOptions,
    selectors.tags || tagOptions,
  ];
}

function getLocalUsers() {
  return SharkMatcher.getCachedValue(USERS_KEY) || [];
}

function getStatsField(field) {
  const stats = SharkMatcher.getCachedValue(DATASET_STATS_KEY);
  return (stats && stats[field]) || [];
}

function getUnembeddedImagesCache() {
  return getStatsField('media_without_embedding_names');
}

function getCachedLoadedModels() {
  return getStatsField('loaded_models');
}

function getCachedMatchMethods() {
  return getStatsField('match_methods');
}

// Render the result count div at the top of a result section.
function renderResultCount(resultCount, customString = '') {
  let plural = '';
  if (resultCount !== 1) {
    plural = 's';
  }
  const html = `<h3>${resultCount} Result${plural}${customString}</h3>`;
  return html;
}

// Matches allowed name characters (letters, digits, spaces, punctuation) while excluding injection
// characters.
// Kept in sync with valid_name() in core/app_utils.py to provide client-side input validation.
const NAME_PATTERN = /^[\p{L}\p{N}_ \-'.,()/]*$/u;
const MAX_NAME_LENGTH = 128;

// Return true if a name is valid, else false.
function validName(nameInput) {
  if (nameInput === null || nameInput === undefined) return true;
  if (typeof nameInput !== 'string') return false;
  if (nameInput.length > MAX_NAME_LENGTH) return false;
  return NAME_PATTERN.test(nameInput);
}

// Escape text for safe interpolation into innerHTML.
function escapeHtml(value) {
  if (value === null || value === undefined) return '';
  return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
}

const RESULTS_PER_LOAD = 50;
const PAGINATE = true;
const PAGE_SIZE = 200;
function computeResultBounds(resultLength, loadIndex) {
  const resultMinimum = 0 + (RESULTS_PER_LOAD * (loadIndex - 1));
  const resultMaximum = Math.min(resultLength, (RESULTS_PER_LOAD * loadIndex));
  const loadMoreButton = document.querySelector('.load-more-button');
  if (resultLength > resultMaximum) {
    loadMoreButton.style.display = 'block';
  } else {
    loadMoreButton.style.display = 'none';
  }
  return [resultMinimum, resultMaximum];
}

// Extract the reference individual by following the reference pointer into the
// list of associated individuals.
function individualGetReferenceIndividual(individual) {
  let sourceIndividualReferencePointer;

  if (!individual) return undefined;

  // Non-source individuals require additional reference-pointer lookup.
  if (individual.source === individual.id) {
    sourceIndividualReferencePointer = individual.reference;
  } else {
    // The source individual is the first associated individual.
    if (individual.associated_individuals &&
        individual.associated_individuals.length > 0) {
      sourceIndividualReferencePointer = individual.associated_individuals[0].reference;
    } else {
      console.log('Warning, reference individual not found, is this a FULL source individual?');
      return individual;
    }
  }

  // Find the reference individual among the associated individuals, this is what will be dispyed.
  if (individual.associated_individuals) {
    for (const associatedIndividual of individual.associated_individuals) {
      if (associatedIndividual != null) {
        if (associatedIndividual.id === sourceIndividualReferencePointer) {
          return associatedIndividual;
        }
      }
    }
  }

  console.log('Warning, reference individual not found, is this a FULL source individual?');
  return individual;
}

// Note that this will be easier once Uint8Array.fromBase64() is widely supported.
// https://developer.mozilla.org/en-US/docs/Web/API/Window/btoa#unicode_strings
function textToB64String(text) {
  const bytes = textToBytes(text);
  const binaryString = Array.from(bytes, function(byte) {
    return String.fromCodePoint(byte);
  }).join('');

  return btoa(binaryString);
}
function b64StringToText(contentB64) {
  const binaryString = atob(contentB64);
  const bytes = Uint8Array.from(binaryString, function(element) {
    return element.codePointAt(0);
  });

  const decoder = new TextDecoder('utf-8');
  return decoder.decode(bytes);
}

// Remove search filter parameters from the URL and reload, keeping parameters a page needs,
// such as 'id' and 'method' on the match page.
function clearSearchFilters() {
  const [path, params] = Routes.parsePath(window.location.hash);
  const preservedParams = {};

  // Keep any non filter params in place.
  if (path === 'match') {
    for (const key of ['name', 'method', 'model']) {
      if (params[key]) {
        preservedParams[key] = params[key];
      }
    }
  }

  // Clear the filter parameters stored locally for this tab only.
  SharkMatcher.clearFilters(path);

  // Setting the hash triggers a re-render; reload if the hash is unchanged.
  const newPath = Core.formHashPath(path, preservedParams);
  if (newPath !== window.location.hash) {
    window.location.hash = newPath;
  } else {
    location.reload(true);
  }
}

// Remove every query parameter from the current hash, for pages that store no filters locally.
function clearUrlParams() {
  const [path] = Routes.parsePath(window.location.hash);

  // Form a new hash with just the page name, effectively clearing params.
  const newPath = Core.formHashPath(path, {});

  if (newPath !== window.location.hash) {
    window.location.hash = newPath;
  } else {
    location.reload(true);
  }
}

// WebKit rejects in-flight requests while the page unloads, and those are not errors to show.
let pageUnloading = false;
window.addEventListener('beforeunload', () => {
  pageUnloading = true;
});

// Handle a standardized API error, returning its message when the caller must still act or
// null when it was fully handled, such as an alerted authentication error.
function handleAPIError(error) {
  if (pageUnloading && error?.status === 0) {
    return null;
  }
  if (error && error.isAuthError) {
    alert(`Authentication Error: ${error.message}`);
    return null;
  }

  // Return the message for the caller to handle specific logic.
  return error.message || error.toString();
}

// Constants.
const UNKNOWN_VALUE = 'Unknown';

const IMAGES_URL = '/static/frontend-images/';
const labelOptions = ['curated-individual', 'fresh', 'todo', 'unmatched', 'proposal'];
const locationOptions = ['AN', 'FAR', 'MB', 'MO', 'NB', 'PR', UNKNOWN_VALUE];
const sexOptions = ['F', 'M', UNKNOWN_VALUE];
const qualityOptions = ['poor', 'medium', 'good', 'unmatchable', UNKNOWN_VALUE];
const tagOptions = ['unknown', 'acoustic', 'satellite', 'camera'];
const typeOptions = ['topshot', 'underwater', UNKNOWN_VALUE];
const typedObjects = ['tag'];
const timedObjects = ['tag'];

const hiddenKeys = ['hash', 'metadata'];

const titles = {
  'export': 'Export Current Dataset',
  'history': 'Action History',
  'match': 'Match',
  'merge': 'Merge',
  'proposals': 'Proposals',
  'review': 'Review',
  'search': 'Search',
  'settings': 'Settings',
  'individual': 'Individual',
  'upload': 'Upload',
};
const instructions = {
  'export': 'Download a zip file containing the current image library and annotations information.',
  'history': 'View a history of all actions performed on the data. Filter and sort actions to see and understand previous changes made.',
  'match': 'Query an individual and see the match results provided by Shark Matcher with the option to filter match results. View images in focus mode by double clicking on any query or result image.',
  'merge': 'View the 2 input individuals (leftmost columns) and a preview of the merged output individual that will result. Modify properties or associated objects of the output individual before saving the merge.',
  'proposals': 'View and modify proposed updates to individuals and images. Admin users can approve, decline, or modify proposed udpates.',
  'review': 'Review individuals using the configured filters. Apply filters and click the Review button to begin.',
  'search': 'Search individuals by keywords with query search. Keywords may include individual id numbers, nicknames, tags, labels, or image names. Advanced search allows searching by a multitude of associated objects or properties.',
  'settings': 'View and edit your user profile.',
  'individual': 'View and edit an individual\'s properties, associated objects, and associated images. Select images from the gallery to view or edit the image\'s properties. Images may also be set as the new reference image for this individual or unmatched to be their own individual that may be rematched.',
  'upload': 'Upload new images to Shark Matcher. Drop image files and update image properties before uploading. All uploaded images become new unmatched individuals that may then be matched.',
};

// --- Smart Search Utilities ---

// Return the HTML for a tab's smart search bar, prefilled through inputValueAttr (`value="foo"`).
function buildSmartSearchBar(
    tabName, placeholder, inputValueAttr, extraInputClass = '', sortDisplay = '') {
  return `
        <div class="search-controls-row">
            <div class="controls-right-side">
                <div class="input-section sort" ${sortDisplay}></div>
            </div>
            <div class="smart-search-container">
                <input
                    class="smart-search-input ${tabName} ${extraInputClass}"
                    id="${tabName}-keywords-input"
                    type="text"
                    placeholder="${placeholder}"
                    ${inputValueAttr}/>
                <button class="filter-toggle-btn" title="Toggle Filters">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
                    </svg>
                </button>
                <div class="filter-popover" id="${tabName}-filter-popover"></div>
            </div>
        </div>
    `;
}

// Bind the smart search bar's filter toggle and click-away listeners before the bar is inserted.
function bindSmartSearchEvents(doc, tabName) {
  const filterBtn = doc.querySelector('.filter-toggle-btn');
  const searchContainer = doc.querySelector('.smart-search-container');

  if (filterBtn) {
    filterBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      const currentPopover = document.getElementById(`${tabName}-filter-popover`);
      if (currentPopover) {
        currentPopover.classList.toggle('active');
      }
    });
  }

  if (searchContainer) {
    searchContainer.addEventListener('click', function(e) {
      e.stopPropagation();
    });
  }

  document.addEventListener('click', function(e) {
    const currentPopover = document.getElementById(`${tabName}-filter-popover`);
    const currentContainer = currentPopover ? currentPopover.closest('.smart-search-container') : null;

    if (currentPopover && currentPopover.classList.contains('active')) {
      if (currentContainer && !currentContainer.contains(e.target)) {
        currentPopover.classList.remove('active');
      }
    }
  });

  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      const currentPopover = document.getElementById(`${tabName}-filter-popover`);
      if (currentPopover && currentPopover.classList.contains('active')) {
        currentPopover.classList.remove('active');
      }
    }
  });
}

// Read shared search filter inputs from the active filter popover into an API parameter object.
function collectSearchFilters() {
  const alternateNames = document.querySelector('.search.alternate_name-input').value;
  const tags = document.querySelector('.search.tag-input').value;
  const labels = document.querySelector('.search.label-input').value;
  const minAssociated = document.querySelector('.search.min_associated-input').value;
  const maxAssociated = document.querySelector('.search.max_associated-input').value;

  const selectedLocations = Array.from(document.querySelectorAll(
      '.search-location-dropdown input[type="checkbox"]:checked'))
      .map((cb) => cb.value);
  const locations = selectedLocations.join(',');
  const qualities = document.querySelector('.search.min-quality-input').value;
  const maxQualities = document.querySelector('.search.max-quality-input').value;
  const sex = document.querySelector('.search.sex-input').value;
  const minLength = document.querySelector('.search.min_length-input').value;
  const maxLength = document.querySelector('.search.max_length-input').value;
  const minTime = inputToTimestamp(
      document.querySelector('.search.min_time-input').value, false);
  const maxTime = inputToTimestamp(
      document.querySelector('.search.max_time-input').value, false);
  const allowUnknown = document.querySelector('.search.allow_unknown-input').checked;

  return {
    'min_length': minLength,
    'max_length': maxLength,
    'min_time': minTime,
    'max_time': maxTime,
    'min_associated': minAssociated,
    'max_associated': maxAssociated,
    'sex': sex,
    'locations': locations,
    'qualities': qualities,
    'max_qualities': maxQualities,
    'alternate_names': alternateNames,
    'labels': labels,
    'tags': tags,
    'allow_unknown': allowUnknown,
  };
}

// The filter fields that can appear on a path. Used to decide whether a URL
// already carries filter state, in which case stored filters must not override it.
const FILTER_PARAM_KEYS = [
  'keywords', 'min_length', 'max_length', 'min_time', 'max_time',
  'min_associated', 'max_associated', 'sex', 'locations', 'qualities',
  'max_qualities', 'alternate_names', 'labels', 'tags', 'allow_unknown',
];

function hasFilterParams(params) {
  if (!params) {
    return false;
  }
  return FILTER_PARAM_KEYS.some(
      (key) => params[key] !== undefined && params[key] !== '');
}

// Mark a tab's filter toggle button when any filter in its popover is active.
function updateFilterIndicator(tabName) {
  const popover = document.getElementById(`${tabName}-filter-popover`);

  // Find the toggle button associated with this popover.
  // It's normally a sibling of the popover inside the smart-search-container
  let toggleBtn = null;
  if (popover && popover.parentElement) {
    toggleBtn = popover.parentElement.querySelector('.filter-toggle-btn');
  }

  if (!popover || !toggleBtn) return;

  let isActive = false;

  // Check text, number, and date inputs
  const inputs = popover.querySelectorAll('input[type="text"], input[type="number"], input[type="date"]');
  inputs.forEach((input) => {
    if (input.value && input.value.trim() !== '') {
      isActive = true;
    }
  });

  // Check select inputs
  const selects = popover.querySelectorAll('select');
  selects.forEach((select) => {
    // 'all' or empty are default placeholder values
    if (select.value && select.value !== '' && select.value !== 'all') {
      // Special exception: Proposals route defaults to 'open' status.
      if (tabName === 'proposals' &&
          select.classList.contains('status-input') &&
          select.value === 'open') {
        return; // Do not flag as active
      }
      // Match method and model selections do not activate the filter indicator.
      if (tabName === 'match' &&
          (select.classList.contains('method-input') ||
           select.classList.contains('model-input'))) {
        return; // Do not flag as active
      }
      // Same exception for the Review route's model selector.
      if (tabName === 'review' && select.classList.contains('model-input')) {
        return; // Do not flag as active
      }
      isActive = true;
    }
  });

  // Check checkboxes (specifically in location dropdowns or allow_unknown)
  const checkboxes = popover.querySelectorAll('input[type="checkbox"]');
  checkboxes.forEach((checkbox) => {
    if (checkbox.classList.contains('allow_unknown-input') || checkbox.id.includes('allow_unknown')) {
      // allow_unknown defaults to true. Unchecking it means filtering is active.
      if (!checkbox.checked) {
        isActive = true;
      }
    } else {
      // For location checkboxes, if any are checked, it's filtering.
      if (checkbox.checked) {
        isActive = true;
      }
    }
  });

  if (isActive) {
    toggleBtn.classList.add('has-active-filters');
  } else {
    toggleBtn.classList.remove('has-active-filters');
  }
}

// Constraint color conventions: green for match, yellow for partial/unknown, red for conflict, and
// null for uninformative.
// Return the constraint color for a sex comparison ('F', 'M', or unknown).
function sexConstraintColor(querySex, resultSex) {
  // Normalize unknown representations to null for clean comparisons.
  const normalizedQuery = (!querySex || querySex === UNKNOWN_VALUE) ? null : querySex;
  const normalizedResult = (!resultSex || resultSex === UNKNOWN_VALUE) ? null : resultSex;

  const queryUnknown = normalizedQuery === null;
  const resultUnknown = normalizedResult === null;

  // Both unknown: no meaningful comparison.
  if (queryUnknown && resultUnknown) {
    return null;
  }

  // One is unknown: partial agreement.
  if (queryUnknown || resultUnknown) {
    return 'yellow';
  }

  // Both known: check for agreement or strict disagreement.
  if (normalizedQuery === normalizedResult) {
    return 'green';
  }
  return 'red';
}

// Return constraint color for length difference in feet: <=2 green, <=4 yellow, >4 red, or null if
// both unknown.
function lengthConstraintColor(queryLength, resultLength) {
  const queryUnknown = (queryLength === null || queryLength === undefined);
  const resultUnknown = (resultLength === null || resultLength === undefined);

  // Both unknown: no meaningful comparison.
  if (queryUnknown && resultUnknown) {
    return null;
  }

  // One is unknown: partial agreement.
  if (queryUnknown || resultUnknown) {
    return 'yellow';
  }

  // Both known: compare the absolute difference against the thresholds.
  const diff = Math.abs(queryLength - resultLength);
  if (diff <= 2) {
    return 'green';
  }
  if (diff <= 4) {
    return 'yellow';
  }
  return 'red';
}

// Inject a retrieval method selector into the filter popover, populated from cached match methods.
function renderMethodSelector(tabPrefix, params, context) {
  const popover = document.querySelector('.filter-popover');
  const html = `
        <div class="input-item">
            <label for="${tabPrefix}-method-input">Retrieval Method: </label>
            <select class="${tabPrefix} method-input" id="${tabPrefix}-method-input"
                name="${tabPrefix} method-input"></select>
        </div>
    `;
  popover.insertAdjacentHTML('afterbegin', html);

  let placeholder = 'nearest neighbor';
  if (params.method && params.method !== '') {
    placeholder = params.method;
  }

  const matchMethods = getCachedMatchMethods();

  if (matchMethods.length > 0 && !matchMethods.includes(placeholder)) {
    placeholder = matchMethods[0];
  }

  const options = [];
  if (placeholder !== '') options.push(placeholder);
  options.push(...matchMethods.filter((m) => m !== placeholder));
  if (options.length === 0) options.push('nearest neighbor');

  const selector = `.${tabPrefix}.method-input`;
  renderSelectOptions(options, context, {'selector': selector});

  loadCache(DATASET_STATS_KEY).then(function(result) {
    if (result && result.match_methods && result.match_methods.length > 0) {
      const currentVal = document.querySelector(selector).value;
      const newPlaceholder = result.match_methods.includes(currentVal) ?
                currentVal : result.match_methods[0];

      const freshOptions = [
        newPlaceholder,
        ...result.match_methods.filter((m) => m !== newPlaceholder)];
      renderSelectOptions(freshOptions, context, {'selector': selector});
    }
  }).catch(function(e) {
    console.log(`Failed to asynchronously refresh match methods for ${tabPrefix} tab:`, e);
  });
}

// Inject a model selector into the filter popover, populated from cached loaded models.
function renderModelSelector(tabPrefix, params, context) {
  const popover = document.querySelector('.filter-popover');
  const html = `
        <div class="input-item">
            <label for="${tabPrefix}-model-input">Model: </label>
            <select class="${tabPrefix} model-input" id="${tabPrefix}-model-input"
                name="${tabPrefix} model-input"></select>
        </div>
    `;
  popover.insertAdjacentHTML('afterbegin', html);

  let placeholder = '';
  if (params.model && params.model !== '') {
    placeholder = params.model;
  }

  const loadedModels = getCachedLoadedModels();

  if (loadedModels.length > 0 &&
        (placeholder === '' || !loadedModels.includes(placeholder))) {
    placeholder = loadedModels[0];
  }

  const options = [];
  if (placeholder !== '') options.push(placeholder);
  options.push(...loadedModels.filter((m) => m !== placeholder));
  if (options.length === 0) options.push('Loading Models...');

  const selector = `.${tabPrefix}.model-input`;
  renderSelectOptions(options, context, {'selector': selector});

  // Asynchronously refresh the dropdown to reflect the latest loaded models.
  loadCache(DATASET_STATS_KEY).then(function(result) {
    if (result && result.loaded_models && result.loaded_models.length > 0) {
      const currentVal = document.querySelector(selector).value;
      let newPlaceholder = currentVal;
      if (newPlaceholder === 'Loading Models...' ||
                !result.loaded_models.includes(newPlaceholder)) {
        newPlaceholder = result.loaded_models[0];
      }

      const freshOptions = [
        newPlaceholder,
        ...result.loaded_models.filter((m) => m !== newPlaceholder)];
      renderSelectOptions(
          freshOptions, context, {'selector': selector});
    }
  }).catch(function(e) {
    console.log(
        `Failed to asynchronously refresh loaded models for ${tabPrefix} tab:`, e);
  });
}


// Render a pill-style dropdown button that toggles a popover, returning the created elements.
function renderPillSelector(kwargs = {}) {
  const html = `
    <div class="input-section sort" style="display: flex; gap: 10px; align-items: center; border: none; padding: 0; background: none; box-shadow: none;">
      <div class="sort-container ${kwargs.containerClass || ''}">
        <button class="sort-pill-btn">
          <span class="sort-icon">${kwargs.icon || ''}</span> ${kwargs.prefix || ''} <span class="sort-label">${kwargs.labelText || ''}</span>
        </button>
        <div class="sort-popover">
          ${kwargs.bodyHtml || ''}
        </div>
      </div>
    </div>
  `;

  const doc = document.createElement('div');
  doc.innerHTML = html;
  const wrapper = doc.querySelector('.input-section.sort');
  const container = wrapper.querySelector('.sort-container');
  const pillBtn = container.querySelector('.sort-pill-btn');
  const popover = container.querySelector('.sort-popover');

  pillBtn.addEventListener('click', () => {
    popover.classList.toggle('active');
  });

  document.addEventListener('click', (event) => {
    if (!container.contains(event.target)) {
      popover.classList.remove('active');
    }
  });

  document.querySelector(kwargs.selector).replaceWith(wrapper);
  return {wrapper, container, pillBtn, popover};
}


// Future: dateConstraintColor(queryTime, resultTime) { ... }
// Future: locationConstraintColor(queryLocation, resultLocation) { ... }

// Map a semantic constraint color ('green', 'yellow', 'red') to its CSS class name.
// Returns null when color is null so callers can cleanly skip class addition.
function constraintColorClass(color) {
  if (color === 'green') return 'constraint-agree';
  if (color === 'yellow') return 'constraint-partial';
  if (color === 'red') return 'constraint-disagree';
  return null;
}

export {
  sha256,
  loadingStart,
  loadingStop,
  caseInsensitiveStringCompare,
  getLocationHash,
  init,
  onKeyEvent,
  queryAncestor,
  registerHandler,
  listToString,
  listToLinks,
  timestampToString,
  inputToTimestamp,
  toggleVisibility,
  sortMediaReferenceFirst,
  sortList,
  sortByAttribute,
  renderSelectOptions,
  getSelectorOptions,
  loadCache,
  refreshCache,
  clearCache,
  getLocalUsers,
  getUnembeddedImagesCache,
  getCachedLoadedModels,
  getCachedMatchMethods,
  renderResultCount,
  timestampToInput,
  validName,
  escapeHtml,
  individualGetReferenceIndividual,
  computeResultBounds,
  textToB64String,
  b64StringToText,
  clearSearchFilters,
  clearUrlParams,
  renderResultDescription,
  handleAPIError,
  displayValue,
  getDisplayName,
  buildSmartSearchBar,
  bindSmartSearchEvents,
  renderMethodSelector,
  renderModelSelector,
  renderPillSelector,
  collectSearchFilters,
  hasFilterParams,

  hiddenKeys,
  typedObjects,
  timedObjects,
  titles,
  instructions,
  locationOptions,
  sexOptions,
  qualityOptions,
  tagOptions,
  typeOptions,
  UNKNOWN_VALUE,
  IMAGES_URL,
  RESULTS_PER_LOAD,
  PAGINATE,
  PAGE_SIZE,
  getAuthorUsername,
  getLocalUsersFormatted,
  updateFilterIndicator,
  compactAssociatedObjects,
  sexConstraintColor,
  lengthConstraintColor,
  constraintColorClass,
};
// Cached users are {id, name} pairs, sorted by name. See fetchUsers().
function getLocalUsersFormatted() {
  return getLocalUsers().map((user) => ({value: user.id, text: user.name}));
}

function getAuthorUsername(authorId) {
  // If no author ID, just return the placeholder.
  if (authorId === undefined || authorId === null || authorId === '') return 'unknown';

  const matchedUser = getLocalUsers().find(
      (user) => String(user.id) === String(authorId));

  // Fall back to the raw ID when the users cache has not loaded yet.
  // Escaped here because both call sites render the result into markup.
  return matchedUser ? escapeHtml(matchedUser.name) : authorId;
}

function compactAssociatedObjects(individual, preserveObjects = false) {
  if (!individual) return null;
  const compacted = structuredClone(individual);

  // Shrink lists to primitive identifiers in order to reduce network payload sizes.
  // Also protect against null objects that might be present natively in frontend arrays.
  if (compacted.associated_individuals &&
      Array.isArray(compacted.associated_individuals)) {
    compacted.associated_individuals = compacted.associated_individuals
        .map((x) => x ? x.id : null)
        .filter((x) => x !== null);
  }
  if (compacted.associated_medias && Array.isArray(compacted.associated_medias)) {
    compacted.associated_medias = compacted.associated_medias
        .map((x) => x ? x.name : null)
        .filter((x) => x !== null);
  }

  if (!preserveObjects) {
    if (compacted.alternate_names && Array.isArray(compacted.alternate_names)) {
      compacted.alternate_names = compacted.alternate_names
          .map((x) => x ? x.name : null)
          .filter((x) => x !== null);
    }
    if (compacted.labels && Array.isArray(compacted.labels)) {
      compacted.labels = compacted.labels
          .map((x) => x ? x.name : null)
          .filter((x) => x !== null);
    }
    if (compacted.tags && Array.isArray(compacted.tags)) {
      compacted.tags = compacted.tags
          .map((x) => x ? x.name : null)
          .filter((x) => x !== null);
    }
  }

  return compacted;
}
