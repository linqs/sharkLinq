import * as SharkMatcher from '/js/modules/shark_matcher/base.js';
import * as Core from './core.js';
import * as Home from './home.js';
import * as Routes from './routes.js';
import * as Individual from './individual.js';
import * as Util from './util.js';

function init() {
  Routes.addRoute(/^upload$/, handlerUpload, 'upload');
}

function handlerUpload(path, params, context) {
  Home.handlerHome(path, params, context);

  // Only allow uploading in editing mode.
  const currentMode = Core.getCurrentUserMode();
  let uploadArea;
  if (currentMode === 'viewing' || currentMode === 'demo') {
    uploadArea = `
        <div class="input-item">
            <p>Uploading is only allowed in proposing or editing mode.</p>
        </div>
        `;
  } else {
    uploadArea = `
            <div class="input-item">
                <label for="upload-file-input">JPEG Images</label>
                <br />
                <input
                    class="upload file-input"
                    id="upload-file-input"
                    type="file"
                    accept="image"
                    multiple/>
            </div>
            <div class="input-item" style="display: none">
                <label for="upload-annotations-input">Upload Annotations Data</label>
                <br />
                <input
                    class="upload annotations-input"
                    id="upload-annotations-input"
                    type="file"
                    accept="image"
                    multiple/>
            </div>
        `;
  }

  // Tab content.
  let html = `
        <div class="tab-content upload">
            <div class="tab-title"></div>
            <div class="file-upload-screen">
                <div class="file-upload-area"></div>
            </div>
            <div class="input-section">
                ${uploadArea}
            </div>
        </div>
    `;
  let doc = document.createElement('div');
  doc.innerHTML = html;

  // Only allow uploading in proposing or editing mode.
  if (currentMode !== 'viewing' && currentMode !== 'demo') {
    // Handle image file uploads.
    const fileInput = doc.querySelector('.upload.file-input');
    const annotationsInput = doc.querySelector('.upload.annotations-input');
    fileInput.addEventListener('change', async function(event) {
      // Build a list of media objects to represent uploaded files.
      const inputFiles = Array.from(event.target.files);
      const previewIndividuals = buildPreviewIndividuals(inputFiles);

      try {
        // Render an area for each uploaded image.
        await processImages(previewIndividuals, inputFiles);
      } catch (error) {
        console.error('Error processing uploaded images.', error);
        const message = Util.handleAPIError(error);
        if (message) {
          alert(`Error processing uploaded images: ${message}`);
        }
        return;
      }

      renderUploadResults(previewIndividuals, context);
    });
    // TODO upload annotations files: bulk upload data
    // Handle annotations file uploads.
    annotationsInput.addEventListener('change', function(event) {
      console.log('Upload annotations files');
    });
    document.querySelector('.tab-content').replaceWith(doc.querySelector('.tab-content.upload'));

    // Populate sort options, title, filters.
    Home.renderTitle('upload', context);

    // Results content.
    html = `
        <div class="tab-results">
            <div class="results-container"></div>
        </div>
        `;
    doc = document.createElement('div');
    doc.innerHTML = html;
    document.querySelector('.tab-results').replaceWith(doc.querySelector('.tab-results'));
  } else {
    const doc = document.createElement('div');
    doc.innerHTML = html;
    document.querySelector('.tab-content').replaceWith(doc.querySelector('.tab-content.upload'));
    Home.renderTitle('upload', context);
  }
}

// Path update functions.

// API execution functions.
async function executeMediaUpload(previewIndividuals) {
  // Save all preview individuals to capture any user edits.
  for (let i = 0; i < previewIndividuals.length; i++) {
    const saveSuccess = previewExecuteIndividualSave(previewIndividuals, {'index': i});
    if (saveSuccess !== true) {
      return; // Validation failed for one individual, stop.
    }
  }

  // Build the batch items array with all image data.
  const items = [];
  for (let i = 0; i < previewIndividuals.length; i++) {
    const p = previewIndividuals[i];
    items.push({
      'name': p.name,
      'hash': p.hash,
      'metadata': p.metadata || null,
      'location': p.location,
      'time': p.time,
      'type': p.type,
      'quality': p.quality,
      'length': p.length,
      'sex': p.sex,
      'data_url': p.image.src,
      'alternate_names': p.alternate_names || [],
      'tags': p.tags || [],
      'labels': p.labels || [],
    });
  }

  // Send the entire batch in a single atomic request.
  try {
    await SharkMatcher.Media.upload_batch(items);

    // Invalidate the stats cache so the next page load picks up the new images.
    Util.clearCache('dataset_stats');

    // Navigate to the uploaded view.
    const uploadedPath = Core.formHashPath('search', {
      'keywords': 'fresh',
      'attribute': 'name',
      'order': 'descending',
    });
    window.location.hash = uploadedPath;
  } catch (error) {
    console.error('Batch upload failed:', error);
    const message = Util.handleAPIError(error);
    if (message) {
      // Try to parse the error for detailed per-image info.
      try {
        const parsed = JSON.parse(message);
        if (parsed && parsed.errors && Array.isArray(parsed.errors)) {
          let errorDetails = 'Upload failed. No images were uploaded.\n\n';
          for (const err of parsed.errors) {
            errorDetails += `• ${err.filename}: ${err.reason}\n`;
          }
          alert(errorDetails);
        } else if (parsed && parsed.message) {
          alert(parsed.message);
        } else {
          alert(`Upload failed: ${message}`);
        }
      } catch {
        alert(`Upload failed: ${message}`);
      }
    }
  }
}

function previewExecuteIndividualDelete(previewIndividuals, context, kwargs) {
  const index = kwargs['index'];

  // Update list of input files.
  const newPreviewIndividuals = [];
  for (let i = 0; i < previewIndividuals.length; i++) {
    // Save all changes made to the preview.
    previewExecuteIndividualSave(previewIndividuals, {'index': i});

    const previewIndividual = previewIndividuals[i];
    // Id always applies to list position.
    previewIndividual.id = i;
    if (i !== index) {
      newPreviewIndividuals.push(previewIndividual);
    }
  }
  renderUploadResults(newPreviewIndividuals, context);
}

function previewExecuteIndividualSave(previewIndividuals, kwargs) {
  const index = kwargs['index'];

  // Get and validate inputs. TODO, allow like"ish" inputs, validate inputs.
  const nameInput = document.getElementById(`media-name-input-${index}`).value;
  const locationInput = document.getElementById(`media-location-input-${index}`).value;
  const timeInput = document.getElementById(`media-time-input-${index}`).value;
  const typeInput = document.getElementById(`media-type-input-${index}`).value;
  const sexInput = document.getElementById(`media-sex-input-${index}`).value;
  const lengthInput = document.getElementById(`media-length-input-${index}`).value;
  const qualityInput = document.getElementById(`media-min-quality-input-${index}`).value;

  // Update individual preview.
  const previewIndividual = previewIndividuals[index];
  if (Util.validName(nameInput) === false) {
    alert('Name input contains disallowed characters.');
    return false;
  }

  previewIndividual.name = nameInput;
  previewIndividual.filename = nameInput + '.jpeg';
  previewIndividual.sex = sexInput;
  previewIndividual.type = typeInput;
  previewIndividual.location = locationInput;
  previewIndividual.time = Util.inputToTimestamp(timeInput);
  previewIndividual.length = parseFloat(lengthInput);
  if (isNaN(previewIndividual.length)) {
    previewIndividual.length = Util.UNKNOWN_VALUE;
  }
  previewIndividual.quality = qualityInput;

  return true;
}

// HTML render functions.
function renderUploadResults(previewIndividuals, context) {
  // Clear old result.
  const html = `
    <div class="results-container">
        <button class="upload save">Upload Files</button>
    </div>
    `;
  const doc = document.createElement('div');
  doc.innerHTML = html;
  const saveButton = doc.querySelector('.upload.save');
  saveButton.addEventListener('click', function(event) {
    executeMediaUpload(previewIndividuals);
  });
  document.querySelector('.results-container').replaceWith(doc.querySelector('.results-container'));

  // Render list of individual upload objects.
  for (let i = 0; i < previewIndividuals.length; i++) {
    renderUploadResult(previewIndividuals, context, {'index': i});
  }
}

function renderUploadResult(previewIndividuals, context, kwargs = {}) {
  // Save all changes made to the individuals (if re-rendering), then render.
  try {
    previewExecuteIndividualSave(previewIndividuals, kwargs);
  } catch { }

  const previewIndividual = previewIndividuals[kwargs['index']];
  const html = `
    <div class="results-item" object-id=${previewIndividual.id}>
        <div class="canvas-container">
            <canvas class="canvas"></canvas>
        </div>
        <div class="individual-edit"></div>
        <div class="media-menu"></div>
        <button class="result delete" object-id=${previewIndividual.id}>Delete</button>
    </div>
    `;
  let doc = document.createElement('div');
  doc.innerHTML = html;

  // Build canvas with width of 500, dynamic height, populate with image.
  const canvas = doc.querySelector('canvas');
  const scaleInfo = scaleImage(previewIndividual.image, 500, 500);
  canvas.width = 500;
  canvas.height = parseInt(scaleInfo['newHeight']) + 1;
  const canvasContext = canvas.getContext('2d');
  canvasContext.drawImage(previewIndividual.image, 0, 0, scaleInfo['newWidth'], scaleInfo['newHeight']);

  const deleteButton = doc.querySelector('.result.delete');
  deleteButton.addEventListener('click', function() {
    previewExecuteIndividualDelete(previewIndividuals, context, {'index': kwargs['index']});
  });

  doc = Individual.renderIndividualEdit(previewIndividuals, context, {'container': doc, 'selector': '.individual-edit', 'index': kwargs['index'], 'page': 'upload'});
  doc = Home.renderObjectList(previewIndividuals, 'alternate_name', context, {'container': doc, 'selector': '.alternate_name .object-list', 'index': kwargs['index'], 'page': 'upload'});
  doc = Home.renderObjectList(previewIndividuals, 'tag', context, {'container': doc, 'selector': '.tag .object-list', 'index': kwargs['index'], 'page': 'upload'});
  doc = Home.renderObjectList(previewIndividuals, 'label', context, {'container': doc, 'selector': '.label .object-list', 'index': kwargs['index'], 'page': 'upload'});
  doc = Home.renderMediaEdit(previewIndividuals, context, {'container': doc, 'selector': '.media-menu', 'index': kwargs['index'], 'page': 'upload'});

  // Replace old result, or create a new list item
  const previousResult = document.querySelector(`.results-item[object-id="${previewIndividual.id}"]`);
  if (previousResult !== null) {
    const parent = previousResult.parentElement;
    parent.replaceChild(doc.querySelector('div'), previousResult);
  } else {
    document.querySelector('.results-container').prepend(doc.querySelector('.results-item'));
  }
}

// Helper functions
// Filename sex letters map to the Sex selector values F, M and Unknown; other letters stay empty.
function sexFromNameLetter(letter) {
  const sex = letter.toUpperCase();
  if (sex === 'F' || sex === 'M') {
    return sex;
  }
  return sex === 'U' ? Util.UNKNOWN_VALUE : null;
}

function parseFileName(file) {
  const splitName = file.name.split('.');
  const fileName = splitName[0];
  const now = Math.floor((new Date().getTime() / 1000));

  // Default object.
  const previewIndividual = {
    'id': undefined,
    'name': fileName,
    'filename': fileName + '.jpeg',
    'hash': null,
    'type': 'underwater',
    'location': null,
    'length': null,
    'sex': null,
    'source': null,
    'reference': null,
    'time': now,
    'quality': null,
    'extension': undefined,
    'associated_medias': [],
    'associated_individuals': [],
    'alternate_names': [],
    'tags': [],
    'labels': [],
  };

  // Parse name see if a match is produced.
  const nameRegex = /^([A-Z][A-Z]+)(\d\d\d\d\d\d\d\d)(_[A-Za-z])?(_[A-Za-z]|_[\d]*[.]*[\d]+)?(\.[A-Za-z]+)*$/;
  const match = file.name.match(nameRegex);

  if (match === null) {
    return previewIndividual;
  }

  // Extract all possible information from name.
  try {
    previewIndividual.name = match[1] + match[2];
    previewIndividual.filename = previewIndividual.name + '.jpeg';
    previewIndividual.location = match[1];

    const numerical = match[2];

    if (match[3] !== undefined) {
      previewIndividual.sex = sexFromNameLetter(match[3].slice(1));
    }
    if (match[4] !== undefined) {
      previewIndividual.length = match[4].slice(1);
    }
    if (match[5] !== undefined) {
      previewIndividual.extension = match[5].slice(1);
    }

    try {
      // Parse numerical information
      const year = parseInt(numerical.slice(0, 2));
      let month = parseInt(numerical.slice(2, 4));
      const day = numerical.slice(4, 6);

      // Year logic
      let fullYear;
      if (year <= 50) {
        fullYear = 2000 + year;
      } else {
        fullYear = 1900 + year;
      }

      // Generalized Month Logic (1-based input, modulo arithmetic)
      // Month 13 -> Jan next year
      if (month > 12) {
        const monthIndex = month - 1;
        const yearsToAdd = Math.floor(monthIndex / 12);
        fullYear += yearsToAdd;
        month = (monthIndex % 12) + 1;
      }

      // Pad month for date string
      const monthStr = month.toString().padStart(2, '0');

      const dateString = fullYear + '-' + monthStr + '-' + day + 'T00:00:00Z';
      const time = (
        Date.parse(dateString) +
        (new Date(dateString).getTimezoneOffset() * 60 * 1000)
      ) / 1000;
      previewIndividual.time = time;
    } catch {
      console.log('ERROR: Unable to parse file for date:', file.name);
    }
  } catch {
    console.log('ERROR: Unable to parse file for metadata:', file.name);
  }

  return previewIndividual;
}

function buildPreviewIndividuals(inputFiles) {
  const previewIndividuals = [];
  for (let i = 0; i < inputFiles.length; i++) {
    const previewIndividual = parseFileName(inputFiles[i]);
    previewIndividual.id = i;
    previewIndividuals.push(previewIndividual);
  }
  return previewIndividuals;
}

async function processImages(previewIndividuals, inputFiles) {
  for (let i = 0; i < inputFiles.length; i++) {
    await processImage(previewIndividuals, inputFiles, i);
  }
}

function processImage(previewIndividuals, inputFiles, i) {
  return new Promise((resolve, reject) => {
    // Populate the canvas for this uploaded image.
    const inputFile = inputFiles[i];
    const reader = new FileReader();
    reader.readAsDataURL(inputFile);
    reader.onloadend = function(innerEvent) {
      previewIndividuals[i].image = new Image();
      previewIndividuals[i].image.src = innerEvent.target.result;
      previewIndividuals[i].image.onload = async function() {
        try {
          // Hashing the data_url to store in db.
          previewIndividuals[i].hash = await Util.sha256(
              previewIndividuals[i].image.src);
          resolve();
        } catch (error) {
          reject(error);
        }
      };
    };
  });
}

function scaleImage(img, canvasWidth, canvasHeight) {
  const imageWidth = img.width;
  const imageHeight = img.height;

  const scaleFactor = Math.min(
      canvasWidth / imageWidth,
      canvasHeight / imageHeight,
  );

  const newWidth = imageWidth * scaleFactor;
  const newHeight = imageHeight * scaleFactor;

  return {'newWidth': newWidth, 'newHeight': newHeight};
}

export {
  init,
  renderUploadResult,
};
