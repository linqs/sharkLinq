import {sendRequest} from '/js/modules/shark_matcher/core.js';
import * as Core from '/static/js/modules/webui/core.js';
import * as Util from '/static/js/modules/webui/util.js';

function active() {
  return Core.getCurrentUserMode() === 'demo';
}

function renderHome(container) {
  if (!active()) return;
  const section = document.createElement('div');
  section.className = 'settings-section demo-section';
  section.innerHTML = `
    <button class="pill-btn demo-toggle" aria-expanded="false">Demo</button>
    <div class="demo-controls" style="display: none">
      <p class="demo-status" role="status">Loading the shared round…</p>
      <p>Randomizing creates a new randomized set, existing randomized sets are cleared.</p>
      <div class="input-item">
        <label for="demo-count">Images:</label>
        <input id="demo-count" type="number" min="1" step="1" value="10">
      </div>
      <button class="pill-btn demo-randomize" disabled>Randomize</button>
      <button class="pill-btn demo-clear" disabled>Clear Randomized Images</button>
      <button class="pill-btn demo-browse" hidden>Search Randomized Images</button>
    </div>`;
  container.appendChild(section);
  const toggle = section.querySelector('.demo-toggle');
  toggle.addEventListener('click', () => {
    Util.toggleVisibility('', false, false, {container: section, selector: '.demo-controls'});
    toggle.setAttribute('aria-expanded', section.querySelector('.demo-controls').style.display !== 'none');
  });
  const status = section.querySelector('.demo-status');
  const input = section.querySelector('#demo-count');
  const button = section.querySelector('.demo-randomize');
  const clearButton = section.querySelector('.demo-clear');
  clearButton.title = 'Restore the original demo dataset and discard all round edits.';
  const browseButton = section.querySelector('.demo-browse');
  browseButton.addEventListener('click', () => {
    window.location.hash = Core.formHashPath('search', {keywords: 'randomized'});
  });
  sendRequest({endpoint: 'demo/status'}).then((result) => {
    status.textContent = result.selected ?
      `${result.selected} currently randomized images.` :
      'No randomized set yet. Select number of images to Randomize.';
    input.max = result.maximum;
    button.disabled = !Core.canManageDemo() || result.maximum === 0;
    clearButton.disabled = !Core.canManageDemo() || result.selected === 0;
    input.disabled = !Core.canManageDemo() || result.maximum === 0;
    if (result.maximum === 0) status.textContent = 'No images can be detached while leaving a matching image behind.';
    browseButton.hidden = result.selected === 0;
  }).catch((error) => {
    status.textContent = error.message;
  });
  for (const actionButton of [button, clearButton]) {
    actionButton.addEventListener('click', async () => {
      const randomizing = actionButton === button;
      if (randomizing && !input.reportValidity()) return;
      const disabledStates = [button.disabled, clearButton.disabled];
      button.disabled = true;
      clearButton.disabled = true;
      try {
        await sendRequest({
          endpoint: randomizing ? 'demo/randomize' : 'demo/clear',
          payload: randomizing ? {count: input.valueAsNumber} : {},
        });
        // A full reload acquires the new round and its separate cache namespace.
        window.location.reload();
      } catch (error) {
        status.textContent = error.message;
        [button.disabled, clearButton.disabled] = disabledStates;
      }
    });
  }
}

async function renderKeys(container, individual) {
  if (!active()) return;
  try {
    const payload = {source_id: individual.source};
    const result = await sendRequest({endpoint: 'demo/keys', payload});
    if (!result.count || !container.isConnected) return;
    const row = document.createElement('div');
    row.className = 'info-row demo-match-key';
    const label = document.createElement('span');
    label.className = 'label';
    label.textContent = 'Match Key:';
    const button = document.createElement('button');
    button.className = 'pill-btn';
    button.textContent = 'Click to reveal';
    button.setAttribute('aria-expanded', 'false');
    row.append(label, button);
    container.appendChild(row);
    button.addEventListener('click', async () => {
      const visibleKey = row.querySelector('.value');
      if (visibleKey) {
        visibleKey.remove();
        button.textContent = 'Click to reveal';
        button.setAttribute('aria-expanded', 'false');
        return;
      }
      button.disabled = true;
      try {
        const revealed = await sendRequest({endpoint: 'demo/keys', payload: {...payload, reveal: true}});
        const values = document.createElement('span');
        values.className = 'value';
        values.textContent = revealed.keys.map((key) =>
          `${key.image_name} -> Shark ${key.original_name}`).join('; ');
        row.insertBefore(values, button);
        button.textContent = 'Hide';
        button.setAttribute('aria-expanded', 'true');
      } catch (error) {
        alert(error.message);
      } finally {
        button.disabled = false;
      }
    });
  } catch (error) {
    if (container.isConnected) {
      const message = document.createElement('p');
      message.textContent = error.message;
      container.appendChild(message);
    }
  }
}

function showMergeResult(result) {
  const grades = result.demo_results || [];
  if (!grades.length) return;
  const correct = grades.filter((grade) => grade.correct).length;
  if (grades.length === 1) {
    alert(correct ? 'Correct match!' : 'Incorrect match.');
  } else {
    alert(`${correct} correct, ${grades.length - correct} incorrect. Merge committed.`);
  }
}

export {renderHome, renderKeys, showMergeResult};
