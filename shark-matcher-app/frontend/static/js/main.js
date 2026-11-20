// Original author Eriq Augustine (edulinq).

import * as WebUI from './modules/webui/base.js';
import * as API from './modules/shark_matcher/core.js';

async function main() {
  window.sharkMatcher = window.sharkMatcher || {};
  window.sharkMatcher.config = {
    displayName: 'Shark',
  };

  try {
    const response = await fetch('/api/runtime/config', {cache: 'no-store'});
    if (!response.ok) throw new Error('Unable to load server configuration.');
    const result = await response.json();
    Object.assign(window.sharkMatcher.config, result.content);
    if (result.content.demo_enabled) {
      window.sharkMatcher.demo = await import('/demo/demo.js');
    } else if (localStorage.getItem('userMode') === 'demo') {
      localStorage.setItem('userMode', 'viewing');
    }
    // Pin mode to this document: changing another tab cannot redirect a pending edit.
    window.sharkMatcher.config.userMode = localStorage.getItem('userMode');
  } catch (error) {
    document.body.textContent = 'Unable to load the application. Reload to try again.';
    console.error(error);
    return;
  }

  // Revalidate saved sessions and establish image access before rendering.
  if (API.hasCredentials()) {
    try {
      await API.sendRequest({endpoint: 'user/auth'});
    } catch {
      API.clearAllLocal();
    }
  }
  WebUI.init();
}

document.addEventListener('DOMContentLoaded', main);
