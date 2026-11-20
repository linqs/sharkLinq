// Original author Eriq Augustine (edulinq).

import * as History from './history.js';
import * as Home from './home.js';
import * as Log from './log.js';
import * as Login from './login.js';
import * as Match from './match.js';
import * as Merge from './merge.js';
import * as Proposal from './proposal.js';
import * as Review from './review.js';
import * as Search from './search.js';
import * as Settings from './settings.js';
import * as Individual from './individual.js';
import * as Routes from './routes.js';
import * as Upload from './upload.js';
import * as Util from './util.js';

function init() {
  initHandlers();

  History.init();
  Home.init();
  Log.init();
  Login.init();
  Match.init();
  Merge.init();
  Proposal.init();
  Review.init();
  Search.init();
  Settings.init();
  Individual.init();
  Upload.init();
  Util.init();

  Routes.init();

  // Activate home event listeners.
  Home.keydownFunctionality();
}

function initHandlers() {
  window.sharkMatcher = window.sharkMatcher || {};
  window.sharkMatcher.handlers = window.sharkMatcher.handlers || {};

  window.sharkMatcher.handlers.login = Login.login;
}

export {
  init,
};
