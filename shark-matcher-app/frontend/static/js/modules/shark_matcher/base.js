// Browser API entry point re-exporting endpoint modules and app helpers.

import * as Action from './action.js';
import * as Core from './core.js';
import * as Match from './match.js';
import * as Media from './media.js';
import * as Individual from './individual.js';
import * as User from './user.js';
import * as Proposal from './proposal.js';
import * as Review from './review.js';

const getCached = Core.getCached;
const getCachedValue = Core.getCachedValue;
const clearCache = Core.clearCache;
const clearAllLocal = Core.clearAllLocal;

const getCredentials = Core.getCredentials;
const hasCredentials = Core.hasCredentials;
const setCredentials = Core.setCredentials;

const getFilters = Core.getFilters;
const setFilters = Core.setFilters;
const clearFilters = Core.clearFilters;

const getTheme = Core.getTheme;
const setTheme = Core.setTheme;

// Endpoint functions for miscellaneous tasks.

function export_data(content, format, filters) {
  const payload = {
    'content': content,
    'format': format,
  };
  if (filters) {
    Object.assign(payload, filters);
  }
  return Core.sendRequest({
    endpoint: 'data/export',
    payload: payload,
  });
}

function get_dataset_stats() {
  return Core.sendRequest({
    endpoint: 'data/stats',
    payload: {},
  });
}

function get_embeddings_status() {
  return Core.sendRequest({
    endpoint: 'embeddings/status',
    payload: {},
  });
}

function update_embeddings() {
  return Core.sendRequest({
    endpoint: 'embeddings/update',
    payload: {},
  });
}

export {
  Action,
  Match,
  Media,
  Individual,
  User,
  Proposal,
  Review,

  export_data,
  get_dataset_stats,
  get_embeddings_status,
  update_embeddings,

  getCached,
  getCachedValue,
  clearCache,
  clearAllLocal,

  getCredentials,
  hasCredentials,
  setCredentials,

  getFilters,
  setFilters,
  clearFilters,

  getTheme,
  setTheme,
};
