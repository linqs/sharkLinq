// Endpoint functions for proposal objects.

import * as Core from './core.js';

function filter_proposals(minTime, maxTime, object, action, original, final,
    status, author, ids, keywords,
    paginate, page, pageSize, attribute, order) {
  return Core.sendRequest({
    endpoint: 'proposal/filter',
    payload: {
      'min_time': minTime,
      'max_time': maxTime,
      'object': object,
      'action': action,
      'original': original,
      'status': status,
      'final': final,
      'author': author,
      'ids': ids,
      'keywords': keywords,
      'paginate': paginate,
      'page': page,
      'page_size': pageSize,
      'attribute': attribute,
      'order': order,
    },
  });
}

function post_proposal(id, original, final, action, object, status) {
  return Core.sendRequest({
    endpoint: 'proposal/post',
    payload: {
      'id': id,
      'original': original,
      'final': final,
      'action': action,
      'object': object,
      'status': status,
    },
  });
}

function delete_proposal(id) {
  return Core.sendRequest({
    endpoint: 'proposal/delete',
    payload: {
      'id': id,
    },
  });
}

export {
  filter_proposals,
  post_proposal,
  delete_proposal,
};
