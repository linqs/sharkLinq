// Endpoint functions for action objects.

import * as Core from './core.js';

function filter_actions(
    minTime, maxTime, object, action, original, final, author, ids, keywords,
    paginate, page, pageSize, attribute, order) {
  return Core.sendRequest({
    endpoint: 'action/filter',
    payload: {
      'min_time': minTime,
      'max_time': maxTime,
      'object': object,
      'action': action,
      'original': original,
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

export {
  filter_actions,
};
