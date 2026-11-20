// Endpoint functions for review objects.

import * as Core from './core.js';

function filter_reviews(minLength, maxLength, minTime, maxTime,
    minAssociated, maxAssociated, sex, locations, qualities, maxQualities,
    alternateNames, labels, tags, allowUnknown, keywords, model, reviewType,
    page, pageSize) {
  return Core.sendRequest({
    endpoint: 'review/get',
    payload: {
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
      'keywords': keywords,
      'model': model,
      'review_type': reviewType,
      'page': page,
      'page_size': pageSize,
    },
  });
}

export {
  filter_reviews,
};
