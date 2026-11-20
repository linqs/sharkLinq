// Endpoint functions for matching individual objects.

import * as Core from './core.js';

function filter_matches(name, method = 'nearest neighbor', model = 'ViT',
    minLength, maxLength, minTime, maxTime,
    minAssociated, maxAssociated, sex, locations, qualities, maxQualities,
    ids, medias, alternateNames, labels, tags, allowUnknown, keywords) {
  return Core.sendRequest({
    endpoint: 'match/filter',
    payload: {
      'name': name,
      'method': method,
      'model': model,
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
      'ids': ids,
      'medias': medias,
      'alternate_names': alternateNames,
      'labels': labels,
      'tags': tags,
      'allow_unknown': allowUnknown,
      'keywords': keywords,
    },
  });
}

export {
  filter_matches,
};
