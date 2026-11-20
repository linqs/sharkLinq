// Endpoint functions for individual and associated objects.

import * as Core from './core.js';

function get_individuals(ids, names, keywords, getFull, getSource) {
  return Core.sendRequest({
    endpoint: 'individual/get',
    payload: {
      'ids': ids,
      'names': names,
      'keywords': keywords,
      'get_full': getFull,
      'get_source': getSource,
    },
  });
}

function filter_individuals(minLength, maxLength, minTime, maxTime,
    minAssociated, maxAssociated, sex, locations, qualities, maxQualities,
    ids, names, medias, alternateNames, labels, tags, allowUnknown, keywords,
    paginate, page, pageSize, attribute, order) {
  return Core.sendRequest({
    endpoint: 'individual/filter',
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
      'ids': ids,
      'names': names,
      'medias': medias,
      'alternate_names': alternateNames,
      'labels': labels,
      'tags': tags,
      'allow_unknown': allowUnknown,
      'keywords': keywords,
      'paginate': paginate,
      'page': page,
      'page_size': pageSize,
      'attribute': attribute,
      'order': order,
    },
  });
}

function post_individual(id, name, length, sex, source, reference, media) {
  return Core.sendRequest({
    endpoint: 'individual/post',
    payload: {
      'id': id,
      'name': name,
      'length': length,
      'sex': sex,
      'source': source,
      'reference': reference,
      'media': media,
    },
  });
}

function merge_individuals(majorId, minorId, preview) {
  return Core.sendRequest({
    endpoint: 'individual/merge',
    payload: {
      'major_id': majorId,
      'minor_id': minorId,
      'preview': preview,
    },
  });
}

function unmatch_individual(id) {
  return Core.sendRequest({
    endpoint: 'individual/unmatch',
    payload: {
      'id': id,
    },
  });
}

function update_reference(id) {
  return Core.sendRequest({
    endpoint: 'individual/reference',
    payload: {
      'id': id,
    },
  });
}

function post_object(id, individual, name, type, time, table) {
  return Core.sendRequest({
    endpoint: 'object/post',
    payload: {
      'id': id,
      'individual': individual,
      'name': name,
      'type': type,
      'time': time,
      'table': table,
    },
  });
}

function delete_object(id, table) {
  return Core.sendRequest({
    endpoint: 'object/delete',
    payload: {
      'id': id,
      'table': table,
    },
  });
}

export {
  get_individuals,
  filter_individuals,
  post_individual,
  merge_individuals,
  unmatch_individual,
  update_reference,

  post_object,
  delete_object,
};
