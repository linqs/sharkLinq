// Endpoint functions for media objects.

import * as Core from './core.js';

function get_medias(ids, getFull) {
  return Core.sendRequest({
    endpoint: 'media/get',
    payload: {
      'ids': ids,
      'get_full': getFull,
    },
  });
}

function post_media(id, hash, name, source, individual, metadata, sex,
    length, location, time, type, quality,
    alternateNames = null, tags = null, labels = null) {
  return Core.sendRequest({
    endpoint: 'media/post',
    payload: {
      'id': id,
      'hash': hash,
      'name': name,
      'source': source,
      'individual': individual,
      'metadata': metadata,
      'sex': sex,
      'length': length,
      'location': location,
      'time': time,
      'type': type,
      'quality': quality,
      'alternate_names': alternateNames,
      'tags': tags,
      'labels': labels,
    },
  });
}

function delete_media(id) {
  return Core.sendRequest({
    endpoint: 'media/delete',
    payload: {
      'id': id,
    },
  });
}

function upload_batch(items) {
  return Core.sendRequest({
    endpoint: 'media/upload_batch',
    payload: {
      'items': items,
    },
  });
}

export {
  get_medias,
  post_media,
  delete_media,
  upload_batch,
};
