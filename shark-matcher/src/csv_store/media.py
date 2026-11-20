"""Media access functions backed by media.csv and individuals.csv."""

from src.core import constants
from src.csv_store import _loader
from src.csv_store.models import Individual, Media


def get_medias(ids=None, get_full=False, **_kwargs) -> list:
    """Return Media objects, optionally filtered by ID and joined with individuals."""
    all_media = _loader.load_media()

    if ids is None:
        result = [_copy_media(m) for m in all_media]
    else:
        id_set = set(ids)
        by_id = {m.id: m for m in all_media}
        # Preserve the requested order.
        result = [_copy_media(by_id[i]) for i in ids if i in by_id]

    if get_full and result:
        _attach_individuals(result)

    return result


def load_medias_with_individuals(csv_dir=None, ids=None):
    """Load Media records with their linked Individual attribute populated."""
    if csv_dir is not None:
        _loader.set_db_dir(csv_dir)
    return get_medias(ids=ids, get_full=True)


def get_matching_media_ids(ids=None, locations=None, types=None,
                            qualities=None, max_qualities=None,
                            min_time=None, max_time=None,
                            allow_unknown=False, **_kwargs) -> list:
    """Return media IDs matching all supplied filter criteria."""
    all_media = _loader.load_media()

    id_set = set(ids) if ids is not None else None
    loc_set = {l.lower() for l in locations} if locations is not None else None
    type_set = {t.lower() for t in types} if types is not None else None

    min_q_set = _expand_min_qualities(qualities) if qualities is not None else None
    max_q_set = _expand_max_qualities(max_qualities) if max_qualities is not None else None
    if allow_unknown and min_q_set is not None:
        min_q_set.add(None)
    if allow_unknown and max_q_set is not None:
        max_q_set.add(None)

    min_time_f = float(min_time) if min_time not in (None, '') else None
    max_time_f = float(max_time) if max_time not in (None, '') else None

    result = []
    for m in all_media:
        if id_set is not None and m.id not in id_set:
            continue
        if not _match_field(m.location, loc_set, allow_unknown):
            continue
        if not _match_field(m.type, type_set, allow_unknown):
            continue
        if not _match_quality(m.quality, min_q_set, max_q_set, allow_unknown):
            continue
        if min_time_f is not None:
            if m.time is None:
                if not allow_unknown:
                    continue
            elif float(m.time) < min_time_f:
                continue
        if max_time_f is not None:
            if m.time is None:
                if not allow_unknown:
                    continue
            elif float(m.time) > max_time_f:
                continue
        result.append(m.id)

    return result


def get_individual_ids_by_media_ids(ids, dict_format=False, **_kwargs):
    """Return the individual ID(s) associated with each media ID as a list or dictionary."""
    id_set = set(ids)
    all_individuals = _loader.load_individuals()

    media_to_ind = {}
    for ind in all_individuals:
        if ind.media is not None and ind.media in id_set:
            media_to_ind[ind.media] = ind.id

    if dict_format:
        return media_to_ind

    return list(set(media_to_ind.values()))


def _copy_media(m: Media) -> Media:
    """Return a fresh Media copy (individual=None) so callers can mutate it."""
    return Media(id=m.id, hash=m.hash, name=m.name, filename=m.filename,
                 source=m.source, metadata=m.metadata, location=m.location,
                 time=m.time, type=m.type, quality=m.quality, individual=None)


def _attach_individuals(media_list: list):
    """Populate the ``individual`` attribute on each Media object in-place."""
    all_individuals = _loader.load_individuals()

    # Build media_id -> individual mapping.
    media_to_ind = {}
    for ind in all_individuals:
        if ind.media is not None:
            media_to_ind[ind.media] = ind

    for m in media_list:
        m.individual = media_to_ind.get(m.id)


def _expand_min_qualities(qualities: list) -> set:
    """Expand a min-quality list upward (e.g. 'poor' -> {'poor','medium','good'})."""
    expanded = set(q.lower() for q in qualities)
    order = constants.QUALITY_ORDER
    for threshold in list(expanded):
        if threshold in order:
            idx = order.index(threshold)
            expanded.update(order[idx:])
    return expanded


def _expand_max_qualities(max_qualities: list) -> set:
    """Expand a max-quality list downward (e.g. 'medium' -> {'unmatchable','poor','medium'})."""
    expanded = set(q.lower() for q in max_qualities)
    order = constants.QUALITY_ORDER
    for threshold in list(expanded):
        if threshold in order:
            idx = order.index(threshold)
            expanded.update(order[:idx + 1])
    return expanded


def _match_field(value, allowed_set, allow_unknown: bool) -> bool:
    """Return True if value is in allowed_set (or allowed_set is None)."""
    if allowed_set is None:
        return True
    if value is None:
        return allow_unknown
    return value.lower() in allowed_set


def _match_quality(quality, min_q_set, max_q_set, allow_unknown: bool) -> bool:
    """Return True if quality satisfies both min and max quality filters."""
    if min_q_set is None and max_q_set is None:
        return True
    q = quality.lower() if quality is not None else None
    if min_q_set is not None:
        if q is None:
            if not allow_unknown:
                return False
        elif q not in min_q_set:
            return False
    if max_q_set is not None:
        if q is None:
            if not allow_unknown:
                return False
        elif q not in max_q_set:
            return False
    return True
