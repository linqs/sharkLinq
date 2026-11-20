"""Individual access functions backed by individuals.csv."""

import src.csv_store._loader as _loader
import src.csv_store.media as _media_module
from src.csv_store.models import Individual


def is_media_individual(individual: Individual) -> bool:
    """Return True if this individual has an associated media record and a separate source ID."""
    return (individual.source != individual.id) and (individual.media is not None)


def is_source_individual(individual: Individual) -> bool:
    """Return True if this individual is a source (grouping) individual."""
    if individual.id == -1:
        return True
    return (individual.source == individual.id and
            individual.media is None and
            individual.reference is not None)


def get_individuals(ids=None, get_full=False, get_source=False,
                    source_only=True, **_kwargs) -> list:
    """Return filtered Individual objects loaded from individuals.csv."""
    all_individuals = _loader.load_individuals()
    by_id = {ind.id: ind for ind in all_individuals}

    # --- Step 1: determine the ID set to fetch ---
    if ids is None:
        if source_only:
            ids_to_fetch = [
                ind.id for ind in all_individuals
                if ind.id >= 0 and is_source_individual(ind)
            ]
        else:
            ids_to_fetch = [ind.id for ind in all_individuals if ind.id >= 0]
    else:
        ids_to_fetch = [i for i in ids if i in by_id]

    # --- Step 2: traverse to source individuals if requested ---
    if get_source and ids_to_fetch:
        source_ids = set()
        for ind_id in ids_to_fetch:
            ind = by_id.get(ind_id)
            if ind is not None and ind.source is not None:
                source_ids.add(ind.source)
        ids_to_fetch = list(source_ids)

    if not ids_to_fetch:
        return []

    # --- Step 3: build a fresh result list ---
    result = []
    for ind_id in ids_to_fetch:
        ind = by_id.get(ind_id)
        if ind is None:
            continue
        result.append(_fresh(ind))

    if not get_full:
        return result

    # --- Step 4: populate associated data ---
    # Group all individuals by their source id.
    source_map: dict = {}
    for ind in all_individuals:
        if ind.id < 0 or ind.source is None:
            continue
        source_map.setdefault(ind.source, []).append(ind)

    # Collect all media IDs we will need.
    needed_media_ids = set()
    for fresh_ind in result:
        for assoc in source_map.get(fresh_ind.id, []):
            if assoc.media is not None:
                needed_media_ids.add(assoc.media)

    # Fetch those media in one pass.
    media_objects = _media_module.get_medias(ids=list(needed_media_ids))
    media_by_id = {m.id: m for m in media_objects}

    # Attach associated data to each result individual.
    for fresh_ind in result:
        group = source_map.get(fresh_ind.id, [])

        # Sort: source individual (id == source) first.
        group_sorted = sorted(group, key=lambda s: s.id != s.source)

        for assoc in group_sorted:
            fresh_ind.associated_individuals.append(_fresh(assoc))
            if assoc.media is not None and assoc.media in media_by_id:
                fresh_ind.associated_medias.append(media_by_id[assoc.media])

    return result


def get_matching_individual_ids(ids=None, min_length=None, max_length=None,
                                 sex=None, allow_unknown=False,
                                 source_only=True, **_kwargs) -> list:
    """Return individual IDs matching all supplied filter criteria."""
    all_individuals = _loader.load_individuals()

    id_set = set(ids) if ids is not None else None
    sex_set = {s.lower() for s in sex} if sex is not None else None
    min_len = float(min_length) if min_length not in (None, '') else None
    max_len = float(max_length) if max_length not in (None, '') else None

    result = []
    for ind in all_individuals:
        if ind.id < 0:
            continue
        if source_only and not is_source_individual(ind):
            continue
        if id_set is not None and ind.id not in id_set:
            continue

        # Sex filter.
        if sex_set is not None:
            if ind.sex is None:
                if not allow_unknown:
                    continue
            elif ind.sex.lower() not in sex_set:
                continue

        # Length filters.
        if min_len is not None:
            if ind.length is None:
                if not allow_unknown:
                    continue
            elif ind.length < min_len:
                continue

        if max_len is not None:
            if ind.length is None:
                if not allow_unknown:
                    continue
            elif ind.length > max_len:
                continue

        result.append(ind.id)

    return result


def _fresh(ind: Individual) -> Individual:
    """Return a new Individual with empty associated lists (safe to mutate)."""
    return Individual(
        id=ind.id, name=ind.name, sex=ind.sex, length=ind.length,
        media=ind.media, source=ind.source, reference=ind.reference,
    )
