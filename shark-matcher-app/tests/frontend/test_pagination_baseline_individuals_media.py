"""Individual and media contracts for the immutable committed CI fixture."""

import pytest

import frontend.individual
import frontend.media
from tests.frontend.fixtures.pagination_contracts import (
    assert_page_contract,
    collect_paginated_ids,
    expected_ids,
    filter_and_sort,
    ids_of,
    paginate,
)

pytestmark = pytest.mark.usefixtures("reset_ci_test_database")

TEST_PAGE_SIZE = 2
DEFAULT_PAGE_SIZE = 50

INDIVIDUAL_DEFAULTS = {
    "ids": "",
    "keywords": "",
    "names": "",
    "alternate_names": "",
    "tags": "",
    "labels": "",
    "medias": "",
    "min_length": "",
    "max_length": "",
    "sex": "",
    "locations": "",
    "qualities": "",
    "max_qualities": "",
    "min_time": "",
    "max_time": "",
    "min_associated": "",
    "max_associated": "",
    "allow_unknown": "",
}
MEDIA_DEFAULTS = {
    "ids": "",
    "keywords": "",
    "names": "",
    "sources": "",
    "locations": "",
    "qualities": "",
    "types": "",
    "min_time": "",
    "max_time": "",
}

INDIVIDUAL_CASES = (
    ("no_filter", {}),
    ("sex_M", {"sex": "M"}),
    ("sex_F", {"sex": "F"}),
    ("length_10_12", {"min_length": "10", "max_length": "12"}),
    ("label_fresh", {"labels": "fresh"}),
    ("label_any", {"labels": "*"}),
    ("tag_test-tag", {"tags": "test-tag"}),
    ("alternate_name_merged_boy", {"alternate_names": "merged boy"}),
    ("alternate_name_any", {"alternate_names": "*"}),
    ("location_AN", {"locations": "AN"}),
    ("location_FAR", {"locations": "FAR"}),
    ("quality_good", {"qualities": "good"}),
    (
        "time_2011_2012",
        {"min_time": "1318662000", "max_time": "1350284400"},
    ),
    ("min_associated_3", {"min_associated": "3"}),
    ("sex_F_label_fresh", {"sex": "F", "labels": "fresh"}),
    ("keyword_merged_boy", {"keywords": "merged boy"}),
)
MEDIA_CASES = (
    ("no_filter", {}),
    ("location_AN", {"locations": "AN"}),
    ("location_FAR", {"locations": "FAR"}),
    ("location_NB", {"locations": "NB"}),
    ("type_underwater", {"types": "underwater"}),
    ("type_topshot", {"types": "topshot"}),
    ("quality_good", {"qualities": "good"}),
    ("quality_medium", {"qualities": "medium"}),
    ("quality_poor", {"qualities": "poor"}),
    (
        "time_2011_2012",
        {"min_time": "1318662000", "max_time": "1350284400"},
    ),
    ("location_AN_topshot", {"locations": "AN", "types": "topshot"}),
    ("keyword_missing", {"keywords": "not present in CI fixture"}),
)


@pytest.mark.parametrize(("case", "filters"), INDIVIDUAL_CASES)
def test_individual_filter_and_first_page_match_ci_baseline(case, filters):
    """Individual filtering and server page one share one ordered baseline."""
    expected = expected_ids("individuals", case)
    unpaginated = filter_and_sort(
        frontend.individual.filter_individuals,
        INDIVIDUAL_DEFAULTS,
        "id",
        filters,
    )
    assert ids_of(unpaginated) == expected

    response = paginate(
        frontend.individual.filter_individuals,
        INDIVIDUAL_DEFAULTS,
        "id",
        filters,
        page_size=TEST_PAGE_SIZE,
    )
    assert_page_contract(response, expected, 1, TEST_PAGE_SIZE)


@pytest.mark.parametrize(("case", "filters"), MEDIA_CASES)
def test_media_filter_matches_ci_baseline(case, filters):
    """Media filtering preserves its exact browser-visible ordered result."""
    expected = expected_ids("media", case)
    results = filter_and_sort(
        frontend.media.filter_medias,
        MEDIA_DEFAULTS,
        "time",
        filters,
    )
    assert ids_of(results) == expected


def test_all_small_individual_pages_cover_ci_baseline_once():
    """Small pages cover every immutable-fixture individual without overlap."""
    expected = expected_ids("individuals", "no_filter")

    def fetch_page(page):
        return paginate(
            frontend.individual.filter_individuals,
            INDIVIDUAL_DEFAULTS,
            "id",
            {},
            page=page,
            page_size=TEST_PAGE_SIZE,
        )

    assert collect_paginated_ids(fetch_page) == expected


def test_individual_default_page_size_remains_in_service():
    """Omitting page size preserves the application pagination default."""
    expected = expected_ids("individuals", "no_filter")
    response = paginate(
        frontend.individual.filter_individuals,
        INDIVIDUAL_DEFAULTS,
        "id",
        {},
    )
    assert_page_contract(response, expected, 1, DEFAULT_PAGE_SIZE)


def test_individual_page_beyond_ci_baseline_is_empty():
    """A page beyond the immutable baseline is empty and terminal."""
    expected = expected_ids("individuals", "no_filter")
    response = paginate(
        frontend.individual.filter_individuals,
        INDIVIDUAL_DEFAULTS,
        "id",
        {},
        page=100,
        page_size=TEST_PAGE_SIZE,
    )
    assert_page_contract(response, expected, 100, TEST_PAGE_SIZE)


@pytest.mark.skip(reason="Server-side media pagination is deferred to V2 large-query work.")
def test_media_server_pagination_contract_is_deferred():
    """Track the intentional media-pagination boundary explicitly."""
