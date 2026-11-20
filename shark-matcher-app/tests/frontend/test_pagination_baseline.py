"""Proposal and action pagination contracts for the immutable CI fixture."""

import pytest

import frontend.action
import frontend.proposal
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

PROPOSAL_DEFAULTS = {
    "ids": "",
    "keywords": "",
    "original": "",
    "final": "",
    "author": "",
    "min_time": "",
    "max_time": "",
    "action": "",
    "object": "",
    "status": "",
}
ACTION_DEFAULTS = {
    "ids": "",
    "keywords": "",
    "original": "",
    "final": "",
    "author": "",
    "min_time": "",
    "max_time": "",
    "action": "",
    "object": "",
}

PROPOSAL_CASES = (
    ("no_filter", {}),
    ("status_open", {"status": "open"}),
    ("status_archive", {"status": "archive"}),
    ("action_upload", {"action": "upload"}),
    ("action_merge", {"action": "merge"}),
    ("action_update", {"action": "update"}),
    ("action_delete", {"action": "delete"}),
    ("object_individual", {"object": "individual"}),
    ("object_media", {"object": "media"}),
    ("author_0", {"author": "0"}),
    ("keyword_merged_boy", {"keywords": "merged boy"}),
)
ACTION_CASES = (
    ("no_filter", {}),
    ("action_create", {"action": "create"}),
    ("action_merge", {"action": "merge"}),
    ("action_update", {"action": "update"}),
    ("action_delete", {"action": "delete"}),
    ("action_unmatch", {"action": "unmatch"}),
    ("object_individual", {"object": "individual"}),
    ("object_media", {"object": "media"}),
    ("object_tag", {"object": "tag"}),
    ("author_0", {"author": "0"}),
    ("keyword_acoustic", {"keywords": "acoustic"}),
)


@pytest.mark.parametrize(("case", "filters"), PROPOSAL_CASES)
def test_proposal_filter_and_first_page_match_ci_baseline(case, filters):
    """Proposal filtering and server page one share one ordered baseline."""
    expected = expected_ids("proposals", case)
    unpaginated = filter_and_sort(
        frontend.proposal.filter_proposals,
        PROPOSAL_DEFAULTS,
        "time",
        filters,
    )
    assert ids_of(unpaginated) == expected

    response = paginate(
        frontend.proposal.filter_proposals,
        PROPOSAL_DEFAULTS,
        "time",
        filters,
        page_size=TEST_PAGE_SIZE,
    )
    assert_page_contract(response, expected, 1, TEST_PAGE_SIZE)


@pytest.mark.parametrize(("case", "filters"), ACTION_CASES)
def test_action_filter_and_first_page_match_ci_baseline(case, filters):
    """Action filtering and server page one share one ordered baseline."""
    expected = expected_ids("actions", case)
    unpaginated = filter_and_sort(
        frontend.action.filter_actions,
        ACTION_DEFAULTS,
        "time",
        filters,
    )
    assert ids_of(unpaginated) == expected

    response = paginate(
        frontend.action.filter_actions,
        ACTION_DEFAULTS,
        "time",
        filters,
        page_size=TEST_PAGE_SIZE,
    )
    assert_page_contract(response, expected, 1, TEST_PAGE_SIZE)


@pytest.mark.parametrize(
    ("resource", "function", "defaults"),
    (
        ("proposals", frontend.proposal.filter_proposals, PROPOSAL_DEFAULTS),
        ("actions", frontend.action.filter_actions, ACTION_DEFAULTS),
    ),
)
def test_all_small_pages_cover_each_ci_baseline_once(resource, function, defaults):
    """Small pages cover every immutable-fixture row without overlap."""
    expected = expected_ids(resource, "no_filter")

    def fetch_page(page):
        return paginate(
            function,
            defaults,
            "time",
            {},
            page=page,
            page_size=TEST_PAGE_SIZE,
        )

    assert collect_paginated_ids(fetch_page) == expected


@pytest.mark.parametrize(
    ("resource", "function", "defaults"),
    (
        ("proposals", frontend.proposal.filter_proposals, PROPOSAL_DEFAULTS),
        ("actions", frontend.action.filter_actions, ACTION_DEFAULTS),
    ),
)
def test_default_page_size_remains_in_service(resource, function, defaults):
    """Omitting page size preserves the application pagination default."""
    expected = expected_ids(resource, "no_filter")
    response = paginate(function, defaults, "time", {})
    assert_page_contract(response, expected, 1, DEFAULT_PAGE_SIZE)


@pytest.mark.parametrize(
    ("resource", "function", "defaults"),
    (
        ("proposals", frontend.proposal.filter_proposals, PROPOSAL_DEFAULTS),
        ("actions", frontend.action.filter_actions, ACTION_DEFAULTS),
    ),
)
def test_page_beyond_ci_baseline_is_empty(resource, function, defaults):
    """A page beyond the immutable baseline is empty and terminal."""
    expected = expected_ids(resource, "no_filter")
    response = paginate(
        function,
        defaults,
        "time",
        {},
        page=100,
        page_size=TEST_PAGE_SIZE,
    )
    assert_page_contract(response, expected, 100, TEST_PAGE_SIZE)
