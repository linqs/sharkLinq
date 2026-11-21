"""Shared helpers for pagination tests."""

import http
import json
from pathlib import Path

BASELINES = json.loads(
    Path(__file__).with_name("pagination_baselines.json").read_text(encoding="utf-8")
)


def expected_ids(resource, case):
    """Return the expected ordered IDs for a filter case."""
    return BASELINES[resource][case]


def ids_of(results):
    """Return the list of IDs from API result dicts."""
    return [result["id"] for result in results]


def filter_and_sort(function, defaults, sort_attribute, overrides):
    """Run an unpaginated filter and sort results in descending order."""
    arguments = defaults.copy()
    arguments.update(overrides)
    results, status, _ = function(None, "", **arguments)
    assert status == http.HTTPStatus.OK
    assert isinstance(results, list)
    return sorted(
        results,
        key=lambda result: result.get(sort_attribute) or 0,
        reverse=True,
    )


# Keep page arguments explicit for readability across test files.
# pylint: disable=too-many-arguments
def paginate(
    function,
    defaults,
    sort_attribute,
    overrides,
    *,
    page=1,
    page_size=None,
):
    """Call a filter endpoint with pagination enabled."""
    arguments = defaults.copy()
    arguments.update(overrides)
    arguments.update(
        paginate="true",
        page=str(page),
        attribute=sort_attribute,
        order="descending",
    )
    if page_size is not None:
        arguments["page_size"] = str(page_size)

    response, status, _ = function(None, "", **arguments)
    assert status == http.HTTPStatus.OK
    assert isinstance(response, dict)
    return response


def assert_page_contract(response, expected, page, page_size):
    """Check pagination response fields and the returned page of IDs."""
    offset = (page - 1) * page_size
    expected_page = expected[offset: offset + page_size]
    assert response == {
        "results": response["results"],
        "total_count": len(expected),
        "page": page,
        "page_size": page_size,
        "has_more": offset + page_size < len(expected),
    }
    assert ids_of(response["results"]) == expected_page


def collect_paginated_ids(fetch_page):
    """Fetch all pages and check for duplicate IDs or mismatched totals."""
    collected = []
    page = 1
    total_count = None
    while True:
        response = fetch_page(page)
        if total_count is None:
            total_count = response["total_count"]
        assert response["total_count"] == total_count
        page_ids = ids_of(response["results"])
        assert set(page_ids).isdisjoint(collected)
        collected.extend(page_ids)
        if not response["has_more"]:
            break
        page += 1
        assert page <= total_count + 1

    assert len(collected) == total_count
    return collected
