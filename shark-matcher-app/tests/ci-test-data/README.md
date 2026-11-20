# Immutable CI Test Data

This directory contains the committed integration fixture used by the
`shark-matcher-app` test suite and continuous integration jobs.

## Policy

The CSV and TSV data in this directory is immutable.

- Do not edit, regenerate, replace, add, or remove fixture data.
- Tests rely on its exact filenames, identifiers, values, relationships, and
  test-only credentials.
- Add new edge cases with temporary test inputs or mocks instead.

## Usage

- The database test setup reads these files to populate a disposable test database.
- Test and application code must never write into this directory.

## Enforcement

- The packaging contract test in `tests/packaging/` enforces the fixture's aggregate content digest.
- Any intentional exception needs explicit owner review before changing either
  the fixture or its pinned digest.
