-- Migration: add performance indexes to an existing database.
-- Safe to run multiple times; IF NOT EXISTS prevents errors on re-runs.

-- Individuals: core relationship columns hit by every bulk fetch and filter query.
CREATE INDEX IF NOT EXISTS idx_individuals_source    ON Individuals(source);
CREATE INDEX IF NOT EXISTS idx_individuals_media     ON Individuals(media);
CREATE INDEX IF NOT EXISTS idx_individuals_reference ON Individuals(reference);
CREATE INDEX IF NOT EXISTS idx_individuals_name      ON Individuals(name COLLATE NOCASE);

-- Media: columns used in quality, location, and time-range filter queries.
CREATE INDEX IF NOT EXISTS idx_media_quality   ON Media(quality COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_media_location  ON Media(location COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_media_time      ON Media(time);
CREATE INDEX IF NOT EXISTS idx_media_name      ON Media(name COLLATE NOCASE);

-- Associated-object tables: individual FK column is queried on every get_individuals call.
-- Name column is queried on every keyword search.
CREATE INDEX IF NOT EXISTS idx_alternatenames_individual ON AlternateNames(individual);
CREATE INDEX IF NOT EXISTS idx_alternatenames_name       ON AlternateNames(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_tags_individual           ON Tags(individual);
CREATE INDEX IF NOT EXISTS idx_tags_name                 ON Tags(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_labels_individual         ON Labels(individual);
CREATE INDEX IF NOT EXISTS idx_labels_name               ON Labels(name COLLATE NOCASE);
