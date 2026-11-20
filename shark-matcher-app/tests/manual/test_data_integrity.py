"""Manual data integrity tests for the Shark Matcher database."""

import os
import sys
import unittest
from typing import Any, Dict, List

from core import app_constants
import sqlite3

from backend.db.individual import get_associated_objects, get_individuals
from backend.db.media import get_medias
from backend.db.user import get_tokens, get_users


# Helper for direct DB access for faster "exists" checks.
def get_db_connection():
    db_path = os.environ.get("SHARK_INTEGRITY_DB_PATH", app_constants.DB_PATH)
    return sqlite3.connect(db_path)


class TestDataIntegrity(unittest.TestCase):
    """Data Integrity Tests for Shark Matcher Database."""

    @classmethod
    def setUpClass(cls):
        db_path = os.environ.get("SHARK_INTEGRITY_DB_PATH", app_constants.DB_PATH)
        print("\n=== Starting Data Integrity Checks ===")
        print(f"Using Database: {db_path}")
        cls.issues = []
        cls.conn = get_db_connection()
        cls.conn.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        print("\n=== Data Integrity Report ===")
        has_failure = False
        if not cls.issues:
            print("OK: no data integrity issues found.")
        else:
            print(f"WARNING: found {len(cls.issues)} issues:")
            for issue in cls.issues:
                print(f"  - [{issue['type']}] {issue['message']}")
                if issue['type'] in ["ERROR", "CRITICAL"]:
                    has_failure = True
        print("==============================\n")
        if has_failure:
            print("ERROR: exiting with error code due to data integrity failures.")
            sys.exit(1)

    def _log_issue(self, issue_type: str, message: str):
        self.issues.append({"type": issue_type, "message": message})

    def test_01_individuals_media_linkage(self):
        """Constraint: Media Individuals must have a source ID pointing to a valid Source Individual."""
        cursor = self.conn.cursor()

        # Fetch all individuals.
        cursor.execute("SELECT id, media, source, reference FROM Individuals")
        individuals = cursor.fetchall()

        source_ids = set()
        for row in individuals:
             if row['media'] is None:
                 source_ids.add(row['id'])

        for row in individuals:
            ind_id = row['id']
            media = row['media']
            source = row['source']

            # Constraint 1: Media Individual (media IS NOT NULL).
            if media is not None:
                if source is None:
                    self._log_issue("CRITICAL", f"Individual {ind_id}: Is a Media Individual (media={media}) but has NO source.")
                elif source not in source_ids:
                     # Check if it points to a non-existent individual entirely.
                     # The SQL constraints usually enforce FK existence, but we verify logical "Is it a Source?".
                     self._log_issue("ERROR", f"Individual {ind_id}: Points to source {source}, but {source} is not a valid Source Individual (it might be another Media Individual or missing).")

            # Constraint 2: Source Individual (media IS NULL).
            else:
                if source != ind_id:
                     self._log_issue("ERROR", f"Individual {ind_id}: Is a Source Individual but source ({source}) does not point to self ({ind_id}).")

    def test_02_reference_constraints(self):
        """Constraint: Reference Individuals must be Source Individuals (media must be NULL)."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, media, reference FROM Individuals WHERE reference IS NOT NULL AND id != -1")
        refs = cursor.fetchall()

        for row in refs:
            ind_id = row['id']
            media = row['media']
            reference = row['reference']

            if media is not None:
                self._log_issue("ERROR", f"Individual {ind_id}: Has a reference ({reference}) but also has media ({media}). Reference Individuals must be Source Individuals.")

            # Check validity of reference target.
            cursor.execute("SELECT media, source FROM Individuals WHERE id = ?", (reference,))
            target = cursor.fetchone()
            if not target:
                self._log_issue("CRITICAL", f"Individual {ind_id}: Points to non-existent reference {reference}.")
            else:
                # Target must be a Media Individual (has media).
                if target['media'] is None:
                    self._log_issue("ERROR", f"Individual {ind_id}: Reference {reference} is not a valid Media Individual (it has NO media). Targets must be Sighting Images.")

                # Target must point back to this Source Individual.
                if target['source'] != ind_id:
                     self._log_issue("ERROR", f"Individual {ind_id}: Reference {reference} points to a different source ({target['source']}). References must belong to their parent shark.")

    def test_03_media_validity(self):
        """Constraint: Every Media record must be linked to an Individual."""
        cursor = self.conn.cursor()
        # Find media not referenced by any individual.
        cursor.execute("""
            SELECT m.id, m.filename 
            FROM Media m 
            LEFT JOIN Individuals i ON m.id = i.media 
            WHERE i.id IS NULL
        """)
        orphans = cursor.fetchall()

        for row in orphans:
            self._log_issue("ERROR", f"Media {row['id']} ({row['filename']}): Is orphaned! Not linked to any Individual.")

    def test_04_orphaned_associated_objects(self):
        """Constraint: Associated Objects must point to valid Individuals."""
        tables = ["AlternateNames", "Tags", "Labels"]
        cursor = self.conn.cursor()

        for table in tables:
            cursor.execute(f"""
                SELECT t.id, t.name, t.individual 
                FROM {table} t 
                LEFT JOIN Individuals i ON t.individual = i.id 
                WHERE i.id IS NULL
            """)
            orphans = cursor.fetchall()
            for row in orphans:
                self._log_issue("CRITICAL", f"{table} {row['id']} ('{row['name']}'): Points to non-existent Individual {row['individual']}.")

    def test_05_auth_integrity(self):
        """Constraint: Tokens must point to valid users and actions/proposals must have valid authors."""
        cursor = self.conn.cursor()

        # Tokens.
        cursor.execute("SELECT t.id, t.token, t.user FROM Tokens t LEFT JOIN Users u ON t.user = u.id WHERE u.id IS NULL")
        for row in cursor.fetchall():
            self._log_issue("CRITICAL", f"Token {row['id']}: Points to non-existent User {row['user']}.")

        # Actions.
        cursor.execute("SELECT a.id, a.action, a.author FROM Actions a LEFT JOIN Users u ON a.author = u.id WHERE u.id IS NULL")
        for row in cursor.fetchall():
             self._log_issue("CRITICAL", f"Action {row['id']} ({row['action']}): Author {row['author']} does not exist.")

        # Proposals.
        cursor.execute("SELECT p.id, p.action, p.author FROM Proposals p LEFT JOIN Users u ON p.author = u.id WHERE u.id IS NULL")
        for row in cursor.fetchall():
            self._log_issue("CRITICAL", f"Proposal {row['id']} ({row['action']}): Author {row['author']} does not exist.")

    def test_99_database_stats(self):
        """Prints an overview of the database statistics (row counts per table)."""
        print("\n--- Database Statistics ---")
        cursor = self.conn.cursor()

        # Get list of all tables.
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table['name']
            # Exclude sqlite internal tables if any (sequence, etc).
            if table_name.startswith('sqlite_'):
                continue

            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            count = cursor.fetchone()['count']
            print(f"{table_name:<20}: {count:>5} rows")
        print("---------------------------\n")


if __name__ == '__main__':
    # When running directly, ensure we are in the correct mode or explicitly run tests.
    unittest.main()
