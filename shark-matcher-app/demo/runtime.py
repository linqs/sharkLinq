"""Demo lifecycle and ground truth; ordinary mutations retain their shared owners."""

from contextlib import contextmanager, closing
from contextvars import ContextVar
from pathlib import Path
import random
import tempfile
import threading
import uuid

from core import app_utils
import backend.db.individual


_CURRENT = ContextVar("demo_session", default=None)
DATA_TABLES = ("Media", "Individuals", "AlternateNames", "Tags", "Labels")


def current():
    """Return the demo session only inside a demo data request."""
    return _CURRENT.get()


class DemoSession:
    """Own one startup baseline and one mutable round for the server."""

    def __init__(self, source_path):
        self._directory = tempfile.TemporaryDirectory(prefix="shark-demo-")
        self.baseline_path = Path(self._directory.name) / "baseline.db"
        self.database_path = Path(self._directory.name) / "round.db"
        self.lock = threading.RLock()
        self.round_id = uuid.uuid4().hex
        try:
            with closing(app_utils.db_connect(source_path)) as source, \
                    closing(app_utils.db_connect(self.baseline_path)) as baseline:
                source.backup(baseline)
                # Accounts and history belong to the live application, never a round.
                for table in ("Actions", "Proposals", "Tokens", "Users"):
                    baseline.execute(f"DELETE FROM {table}")
                baseline.executescript("""
                    CREATE TABLE DemoTruth (
                        individual INTEGER PRIMARY KEY, original_source INTEGER NOT NULL,
                        original_name TEXT NOT NULL, image_name TEXT NOT NULL
                    );
                    INSERT INTO DemoTruth
                    SELECT image.id, image.source, COALESCE(reference.name, CAST(shark.id AS TEXT)),
                           COALESCE(image.name, CAST(image.id AS TEXT))
                    FROM Individuals image
                    JOIN Individuals shark ON shark.id = image.source
                    LEFT JOIN Individuals reference ON reference.id = shark.reference
                    WHERE image.id >= 0 AND image.media >= 0;
                    CREATE TABLE DemoSelections (individual INTEGER PRIMARY KEY);
                    CREATE TABLE DemoTargets (original_source INTEGER PRIMARY KEY, anchor INTEGER NOT NULL);
                """)
                baseline.commit()
                with closing(app_utils.db_connect(self.database_path)) as working:
                    baseline.backup(working)
        except Exception:
            self.close()
            raise

    def close(self):
        """Discard all demo state on server shutdown."""
        self._directory.cleanup()

    @contextmanager
    def scope(self):
        """Serialize round access and select its database for shared operations."""
        with self.lock, app_utils.database_scope(self.database_path):
            token = _CURRENT.set(self)
            try:
                yield
            finally:
                _CURRENT.reset(token)

    def status(self):
        """Return round information without revealing any answers."""
        with closing(app_utils.db_connect(self.database_path)) as connection:
            maximum = connection.execute("""
                SELECT COALESCE(SUM(count - 1), 0) FROM (
                    SELECT COUNT(*) AS count FROM DemoTruth GROUP BY original_source
                )
            """).fetchone()[0]
            selected = connection.execute("SELECT COUNT(*) FROM DemoSelections").fetchone()[0]
        return {"round_id": self.round_id, "maximum": maximum, "selected": selected}

    def randomize(self, count):
        """Restore and detach a deterministic sample in one rollback-safe transaction."""
        return self._replace_round(count)

    def clear(self):
        """Restore the startup dataset without detaching any observations."""
        return self._replace_round(0, allow_empty=True)

    def _replace_round(self, count, allow_empty=False):
        """Share atomic baseline restoration between randomization and clearing."""
        # Imported here to keep the existing unmatch implementation authoritative.
        from frontend.individual import detach_individual

        with self.scope(), closing(app_utils.db_connect()) as connection:
            maximum = self.status()["maximum"]
            minimum = 0 if allow_empty else 1
            if type(count) is not int or not minimum <= count <= maximum:
                raise ValueError(f"Choose a whole number from 1 to {maximum}.")
            candidates = connection.execute(
                "SELECT individual, original_source FROM DemoTruth ORDER BY individual"
            ).fetchall()
            remaining = {}
            for _, source in candidates:
                remaining[source] = remaining.get(source, 0) + 1
            random.Random(42).shuffle(candidates)
            selected = []
            for individual, source in candidates:
                if len(selected) == count:
                    break
                if remaining[source] > 1:
                    selected.append(individual)
                    remaining[source] -= 1

            connection.execute("ATTACH DATABASE ? AS baseline", (str(self.baseline_path),))
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                for table in reversed(DATA_TABLES):
                    connection.execute(f"DELETE FROM {table}")
                for table in DATA_TABLES:
                    connection.execute(f"INSERT INTO {table} SELECT * FROM baseline.{table}")
                for table in ("DemoSelections", "DemoTargets", "Actions", "Proposals"):
                    connection.execute(f"DELETE FROM {table}")
                for individual in selected:
                    connection.execute("INSERT INTO DemoSelections VALUES (?)", (individual,))
                    detach_individual(individual, connection, labels=("unmatched", "randomized"))
                if selected:
                    connection.execute("""
                        INSERT INTO DemoTargets
                        SELECT original_source, MIN(individual) FROM DemoTruth
                        WHERE individual NOT IN (SELECT individual FROM DemoSelections)
                        GROUP BY original_source
                    """)
            self.round_id = uuid.uuid4().hex
            return self.status()

    def keys(self, source_id, reveal=False):
        """Find randomized observations on a profile; reveal immutable keys on demand."""
        with closing(app_utils.db_connect()) as connection:
            rows = connection.execute("""
                SELECT truth.individual, truth.image_name, truth.original_source, truth.original_name
                FROM DemoSelections selected JOIN DemoTruth truth USING (individual)
                JOIN Individuals image ON image.id = selected.individual
                WHERE image.source = ? ORDER BY truth.individual
            """, (source_id,)).fetchall()
        if not reveal:
            return {"count": len(rows)}
        return {"keys": [dict(zip(
            ("individual", "image_name", "original_source", "original_name"), row
        )) for row in rows]}

    def grade_merge(self, source_id, connection):
        """Grade rejoining the original retained gallery, independent of surviving IDs."""
        rows = connection.execute("""
            SELECT image.id, truth.original_source, selected.individual
            FROM Individuals image LEFT JOIN DemoTruth truth ON truth.individual = image.id
            LEFT JOIN DemoSelections selected ON selected.individual = image.id
            WHERE image.source = ? AND image.media >= 0
        """, (source_id,)).fetchall()
        members = {row[0] for row in rows}
        truths = {row[1] for row in rows}
        results = []
        for individual, original_source, selected in rows:
            if selected is None:
                continue
            anchor = connection.execute(
                "SELECT anchor FROM DemoTargets WHERE original_source = ?", (original_source,)
            ).fetchone()
            results.append({"individual": individual, "correct": bool(
                truths == {original_source} and anchor and anchor[0] in members
            )})
        if results:
            # The tracking label cannot be lost through the client merge preview.
            exists = connection.execute(
                "SELECT 1 FROM Labels WHERE individual = ? AND name = 'randomized'", (source_id,)
            ).fetchone()
            if not exists:
                label = backend.db.individual.AssociatedObject(individual=source_id, name="randomized")
                backend.db.individual.upsert_associated_object(label, "Labels", connection)
        return results
