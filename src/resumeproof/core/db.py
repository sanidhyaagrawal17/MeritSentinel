"""SQLite database management with WAL mode and ACID transaction helpers."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
import logging

logger = logging.getLogger("resumeproof.db")

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS profile_facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    persona_id TEXT,
    fact_type TEXT NOT NULL,
    value TEXT NOT NULL,
    source_line INTEGER,
    source_page INTEGER,
    is_confirmed INTEGER NOT NULL DEFAULT 0,
    user_correction TEXT,
    confirmed_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    persona_id TEXT,
    project_id TEXT,
    claim_type TEXT NOT NULL,
    text TEXT NOT NULL,
    tools TEXT NOT NULL, -- JSON list
    metric_values TEXT NOT NULL, -- JSON list
    source_line INTEGER,
    source_page INTEGER,
    raw_sentence TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_audits (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    extractor_text TEXT NOT NULL,
    extractor_tools TEXT NOT NULL, -- JSON list
    extractor_metrics TEXT NOT NULL, -- JSON list
    user_correction_text TEXT,
    user_correction_tools TEXT, -- JSON list
    user_correction_metrics TEXT, -- JSON list
    action_taken TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY(claim_id) REFERENCES claims(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS answer_records (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    question_id TEXT,
    raw_score INTEGER NOT NULL,
    evidence_quote TEXT,
    feedback TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY(claim_id) REFERENCES claims(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS mastery_states (
    claim_id TEXT PRIMARY KEY,
    aggregate_score REAL NOT NULL,
    effective_score REAL NOT NULL,
    confidence REAL NOT NULL,
    n_effective REAL NOT NULL,
    total_answers INTEGER NOT NULL,
    is_eligible INTEGER NOT NULL,
    last_updated TEXT NOT NULL,
    FOREIGN KEY(claim_id) REFERENCES claims(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rubrics (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    persona_id TEXT,
    topic TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_line INTEGER NOT NULL,
    source_quote TEXT NOT NULL,
    criteria TEXT NOT NULL, -- JSON dict of 0-4 scoring anchors
    atomic_points TEXT DEFAULT '[]', -- JSON array of AtomicPoint objects
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interview_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    persona_id TEXT,
    budget INTEGER NOT NULL,
    total_questions INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS session_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    claim_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    question_text TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    raw_score INTEGER,
    evidence_quote TEXT,
    rubric_reference TEXT,
    feedback TEXT,
    is_followup INTEGER NOT NULL DEFAULT 0,
    turn_latency_ms REAL NOT NULL DEFAULT 0.0,
    timestamp TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES interview_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(claim_id) REFERENCES claims(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS grading_cache (
    cache_key TEXT PRIMARY KEY,
    raw_score INTEGER NOT NULL,
    evidence_quote TEXT NOT NULL,
    rubric_reference TEXT,
    feedback TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_user ON claims(user_id);
CREATE INDEX IF NOT EXISTS idx_profile_facts_user ON profile_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_answers_claim ON answer_records(claim_id);
CREATE INDEX IF NOT EXISTS idx_rubrics_project ON rubrics(project_id);
CREATE INDEX IF NOT EXISTS idx_turns_session ON session_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_claim ON session_turns(claim_id);
"""

class Database:
    def __init__(self, db_path: str | Path):
        self.is_memory = str(db_path) == ":memory:"
        self.db_path = Path(db_path)
        self._shared_conn: Optional[sqlite3.Connection] = None
        if not self.is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys = ON;")
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        if self.is_memory and self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if not self.is_memory:
                conn.close()

    def init_schema(self) -> None:
        with self.transaction() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.debug(f"Initialized database schema at {self.db_path}")

    def close(self) -> None:
        if self._shared_conn is not None:
            try:
                self._shared_conn.close()
            except Exception:
                pass
            self._shared_conn = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
