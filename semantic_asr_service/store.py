import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}
INTERNAL_JOB_PREFIXES = ("translation_smoke_",)


class JobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    config TEXT NOT NULL,
                    filename TEXT,
                    local_path TEXT,
                    wav_path TEXT NOT NULL,
                    outdir TEXT NOT NULL,
                    formats TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            if "filename" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN filename TEXT")
            if "local_path" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN local_path TEXT")

    def create_job(
        self,
        job_id: str,
        user_id: str,
        config: str,
        wav_path: str,
        outdir: str,
        formats: list[str],
        filename: str = "",
        local_path: str = "",
    ) -> dict[str, Any]:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, user_id, config, filename, local_path, wav_path, outdir, formats, status,
                    progress, error, created_at, started_at, finished_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, NULL, ?, NULL, NULL)
                """,
                (
                    job_id,
                    user_id,
                    config,
                    filename,
                    local_path,
                    wav_path,
                    outdir,
                    json.dumps(formats),
                    json.dumps({"stage": "queued"}),
                    now,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def delete_job(self, job_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        return cur.rowcount > 0

    def list_jobs(
        self,
        user_id: str,
        include_all: bool = False,
        limit: int = 100,
        offset: int = 0,
        config: str | None = None,
        filter_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(500, int(limit)))
        offset = max(0, int(offset))
        with self.connect() as conn:
            where_clause, params = self._list_filter(user_id, include_all, config, filter_user_id)
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def count_jobs(
        self,
        user_id: str,
        include_all: bool = False,
        config: str | None = None,
        filter_user_id: str | None = None,
    ) -> int:
        with self.connect() as conn:
            where_clause, params = self._list_filter(user_id, include_all, config, filter_user_id)
            row = conn.execute(f"SELECT COUNT(*) AS count FROM jobs WHERE {where_clause}", params).fetchone()
        return int(row["count"])

    def _list_filter(
        self,
        user_id: str,
        include_all: bool,
        config: str | None,
        filter_user_id: str | None,
    ) -> tuple[str, tuple[str, ...]]:
        clauses = []
        params: list[str] = []
        if include_all:
            if filter_user_id:
                clauses.append("user_id = ?")
                params.append(filter_user_id)
        else:
            clauses.append("user_id = ?")
            params.append(user_id)
            if filter_user_id and filter_user_id != user_id:
                clauses.append("0 = 1")
        if config:
            clauses.append("config = ?")
            params.append(config)
        hidden_clause, hidden_params = _hidden_jobs_filter()
        clauses.append(hidden_clause)
        params.extend(hidden_params)
        return " AND ".join(clauses), tuple(params)

    def claim_next_job(self) -> dict[str, Any] | None:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ?, progress = ? WHERE job_id = ?",
                (now, json.dumps({"stage": "running"}), row["job_id"]),
            )
            conn.commit()
        return self.get_job(row["job_id"])

    def mark_succeeded(self, job_id: str, progress: dict[str, Any] | None = None) -> None:
        self._update_terminal(job_id, "succeeded", progress or {"stage": "done"}, None)

    def mark_failed(self, job_id: str, error: str, progress: dict[str, Any] | None = None) -> None:
        self._update_terminal(job_id, "failed", progress or {"stage": "failed"}, error)

    def cancel_job(self, job_id: str) -> bool:
        now = _utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled', finished_at = ?, progress = ?
                WHERE job_id = ? AND status = 'queued'
                """,
                (now, json.dumps({"stage": "canceled"}), job_id),
            )
        return cur.rowcount > 0

    def retry_job(self, job_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'queued', progress = ?, error = NULL, started_at = NULL, finished_at = NULL
                WHERE job_id = ? AND status IN ('failed', 'canceled')
                """,
                (json.dumps({"stage": "queued"}), job_id),
            )
        return cur.rowcount > 0

    def _update_terminal(
        self,
        job_id: str,
        status: str,
        progress: dict[str, Any],
        error: str | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, progress = ?, error = ?, finished_at = ?
                WHERE job_id = ?
                """,
                (status, json.dumps(progress), error, _utc_now(), job_id),
            )


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    job = dict(row)
    job["formats"] = json.loads(job["formats"])
    job["progress"] = json.loads(job["progress"])
    return job


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hidden_jobs_filter() -> tuple[str, tuple[str, ...]]:
    clause = " AND ".join("job_id NOT LIKE ?" for _ in INTERNAL_JOB_PREFIXES) or "1 = 1"
    params = tuple(f"{prefix}%" for prefix in INTERNAL_JOB_PREFIXES)
    return clause, params
