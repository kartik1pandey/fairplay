"""Tier-2 job table (design §11): unique (account, cutoff), bounded, leased, retried.

- enqueue: one job per (account, cutoff); a duplicate message is a no-op; at most 2,000 jobs
  per cutoff, extra candidates are deferred to the next run.
- lease: a worker takes the next runnable job for LEASE seconds. A crashed worker never
  completes, its lease expires and another worker takes the job (that counts as an attempt).
- fail: only timeouts and engine crashes retry, 3 attempts with exponential backoff; bad input
  fails at once (retrying cannot fix it).
- complete: the first completion wins; a late duplicate from a timed-out worker is ignored, so a
  result is never double-counted.
Every rule is one guarded UPDATE/INSERT, so it holds under concurrent workers on Postgres too.
"""

import datetime as dt
import json

from fairplay.service.db import Database
from fairplay.service.store import iso

MAX_PER_CUTOFF = 2000
MAX_ATTEMPTS = 3
LEASE = dt.timedelta(minutes=10)
BACKOFF = dt.timedelta(seconds=30)  # 30 s, 60 s, 120 s ...
RETRYABLE = {"timeout", "engine_crash"}


def job_key(account_id: str, cutoff: str) -> str:
    return f"{account_id}|{cutoff}"


def enqueue(db: Database, account_id: str, cutoff: str, now: dt.datetime,
            limit: int = MAX_PER_CUTOFF) -> str:
    with db.tx() as t:
        n = t.run("SELECT count(*) FROM jobs WHERE cutoff = ?", (cutoff,)).one()[0]
        if n >= limit:
            return "deferred"
        new = t.run("""
            INSERT INTO jobs (job_key, account_id, cutoff, status, not_before, created_at,
                              updated_at)
            VALUES (?, ?, ?, 'queued', ?, ?, ?)
            ON CONFLICT (job_key) DO NOTHING RETURNING job_key
        """, (job_key(account_id, cutoff), account_id, cutoff, iso(now), iso(now),
              iso(now))).one()
    return "queued" if new else "duplicate"


RUNNABLE = """((status = 'queued' AND not_before <= ?)
              OR (status = 'leased' AND lease_until < ? AND attempts < ?))"""


def lease(db: Database, worker: str, now: dt.datetime) -> dict | None:
    n = iso(now)
    with db.tx() as t:
        t.run("""UPDATE jobs SET status = 'failed', last_error = 'lease expired after max attempts',
                 updated_at = ? WHERE status = 'leased' AND lease_until < ? AND attempts >= ?""",
              (n, n, MAX_ATTEMPTS))
        row = t.run(f"""
            UPDATE jobs SET status = 'leased', lease_owner = ?, lease_until = ?,
                            attempts = attempts + 1, updated_at = ?
            WHERE job_key = (SELECT job_key FROM jobs WHERE {RUNNABLE}
                             ORDER BY cutoff, job_key LIMIT 1)
              AND {RUNNABLE}
            RETURNING job_key, account_id, cutoff, attempts
        """, (worker, iso(now + LEASE), n, n, n, MAX_ATTEMPTS, n, n, MAX_ATTEMPTS)).one()
    if row is None:
        return None
    return dict(zip(("job_key", "account_id", "cutoff", "attempts"), row, strict=True))


def complete(db: Database, key: str, result: dict, now: dt.datetime) -> bool:
    with db.tx() as t:
        done = t.run("""UPDATE jobs SET status = 'done', result = ?, lease_owner = NULL,
                        lease_until = NULL, updated_at = ?
                        WHERE job_key = ? AND status IN ('leased', 'queued')
                        RETURNING job_key""", (json.dumps(result), iso(now), key)).one()
    return done is not None


def fail(db: Database, key: str, worker: str, kind: str, error: str, now: dt.datetime) -> str:
    """Only the current lease holder can fail a job (a stale worker cannot re-queue it)."""
    with db.tx() as t:
        attempts = t.run("SELECT attempts FROM jobs WHERE job_key = ? AND status = 'leased' "
                         "AND lease_owner = ?", (key, worker)).one()
        if attempts is None:
            return "ignored"  # lease already lost or job finished
        if kind in RETRYABLE and attempts[0] < MAX_ATTEMPTS:
            retry_at = now + BACKOFF * 2 ** (attempts[0] - 1)
            t.run("""UPDATE jobs SET status = 'queued', not_before = ?, last_error = ?,
                     lease_owner = NULL, lease_until = NULL, updated_at = ? WHERE job_key = ?""",
                  (iso(retry_at), f"{kind}: {error}", iso(now), key))
            return "retry"
        t.run("""UPDATE jobs SET status = 'failed', last_error = ?, lease_owner = NULL,
                 lease_until = NULL, updated_at = ? WHERE job_key = ?""",
              (f"{kind}: {error}", iso(now), key))
        return "failed"


def counts(db: Database) -> dict[str, int]:
    with db.tx(write=False) as t:
        return dict(t.run("SELECT status, count(*) FROM jobs GROUP BY status").all())
