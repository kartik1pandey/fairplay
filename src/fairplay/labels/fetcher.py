"""Resumable, rate-limited snapshot fetcher for POST https://lichess.org/api/users.

Official limits (lichess-org/api OpenAPI spec, read 2026-10-28): 300 ids per request,
8,000 users per 10 minutes, 120,000 per day. We use 90% of each, one request at a time,
and wait 60 s after any HTTP 429 (Lichess API tips).

Each batch is written in one transaction (labels + batch status), so killing the process at any
moment loses at most the batch in flight; re-running continues with the pending batches.
"""

import json
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx

from fairplay.labels.store import now_utc

API_URL = "https://lichess.org/api/users"
USER_AGENT = "fairplay-research/0.1 (student research project; sampled accounts only)"
WINDOW_SECONDS = 600
WINDOW_LIMIT = int(8_000 * 0.9)
DAILY_LIMIT = int(120_000 * 0.9)
RATE_LIMIT_WAIT = 60
MAX_ATTEMPTS = 5


class DailyBudgetReached(Exception):
    pass


class RateLimited(Exception):
    """A batch was still refused with 429 after every retry: Lichess's limit is longer than our
    waits, so the run stops instead of sending the rest of the queue into the same refusal."""


def seconds_per_id() -> float:
    return WINDOW_SECONDS / WINDOW_LIMIT  # 300 ids -> one request every 25 s


def ids_sent_last_24h(conn: sqlite3.Connection) -> int:
    since = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="seconds")
    (n,) = conn.execute(
        "SELECT COALESCE(SUM(n_ids), 0) FROM batches WHERE fetched_at >= ?", (since,)
    ).fetchone()
    return n


def parse_users(body: list[dict]) -> dict[str, dict]:
    """Keep only the fields the design allows (§9): tosViolation, disabled, createdAt."""
    return {
        u["id"]: {
            "tos_violation": int(bool(u.get("tosViolation"))),
            "disabled": int(bool(u.get("disabled"))),
            "created_at_ms": u.get("createdAt"),
        }
        for u in body
    }


def _write_batch(
    conn: sqlite3.Connection,
    snapshot_id: str,
    batch_no: int,
    usernames: list[str],
    status: str,
    http_status: int | None,
    attempts: int,
    users: dict[str, dict] | None,
) -> None:
    fetched = now_utc()
    ids = dict(
        conn.execute(
            f"SELECT username, account_id FROM accounts WHERE username IN "
            f"({','.join('?' * len(usernames))})",
            usernames,
        ).fetchall()
    )
    rows = []
    for name in usernames:
        if users is None:
            rows.append((snapshot_id, ids[name], "http_error", None, None, None, fetched))
        elif name in users:
            u = users[name]
            rows.append((snapshot_id, ids[name], "ok", u["tos_violation"], u["disabled"],
                         u["created_at_ms"], fetched))
        else:  # deleted, renamed or never existed: excluded and counted, never a negative
            rows.append((snapshot_id, ids[name], "not_returned", None, None, None, fetched))
    with conn:
        conn.executemany("INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.execute(
            "UPDATE batches SET status=?, http_status=?, attempts=?, fetched_at=?, response=? "
            "WHERE snapshot_id=? AND batch_no=?",
            (status, http_status, attempts, fetched,
             json.dumps(users, sort_keys=True) if users is not None else None,
             snapshot_id, batch_no),
        )


def fetch_batch(
    client: httpx.Client, usernames: list[str], sleep: Callable[[float], None]
) -> tuple[int | None, int, dict[str, dict] | None]:
    """Returns (http_status, attempts, users or None on failure)."""
    status = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.post(
                API_URL, params={"profile": "false"}, content=",".join(usernames),
                headers={"Content-Type": "text/plain"},
            )
            status = resp.status_code
        except httpx.TransportError:
            sleep(2**attempt)
            continue
        if status == 200:
            return status, attempt, parse_users(resp.json())
        if status == 429:
            sleep(RATE_LIMIT_WAIT)
        elif status >= 500:
            sleep(2**attempt)
        else:  # other 4xx: our request is wrong, retrying will not help
            return status, attempt, None
    return status, MAX_ATTEMPTS, None


def run_snapshot(
    conn: sqlite3.Connection,
    snapshot_id: str,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    daily_limit: int = DAILY_LIMIT,
    retry_errors: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    if retry_errors:
        with conn:
            conn.execute(
                "UPDATE batches SET status='pending' WHERE snapshot_id=? AND status='http_error'",
                (snapshot_id,),
            )
    pending = conn.execute(
        "SELECT batch_no, usernames FROM batches WHERE snapshot_id=? AND status='pending' "
        "ORDER BY batch_no",
        (snapshot_id,),
    ).fetchall()
    own_client = client is None
    client = client or httpx.Client(timeout=60, headers={"User-Agent": USER_AGENT})
    next_allowed = clock()
    try:
        for i, (batch_no, names_json) in enumerate(pending, 1):
            usernames = json.loads(names_json)
            if ids_sent_last_24h(conn) + len(usernames) > daily_limit:
                raise DailyBudgetReached(
                    f"daily budget of {daily_limit} ids reached; re-run in a few hours"
                )
            wait = next_allowed - clock()
            if wait > 0:
                sleep(wait)
            next_allowed = clock() + len(usernames) * seconds_per_id()
            http_status, attempts, users = fetch_batch(client, usernames, sleep)
            status = "done" if users is not None else "http_error"
            _write_batch(conn, snapshot_id, batch_no, usernames, status, http_status, attempts,
                         users)
            if http_status == 429 and users is None:
                raise RateLimited(f"batch {batch_no} still rate-limited after {attempts} tries; "
                                  "stopped. Re-run later with --retry-errors")
            if i % 20 == 0 or i == len(pending):
                log(f"{snapshot_id}: {i}/{len(pending)} batches this run")
    finally:
        if own_client:
            client.close()
