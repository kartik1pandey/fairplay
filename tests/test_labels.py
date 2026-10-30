import json

import httpx
import pytest

from fairplay.labels import fetcher, store

FLAGGED = {"bob"}
CLOSED = {"carol"}
MISSING = {"dave"}


def fake_lichess(fail_first: dict | None = None):
    """MockTransport imitating POST /api/users. fail_first: {status: times} to inject errors."""
    fail_first = dict(fail_first or {})
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        for status, times in fail_first.items():
            if times:
                fail_first[status] -= 1
                return httpx.Response(status)
        users = []
        for name in request.content.decode().split(","):
            if name in MISSING:
                continue
            u = {"id": name, "username": name.title(), "createdAt": 1_600_000_000_000,
                 "perfs": {"blitz": {"rating": 1500}}}
            if name in FLAGGED:
                u["tosViolation"] = True
            if name in CLOSED:
                u = {"id": name, "username": name, "disabled": True}
            users.append(u)
        return httpx.Response(200, json=users)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


@pytest.fixture
def conn(tmp_path):
    return store.connect(tmp_path / "labels.sqlite")


def names(n):
    return [f"user{i:04d}" for i in range(n)] + ["Bob", "carol", "dave"]


def run(conn, client, **kw):
    fetcher.run_snapshot(conn, "S0", client=client, sleep=lambda s: None, log=lambda m: None, **kw)


def labels(conn):
    return conn.execute(
        "SELECT a.username, l.outcome, l.tos_violation, l.disabled FROM labels l "
        "JOIN accounts a USING (account_id) WHERE snapshot_id='S0' ORDER BY a.username"
    ).fetchall()


def test_uuid_map_is_random_and_stable(conn):
    store.add_accounts(conn, ["Alice"])
    store.add_accounts(conn, ["alice"])  # same account, different case
    rows = conn.execute("SELECT username, account_id FROM accounts").fetchall()
    assert len(rows) == 1 and rows[0][0] == "alice"
    assert len(rows[0][1]) == 36


def test_snapshot_batches_are_sorted_and_capped_at_300(conn):
    assert store.create_snapshot(conn, "S0", names(700)) == 703
    sizes = [n for (n,) in conn.execute("SELECT n_ids FROM batches ORDER BY batch_no")]
    assert sizes == [300, 300, 103]


def test_outcome_classes(conn):
    store.create_snapshot(conn, "S0", names(5))
    client, _ = fake_lichess()
    run(conn, client)
    got = {u: (o, t, d) for u, o, t, d in labels(conn)}
    assert got["bob"] == ("ok", 1, 0)
    assert got["carol"] == ("ok", 0, 1)
    assert got["dave"] == ("not_returned", None, None)  # never treated as a negative
    assert got["user0000"] == ("ok", 0, 0)


def test_only_allowed_fields_are_stored(conn):
    store.create_snapshot(conn, "S0", names(2))
    client, _ = fake_lichess()
    run(conn, client)
    (response,) = conn.execute("SELECT response FROM batches").fetchone()
    assert set(json.loads(response)["bob"]) == {"tos_violation", "disabled", "created_at_ms"}


def test_429_then_success_and_5xx_exhaustion(conn):
    store.create_snapshot(conn, "S0", names(5))
    client, _ = fake_lichess(fail_first={429: 2})
    run(conn, client)
    assert conn.execute("SELECT status, attempts FROM batches").fetchone() == ("done", 3)

    store.create_snapshot(conn, "S1", names(5))
    client, _ = fake_lichess(fail_first={503: 99})
    fetcher.run_snapshot(conn, "S1", client=client, sleep=lambda s: None, log=lambda m: None)
    assert store.summary(conn, "S1")["outcomes"] == {"http_error": 8}


def test_persistent_429_stops_the_run(conn):
    store.create_snapshot(conn, "S0", names(1000))  # 4 batches
    client, calls = fake_lichess(fail_first={429: 99})
    with pytest.raises(fetcher.RateLimited):
        run(conn, client)
    assert calls["n"] == fetcher.MAX_ATTEMPTS  # one batch's retries, nothing more
    assert store.summary(conn, "S0")["batches"] == {"http_error": 1, "pending": 3}


def test_forced_interruption_resumes_to_identical_result(conn, tmp_path):
    store.create_snapshot(conn, "S0", names(1000))  # 4 batches
    client, calls = fake_lichess()
    original = client._transport.handle_request

    def crash_on_third(request):
        if calls["n"] == 2:
            raise KeyboardInterrupt  # simulates Ctrl+C / power loss mid-run
        return original(request)

    client._transport.handle_request = crash_on_third
    with pytest.raises(KeyboardInterrupt):
        run(conn, client)
    assert store.summary(conn, "S0")["batches"] == {"done": 2, "pending": 2}

    client._transport.handle_request = original
    run(conn, client)
    resumed = labels(conn)

    ref = store.connect(tmp_path / "ref.sqlite")
    store.create_snapshot(ref, "S0", names(1000))
    run(ref, fake_lichess()[0])
    assert [r[1:] for r in resumed] == [r[1:] for r in labels(ref)]
    assert store.summary(conn, "S0")["batches"] == {"done": 4}


def test_daily_budget_stops_cleanly(conn):
    store.create_snapshot(conn, "S0", names(1000))
    client, _ = fake_lichess()
    with pytest.raises(fetcher.DailyBudgetReached):
        run(conn, client, daily_limit=650)
    assert store.summary(conn, "S0")["batches"] == {"done": 2, "pending": 2}


def test_pacing_respects_window_limit(conn):
    store.create_snapshot(conn, "S0", names(1000))
    client, _ = fake_lichess()
    t = {"now": 0.0}
    slept = []

    def sleep(s):
        slept.append(s)
        t["now"] += s

    fetcher.run_snapshot(conn, "S0", client=client, sleep=sleep, clock=lambda: t["now"],
                         log=lambda m: None)
    # 4 batches: the 2nd, 3rd and 4th each wait for the previous batch's 300 ids x 600/7200 s
    assert slept == pytest.approx([25.0, 25.0, 25.0])


def test_closed_snapshot_never_fetches_again(conn):
    store.create_snapshot(conn, "S0", names(1000))  # 4 batches
    client, calls = fake_lichess(fail_first={429: 99})
    with pytest.raises(fetcher.RateLimited):
        run(conn, client)
    assert store.close_snapshot(conn, "S0", "rate limited") == 4
    assert store.summary(conn, "S0")["batches"] == {"abandoned": 4}
    assert store.summary(conn, "S0")["outcomes"] == {"http_error": 300}  # refused batch kept
    n = calls["n"]
    run(conn, fake_lichess()[0], retry_errors=True)
    assert calls["n"] == n and store.summary(conn, "S0")["batches"] == {"abandoned": 4}
