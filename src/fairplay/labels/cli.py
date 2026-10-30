"""Label snapshot commands.

uv run python -m fairplay.labels.cli create S0 --accounts private/accounts_eligible.txt
uv run python -m fairplay.labels.cli run S0          # resumable; safe to Ctrl+C and re-run
uv run python -m fairplay.labels.cli summary S0
"""

import argparse
import json
import math
from pathlib import Path

from fairplay.labels import fetcher, store


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="freeze an account list into a new snapshot")
    c.add_argument("snapshot_id")
    c.add_argument("--accounts", type=Path, required=True, help="one username per line")
    c.add_argument("--note", default="")
    r = sub.add_parser("run", help="fetch pending batches")
    r.add_argument("snapshot_id")
    r.add_argument("--retry-errors", action="store_true")
    cl = sub.add_parser("close", help="freeze a snapshot; unfinished batches are abandoned")
    cl.add_argument("snapshot_id")
    cl.add_argument("--reason", required=True)
    s = sub.add_parser("summary")
    s.add_argument("snapshot_id")
    args = p.parse_args()

    conn = store.connect(args.db)
    if args.cmd == "create":
        names = [n.strip() for n in args.accounts.read_text().splitlines() if n.strip()]
        n = store.create_snapshot(conn, args.snapshot_id, names, args.note)
        hours = n * fetcher.seconds_per_id() / 3600
        days = math.ceil(n / fetcher.DAILY_LIMIT)
        print(f"{n} accounts in {math.ceil(n / store.BATCH_SIZE)} batches; "
              f"~{hours:.1f} h of requests, spread over at least {days} day(s)")
    elif args.cmd == "run":
        try:
            fetcher.run_snapshot(conn, args.snapshot_id, retry_errors=args.retry_errors)
        except (fetcher.DailyBudgetReached, fetcher.RateLimited) as e:
            print(e)
        print(json.dumps(store.summary(conn, args.snapshot_id), indent=2))
    elif args.cmd == "close":
        n = store.close_snapshot(conn, args.snapshot_id, args.reason)
        print(f"{args.snapshot_id}: closed, {n} batches abandoned")
        print(json.dumps(store.summary(conn, args.snapshot_id), indent=2))
    else:
        print(json.dumps(store.summary(conn, args.snapshot_id), indent=2))


if __name__ == "__main__":
    main()
