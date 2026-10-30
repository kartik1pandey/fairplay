"""Privacy scan (design §9, §12): no username may appear in any output meant to leave private/.

Tokenises each text file into Lichess-username-shaped tokens ([A-Za-z0-9_-], case-insensitive)
and intersects them with the private username list. Set lookup per token, so a million names
cost nothing extra. Hits are reported as file + line only; the matched name is never printed,
because the report itself would otherwise leak it. Names shorter than 4 characters are skipped:
they collide with ordinary words and ids ("e4", "max") and would make the check useless noise.

uv run python -m fairplay.privacy runs docs README.md                 (local, private name list)
uv run python -m fairplay.privacy --published README.md docs results  (CI, no private data)
"""

import argparse
import re
import sys
from pathlib import Path

from fairplay.labels import store

TOKEN = re.compile(r"[A-Za-z0-9_-]{4,}")
TEXT_SUFFIXES = {".json", ".md", ".txt", ".csv", ".html", ".log", ".py", ".yml", ".yaml"}
MIN_LEN = 4


def load_usernames(conn=None) -> set[str]:
    conn = conn or store.connect()
    return {u.lower() for (u,) in conn.execute("SELECT username FROM accounts")
            if len(u) >= MIN_LEN}


def scan(paths: list[Path], usernames: set[str]) -> list[tuple[str, int]]:
    hits = []
    files = [f for p in paths for f in ([p] if p.is_file() else sorted(p.rglob("*")))
             if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES]
    for f in files:
        with f.open(encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if any(t.lower() in usernames for t in TOKEN.findall(line)):
                    hits.append((f.as_posix(), i))
    return hits


UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
ACCOUNT_COLUMNS = {"account_id", "account", "username", "user", "white", "black", "game_key",
                   "game_id"}
TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".json", ".jsonl"}


def _columns(f: Path) -> set[str]:
    if f.suffix == ".parquet":
        import pyarrow.parquet as pq

        return {c.lower() for c in pq.read_schema(f).names}
    head = f.read_text(encoding="utf-8", errors="replace")[:20_000]
    if f.suffix in {".csv", ".tsv"}:
        first = head.splitlines()[0] if head else ""
        return {c.strip().strip('"').lower() for c in re.split(r"[,\t]", first)} - {""}
    return {k.lower() for k in re.findall(r'"([A-Za-z_]+)"\s*:', head)}


def check_published(paths: list[Path], allow_uuids: frozenset[str] = frozenset()) -> list[str]:
    """CI guard for anything meant to be published (README, docs, results/): runs WITHOUT the
    private username list, so it blocks the other leak path instead: account-level rows.
    - any UUID (real account ids are random UUIDs; synthetic ids are 'syn-...'), except code
      constants passed in allow_uuids
    - any table whose columns include an account- or game-level identifier."""
    problems = []
    files = [f for p in paths for f in ([p] if p.is_file() else sorted(p.rglob("*")))
             if f.is_file()]
    for f in files:
        if f.suffix.lower() in TABLE_SUFFIXES and _columns(f) & ACCOUNT_COLUMNS:
            problems.append(f"{f.as_posix()}: account-level column "
                            f"{sorted(_columns(f) & ACCOUNT_COLUMNS)}")
        if f.suffix.lower() in TEXT_SUFFIXES | TABLE_SUFFIXES - {".parquet"}:
            with f.open(encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if any(u.lower() not in allow_uuids for u in UUID_RE.findall(line)):
                        problems.append(f"{f.as_posix()}:{i}: UUID (possible account id)")
    return problems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--published", action="store_true",
                   help="CI mode: no username list needed; block account-level outputs")
    args = p.parse_args()
    if args.published:
        problems = check_published([x for x in args.paths if x.exists()])
        for line in problems:
            print(line)
        print(f"{len(problems)} problem(s)")
        sys.exit(1 if problems else 0)
    hits = scan(args.paths, load_usernames())
    for f, line in hits:
        print(f"username-like token: {f}:{line}")
    print(f"{len(hits)} hit(s)")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
