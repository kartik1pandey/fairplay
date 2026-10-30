"""Hugging Face mirror of the Lichess database -> sampled blitz games as Parquet, per file.

Why this exists (DECISIONS.md 2026-10-28): database.lichess.org serves ~0.08 MB/s and the 2026
torrents ran at ~37 KB/s; the official Lichess dataset on Hugging Face serves ~16 MB/s, but stops
at September 2025. Experiment R therefore uses June-September 2025.

Each ~1 GB source file is downloaded with ONE GET (remote Parquet reads make thousands of range
requests and hit HF's 3,000 requests / 5 min anonymous limit), verified against its published
SHA256, filtered locally in one DuckDB pass, then deleted. The next file downloads while the
current one is filtered.

The filter is the same rule as prefilter.py, written as DuckDB macros; tests assert both agree.
Output contains usernames, so it goes to private/; Part 4 replaces them with UUIDs.

uv run python -m fairplay.ingest.hf_month 2025-06 2025-07 2025-08 2025-09 --rate-bp 1000
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import duckdb
import httpx

from fairplay.ingest.prefilter import BLITZ_MAX_SECONDS, BLITZ_MIN_SECONDS, SAMPLE_BUCKETS

REPO = "Lichess/standard-chess-games"
REVISION = "de4e636eddf568a9394cc01fb0b9e1da04a6babf"  # pinned 2026-10-28 for reproducibility
OUT_ROOT = Path("private/games_sampled")
TMP_DIR = Path("private/tmp")

KEEP_COLUMNS = """
    regexp_extract(Site, '[^/]+$') AS game_id, Event, White, Black, WhiteTitle, BlackTitle,
    Result, WhiteElo, BlackElo, WhiteRatingDiff, BlackRatingDiff, UTCDate, UTCTime,
    Termination, TimeControl, movetext
"""


def install_macros(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
        CREATE OR REPLACE MACRO sample_bucket(name) AS
            ('0x' || left(sha256(lower(name)), 16))::UBIGINT % {SAMPLE_BUCKETS};
        CREATE OR REPLACE MACRO is_blitz(tc) AS
            COALESCE(
                TRY_CAST(split_part(tc, '+', 1) AS INTEGER)
                + 40 * TRY_CAST(split_part(tc, '+', 2) AS INTEGER)
                BETWEEN {BLITZ_MIN_SECONDS} AND {BLITZ_MAX_SECONDS},
                false);
    """)


def list_files(month: str) -> list[dict]:
    """[{path, size, sha256}] for one month, from the pinned revision."""
    year, mm = month.split("-")
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data/year={year}/month={mm}"
    entries = httpx.get(url, timeout=60).raise_for_status().json()
    return sorted(
        ({"path": e["path"], "size": e["size"], "sha256": e["lfs"]["oid"]}
         for e in entries if e["path"].endswith(".parquet")),
        key=lambda e: e["path"],
    )


def download(entry: dict, dest: Path, attempts: int = 5) -> dict:
    """Parallel ranged download, then SHA256 check against the published LFS hash.

    Measured 2026-10-28: a single connection sometimes stalls at ~50 KB/s on a bad CDN route while
    a fresh one gets MB/s. So the file is split into CHUNK-sized ranges fetched by WORKERS threads;
    a range slower than CHUNK_DEADLINE is abandoned and retried on a new connection.
    One HF resolve request per file (the 3,000/5 min limit); the ranges go to the CDN URL.
    """
    start = time.perf_counter()
    for _ in range(attempts):
        cdn = _CdnUrl(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{entry['path']}")
        with open(dest, "wb") as fh:
            fh.truncate(entry["size"])
        ranges = [(s, min(s + CHUNK, entry["size"]) - 1) for s in range(0, entry["size"], CHUNK)]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            retries = sum(pool.map(lambda r, c=cdn: _fetch_range(c, dest, *r), ranges))
        if _sha256_file(dest) == entry["sha256"]:
            return {"download_seconds": round(time.perf_counter() - start, 1),
                    "range_retries": retries}
        dest.unlink(missing_ok=True)  # corrupt: start again
    raise RuntimeError(f"could not download {entry['path']} after {attempts} attempts")


CHUNK = 32 << 20  # 32 MiB
WORKERS = 8
CHUNK_DEADLINE = 90  # seconds; 32 MiB in 90 s = 0.36 MB/s, far below a healthy connection


class _SlowRange(Exception):
    pass


class _CdnUrl:
    """Resolves the HF URL to its signed CDN URL once; refreshes it if it expires (403)."""

    def __init__(self, resolve_url: str):
        self.resolve_url = resolve_url
        self.url = self.refresh()

    def refresh(self) -> str:
        while True:
            resp = httpx.head(self.resolve_url, follow_redirects=True, timeout=60)
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("retry-after", 60)) + 1)
                continue
            resp.raise_for_status()
            self.url = str(resp.url)
            return self.url


def _fetch_range(cdn: _CdnUrl, dest: Path, first: int, last: int, attempts: int = 10) -> int:
    """Writes bytes [first, last] into dest; resumes from the last byte written. Returns retries."""
    pos = first
    for attempt in range(attempts):
        t0 = time.perf_counter()
        try:
            headers = {"Range": f"bytes={pos}-{last}"}
            with httpx.stream("GET", cdn.url, headers=headers, timeout=30) as resp:
                if resp.status_code == 403:
                    cdn.refresh()
                    continue
                if resp.status_code != 206:
                    resp.raise_for_status()
                    raise httpx.TransportError(f"expected 206, got {resp.status_code}")
                with open(dest, "r+b") as fh:
                    fh.seek(pos)
                    for chunk in resp.iter_bytes(1 << 20):
                        fh.write(chunk)
                        pos += len(chunk)
                        if time.perf_counter() - t0 > CHUNK_DEADLINE:
                            raise _SlowRange
            if pos > last:
                return attempt
        except (httpx.TransportError, httpx.HTTPStatusError, _SlowRange):
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"range {first}-{last} failed after {attempts} attempts")


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(8 << 20):
            sha.update(block)
    return sha.hexdigest()


def filter_file(con: duckdb.DuckDBPyConnection, src: Path, rate_bp: int, out: Path) -> dict:
    keep = (
        f"is_blitz(TimeControl) AND "
        f"(sample_bucket(White) < {rate_bp} OR sample_bucket(Black) < {rate_bp})"
    )
    start = time.perf_counter()
    games_seen, blitz, kept = con.execute(f"""
        SELECT count(*),
               count(*) FILTER (WHERE is_blitz(TimeControl)),
               count(*) FILTER (WHERE {keep})
        FROM read_parquet('{src.as_posix()}')
    """).fetchone()
    tmp = out.with_suffix(".tmp")
    con.execute(f"""
        COPY (SELECT {KEEP_COLUMNS} FROM read_parquet('{src.as_posix()}') WHERE {keep})
        TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    tmp.replace(out)  # atomic: a file in OUT_ROOT is always complete
    return {
        "games_seen": games_seen,
        "dropped_not_blitz": games_seen - blitz,
        "dropped_not_sampled": blitz - kept,
        "kept": kept,
        "filter_seconds": round(time.perf_counter() - start, 1),
        "output_bytes": out.stat().st_size,
    }


def run_month(month: str, rate_bp: int, log=print) -> dict:
    out_dir = OUT_ROOT / f"month={month}"
    out_dir.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_r{rate_bp}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "month": month, "rate_bp": rate_bp, "repo": REPO, "revision": REVISION, "files": {},
    }
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false; SET memory_limit = '2GB'; SET threads = 4")
    install_macros(con)
    entries = list_files(month)
    todo = [e for e in entries
            if not (e["path"] in manifest["files"]
                    and (out_dir / (Path(e["path"]).stem + ".parquet")).exists())]

    def start_download(entry: dict) -> Future:
        return pool.submit(download, entry, TMP_DIR / Path(entry["path"]).name)

    with ThreadPoolExecutor(max_workers=1) as pool:
        nxt = start_download(todo[0]) if todo else None
        for i, entry in enumerate(todo):
            dl_info = nxt.result()
            nxt = start_download(todo[i + 1]) if i + 1 < len(todo) else None  # prefetch
            src = TMP_DIR / Path(entry["path"]).name
            out = out_dir / (Path(entry["path"]).stem + ".parquet")
            info = {"sha256": entry["sha256"], **dl_info, **filter_file(con, src, rate_bp, out)}
            src.unlink()
            manifest["files"][entry["path"]] = info
            manifest_path.write_text(json.dumps(manifest, indent=2))
            log(f"{month} {len(manifest['files'])}/{len(entries)}: kept {info['kept']:,}/"
                f"{info['games_seen']:,}  dl {info['download_seconds']}s "
                f"filter {info['filter_seconds']}s")

    keys = ("games_seen", "dropped_not_blitz", "dropped_not_sampled", "kept", "output_bytes")
    manifest["totals"] = {k: sum(f[k] for f in manifest["files"].values()) for k in keys}
    manifest["complete"] = len(manifest["files"]) == len(entries)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest["totals"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("months", nargs="+", help="YYYY-MM ...")
    p.add_argument("--rate-bp", type=int, default=1000)
    args = p.parse_args()
    for month in args.months:
        print(month, json.dumps(run_month(month, args.rate_bp)), flush=True)


if __name__ == "__main__":
    main()
