"""Official Lichess monthly PGN file -> sampled blitz games as Parquet, in the Hugging Face schema.

Why this exists (DECISIONS.md 2026-11-02): Experiment P needs October 2026, and the Hugging Face
mirror stops at September 2025. Measured 2026-11-02: database.lichess.org serves 1.9-4.9 MB/s on
one connection (it was ~0.08 MB/s on 2026-10-28); the filter pass runs ~5.5 MB/s.

Two paths, same output (the columns hf_month writes, so clean.py runs unchanged):
- download -> ingest (default; needs disk for the raw month). Every step survives being killed
  for low memory: the download appends to a .part file and resumes by a Range request; the
  checksum is checked on the stored file; the ingest checkpoints at zstd frame boundaries
  (Lichess files are ~4.7 MB frames) with the unfinished game carried over.
- stream (when the disk is too small): one pass straight from HTTP that hashes, decompresses
  and filters; all-or-nothing, and a checksum mismatch deletes every part.
Kept games: blitz with a sampled player (prefilter, the same rule as hf_month's macros), in
parts of PART_GAMES under private/games_sampled/month=M/.

uv run python -m fairplay.ingest.pgn_month download 2026-10
uv run python -m fairplay.ingest.pgn_month ingest 2026-10 --rate-bp 1000 --delete-raw
uv run python -m fairplay.ingest.pgn_month stream 2026-09 --max-mb 300   # pilot, no checksum
"""

import argparse
import hashlib
import io
import json
import shutil
import time
from pathlib import Path

import duckdb
import httpx
import pyarrow as pa
import zstandard

from fairplay.ingest.hf_month import OUT_ROOT
from fairplay.ingest.pgn_stream import Headers, iter_games
from fairplay.ingest.prefilter import FilterStats, keep_game
from fairplay.ingest.source import BASE_URL, HashingReader, fetch_checksums, month_filename

RAW_DIR = Path("data/raw")
PART_GAMES = 125_000  # about the size of one Hugging Face input file, so cleaning memory matches
STRING_COLUMNS = ("Event", "White", "Black", "WhiteTitle", "BlackTitle", "Result",
                  "WhiteElo", "BlackElo", "WhiteRatingDiff", "BlackRatingDiff", "UTCDate",
                  "UTCTime", "Termination", "TimeControl")

# Casts to the Hugging Face column types; "?" (unknown rating) becomes NULL, as it is there.
TYPED = """
    regexp_extract(Site, '[^/]+$') AS game_id, Event, White, Black,
    nullif(WhiteTitle, '') AS WhiteTitle, nullif(BlackTitle, '') AS BlackTitle, Result,
    TRY_CAST(WhiteElo AS SMALLINT) AS WhiteElo, TRY_CAST(BlackElo AS SMALLINT) AS BlackElo,
    TRY_CAST(WhiteRatingDiff AS SMALLINT) AS WhiteRatingDiff,
    TRY_CAST(BlackRatingDiff AS SMALLINT) AS BlackRatingDiff,
    strptime(UTCDate, '%Y.%m.%d')::DATE AS UTCDate,
    (UTCTime || '+00')::TIMETZ AS UTCTime,
    Termination, TimeControl, movetext
"""


class HttpReader(io.RawIOBase):
    """The remote file as a forward-only stream on one connection. A dropped connection is
    reopened with a Range request at the current byte, so the decompressor never notices."""

    def __init__(self, url: str, log=print, attempts: int = 50, transport=None, start: int = 0):
        self.url, self.log, self.attempts = url, log, attempts
        self.transport = transport  # tests inject a mock server
        with httpx.Client(timeout=60, transport=transport) as c:
            self.size = int(c.head(url).raise_for_status().headers["content-length"])
        self.offset = start
        self._resp = self._chunks = None
        self._buf = b""

    def readable(self) -> bool:
        return True

    def _open(self) -> None:
        self.close_response()
        req = httpx.Request("GET", self.url, headers={"Range": f"bytes={self.offset}-"})
        self._client = httpx.Client(timeout=60, transport=self.transport)
        self._resp = self._client.send(req, stream=True)
        if self._resp.status_code != 206:
            raise httpx.TransportError(f"expected 206, got {self._resp.status_code}")
        self._chunks = self._resp.iter_raw()  # no chunk_size: a buffer would be lost on a drop

    def close_response(self) -> None:
        if self._resp is not None:
            self._resp.close()
            self._client.close()
        self._resp = self._chunks = None

    def readinto(self, b) -> int:
        if self.offset >= self.size and not self._buf:
            return 0
        for attempt in range(self.attempts):
            try:
                if not self._buf:
                    if self._chunks is None:
                        self._open()
                    self._buf = next(self._chunks, b"")
                    if not self._buf:  # server closed early: reopen at this offset
                        self.close_response()
                        continue
                n = min(len(b), len(self._buf))
                b[:n] = self._buf[:n]
                self._buf = self._buf[n:]
                self.offset += n
                return n
            except httpx.HTTPError as e:
                self.log(f"connection dropped at {self.offset / 1e9:.2f} GB ({e!r}); resuming")
                self.close_response()
                time.sleep(min(60, 2 ** min(attempt, 6)) if self.transport is None else 0)
        raise RuntimeError(f"{self.url}: gave up at byte {self.offset}")


def game_row(headers: Headers, raw: list[bytes]) -> dict:
    row = {c: headers.get(c.encode(), b"").decode("utf-8", "replace") for c in STRING_COLUMNS}
    row["Site"] = headers.get(b"Site", b"").decode()
    row["movetext"] = " ".join(line.decode("utf-8", "replace").strip()
                               for line in raw if line.strip() and not line.startswith(b"["))
    return row


def write_part(con: duckdb.DuckDBPyConnection, rows: list[dict], out: Path) -> None:
    con.register("part_rows", pa.Table.from_pylist(rows))
    tmp = out.with_suffix(".tmp")
    con.execute(f"COPY (SELECT {TYPED} FROM part_rows) TO '{tmp.as_posix()}' "
                "(FORMAT parquet, COMPRESSION zstd)")
    con.unregister("part_rows")
    tmp.replace(out)


def ingest(month: str, rate_bp: int, delete_raw: bool, max_mb: int | None = None,
           log=print) -> dict:
    name = month_filename(month)
    raw_path = RAW_DIR / name
    expected = fetch_checksums()[name]
    out_dir = OUT_ROOT / f"month={month}"
    shutil.rmtree(out_dir, ignore_errors=True)  # a pass is all-or-nothing: no stale parts
    out_dir.mkdir(parents=True)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false; SET memory_limit = '2GB'; SET threads = 2")
    stats, rows, parts = FilterStats(), [], 0
    t0 = time.perf_counter()
    remote = not raw_path.exists()
    with (HttpReader(BASE_URL + name, log) if remote else open(raw_path, "rb")) as fh:
        hashing = HashingReader(fh, max_mb * 1_000_000 if max_mb else None)
        text = zstandard.ZstdDecompressor().stream_reader(io.BufferedReader(hashing, 8 << 20),
                                                          read_across_frames=True)
        try:
            for headers, raw in iter_games(io.BufferedReader(text, 8 << 20)):
                if keep_game(headers, rate_bp, stats):
                    rows.append(game_row(headers, raw))
                    if len(rows) == PART_GAMES:
                        write_part(con, rows, out_dir / f"part-{parts:05d}.parquet")
                        parts, rows = parts + 1, []
                        log(f"{month}: {stats.games_seen:,} games read, {stats.kept:,} kept, "
                            f"{hashing.offset / 1e9:.1f} GB")
        except zstandard.ZstdError:
            if max_mb is None:  # a truncated read is expected only in a pilot
                raise
    if rows:
        write_part(con, rows, out_dir / f"part-{parts:05d}.parquet")
        parts += 1
    seconds = time.perf_counter() - t0
    checksum_ok = max_mb is None and hashing.sha256.hexdigest() == expected
    if max_mb is None and not checksum_ok:
        shutil.rmtree(out_dir)
        raise RuntimeError(f"{name}: SHA256 mismatch; outputs deleted, raw file kept")
    manifest = {"month": month, "rate_bp": rate_bp, "source": BASE_URL + name,
                "streamed_from_http": remote,
                "sha256": expected, "checksum_verified": checksum_ok, "pilot_max_mb": max_mb,
                "bytes_read": hashing.offset, "seconds": round(seconds),
                "mb_per_second": round(hashing.offset / 1e6 / seconds, 1), "parts": parts,
                "totals": {"games_seen": stats.games_seen,
                           "dropped_not_blitz": stats.dropped_not_blitz,
                           "dropped_not_sampled": stats.dropped_not_sampled,
                           "kept": stats.kept},
                "complete": checksum_ok}
    (out_dir / f"manifest_r{rate_bp}.json").write_text(json.dumps(manifest, indent=2))
    if delete_raw and checksum_ok:
        raw_path.unlink()
    return manifest


# --- Resumable path: download -> verify -> ingest frame by frame -------------------------------
# Used when the disk has room for the raw month. Every step survives being killed: the download
# appends to a .part file, the checksum pass is cheap to redo, and the ingest checkpoints at zstd
# frame boundaries (Lichess files are a sequence of ~4.7 MB frames, measured 2026-11-02).

DISK_MARGIN = 2_000_000_000  # bytes kept free beyond the raw file, for the Parquet parts


def download(month: str, log=print, transport=None) -> Path:
    name = month_filename(month)
    final, part = RAW_DIR / name, RAW_DIR / (name + ".part")
    if final.exists():
        return final
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    reader = HttpReader(BASE_URL + name, log, transport=transport, start=have)
    free = shutil.disk_usage(RAW_DIR).free
    if reader.size - have + DISK_MARGIN > free:
        raise SystemExit(f"{name}: needs {(reader.size - have + DISK_MARGIN) / 1e9:.1f} GB, "
                         f"{free / 1e9:.1f} GB free; use `stream` instead")
    t0, last = time.perf_counter(), 0.0
    with open(part, "ab") as fh:
        while chunk := reader.read(8 << 20):
            fh.write(chunk)
            if time.perf_counter() - last > 300:
                last = time.perf_counter()
                fh.flush()
                log(f"{name}: {reader.offset / 1e9:.2f}/{reader.size / 1e9:.2f} GB")
    if part.stat().st_size != reader.size:
        raise RuntimeError(f"{name}: size {part.stat().st_size} != {reader.size}")
    part.replace(final)
    log(f"{name}: downloaded in {(time.perf_counter() - t0) / 3600:.1f} h this session")
    return final


def verify(month: str) -> bool:
    """SHA256 of the raw file vs the published one; the result is cached next to the file."""
    name = month_filename(month)
    raw, ok = RAW_DIR / name, RAW_DIR / (name + ".sha256-ok")
    expected = fetch_checksums()[name]
    if ok.exists() and ok.read_text().strip() == expected:
        return True
    sha = hashlib.sha256()
    with open(raw, "rb") as fh:
        while block := fh.read(16 << 20):
            sha.update(block)
    if sha.hexdigest() != expected:
        return False
    ok.write_text(expected)
    return True


def frames(fh, offset: int):
    """Yield (offset after the frame, decompressed bytes) for each zstd frame from `offset`."""
    fh.seek(offset)
    pending = b""
    while True:
        d = zstandard.ZstdDecompressor().decompressobj()
        out, fed = [], 0
        data = pending or fh.read(8 << 20)
        if not data:
            return
        while True:
            out.append(d.decompress(data))
            fed += len(data)
            if d.eof:
                break
            data = fh.read(8 << 20)
            if not data:
                raise RuntimeError(f"truncated zstd frame at byte {offset}")
        pending = d.unused_data
        offset += fed - len(pending)
        yield offset, b"".join(out)


def split_complete(text: bytes) -> tuple[bytes, bytes]:
    """Text -> (complete games, the start of the next game, carried to the next frame)."""
    cut = text.rfind(b"\n\n[Event ")
    return (b"", text) if cut < 0 else (text[:cut + 2], text[cut + 2:])


def ingest_frames(month: str, rate_bp: int, delete_raw: bool, log=print) -> dict:
    name = month_filename(month)
    raw = RAW_DIR / name
    if not verify(month):
        raise RuntimeError(f"{name}: SHA256 mismatch; download it again")
    out_dir = OUT_ROOT / f"month={month}"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "_state.json"
    carry_of = lambda offset: out_dir / f"_carry-{offset}.bin"  # noqa: E731
    if state_path.exists():  # resume at the last checkpoint; parts after it are rewritten
        state = json.loads(state_path.read_text())
        carry = carry_of(state["offset"]).read_bytes()
        log(f"{month}: resuming at byte {state['offset']:,}, part {state['parts']}")
    else:
        state = {"offset": 0, "parts": 0, "stats": vars(FilterStats())}
        carry = b""
    stats = FilterStats(**state["stats"])
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false; SET memory_limit = '2GB'; SET threads = 2")
    rows: list[dict] = []
    t0 = time.perf_counter()

    def checkpoint(offset: int, carry: bytes) -> None:
        nonlocal rows
        if rows:
            write_part(con, rows, out_dir / f"part-{state['parts']:05d}.parquet")
            state["parts"] += 1
            rows = []
        old = state["offset"]
        state.update(offset=offset, stats=vars(stats))
        carry_of(offset).write_bytes(carry)  # before the state that points at it
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(state_path)
        if old != offset:
            carry_of(old).unlink(missing_ok=True)

    def take(text: bytes) -> None:
        for headers, game in iter_games(io.BytesIO(text)):
            if keep_game(headers, rate_bp, stats):
                rows.append(game_row(headers, game))

    with open(raw, "rb") as fh:
        for offset, text in frames(fh, state["offset"]):
            complete, carry = split_complete(carry + text)
            take(complete)
            if len(rows) >= PART_GAMES:
                checkpoint(offset, carry)
                log(f"{month}: {stats.games_seen:,} games read, {stats.kept:,} kept, "
                    f"{offset / 1e9:.1f} GB")
    take(carry)
    checkpoint(raw.stat().st_size, b"")
    manifest = {"month": month, "rate_bp": rate_bp, "source": BASE_URL + name,
                "sha256": fetch_checksums()[name], "checksum_verified": True,
                "resumable_frames": True, "seconds_this_session": round(time.perf_counter() - t0),
                "parts": state["parts"], "totals": vars(stats), "complete": True}
    (out_dir / f"manifest_r{rate_bp}.json").write_text(json.dumps(manifest, indent=2))
    state_path.unlink()
    carry_of(state["offset"]).unlink()
    if delete_raw:
        raw.unlink()
        (RAW_DIR / (name + ".sha256-ok")).unlink(missing_ok=True)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("stream", "download", "ingest"):
        c = sub.add_parser(cmd)
        c.add_argument("month")
        if cmd != "download":
            c.add_argument("--rate-bp", type=int, default=1000)
            c.add_argument("--delete-raw", action="store_true")
    sub.choices["stream"].add_argument("--max-mb", type=int,
                                       help="pilot: read only the first N MB (no checksum)")
    args = p.parse_args()
    log = lambda m: print(m, flush=True)  # noqa: E731
    if args.cmd == "download":
        out = {"path": str(download(args.month, log))}
    elif args.cmd == "stream":
        out = ingest(args.month, args.rate_bp, args.delete_raw, args.max_mb, log)
    else:
        out = ingest_frames(args.month, args.rate_bp, args.delete_raw, log)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
