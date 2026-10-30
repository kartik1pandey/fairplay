"""One local month file -> filtered PGN (.pgn.zst) + manifest JSON with measured counts and speeds.

The raw file is expected at data/raw/<lichess filename> (downloaded by torrent, see source.py).
Pilot (Phase 0a): uv run python -m fairplay.ingest.run_month 2026-06 --rate-bp 1000 --max-mb 500
Full month:       uv run python -m fairplay.ingest.run_month 2026-06 --rate-bp 1000 --delete-raw
"""

import argparse
import io
import json
import time
from dataclasses import asdict
from pathlib import Path

import zstandard

from fairplay.ingest.prefilter import FilterStats, filter_games
from fairplay.ingest.source import HashingReader, fetch_checksums, month_filename

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/interim/filtered")


def run_month(month: str, rate_bp: int, max_mb: int | None, delete_raw: bool) -> dict:
    name = month_filename(month)
    raw_path = RAW_DIR / name
    expected_sha = fetch_checksums()[name]
    total_bytes = raw_path.stat().st_size
    max_bytes = max_mb * 1_000_000 if max_mb else None
    tag = f"{month}_r{rate_bp}" + (f"_pilot{max_mb}mb" if max_mb else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{tag}.partial.pgn.zst"

    stats = FilterStats()
    truncated = False
    start = time.perf_counter()
    with open(raw_path, "rb") as raw_fh, open(out_path, "wb") as out_fh:
        source = HashingReader(raw_fh, max_bytes=max_bytes)
        reader = zstandard.ZstdDecompressor().stream_reader(source, read_size=1 << 20)
        lines = io.BufferedReader(reader, buffer_size=1 << 22)
        writer = zstandard.ZstdCompressor(level=3, threads=-1).stream_writer(out_fh)
        try:
            filter_games(lines, rate_bp, writer.write, stats)
        except zstandard.ZstdError:
            if max_bytes is None:
                raise
            truncated = True  # pilot cut the stream mid-frame; expected
        writer.close()
    seconds = time.perf_counter() - start

    complete = source.offset >= total_bytes
    checksum_ok = source.sha256.hexdigest() == expected_sha if complete else None
    if checksum_ok is False:
        out_path.unlink()  # output of a corrupt input is never kept
        raise SystemExit(f"checksum mismatch for {name}: re-download it")
    final_path = out_path.with_name(f"{tag}.pgn.zst")
    out_path.replace(final_path)
    if delete_raw and checksum_ok:
        raw_path.unlink()

    manifest = {
        "month": month,
        "rate_bp": rate_bp,
        "complete_file": complete,
        "checksum_ok": checksum_ok,
        "expected_sha256": expected_sha,
        "compressed_bytes_read": source.offset,
        "compressed_bytes_total": total_bytes,
        "seconds": round(seconds, 1),
        "mb_per_s_compressed": round(source.offset / 1e6 / seconds, 2),
        "truncated_pilot": truncated,
        "stats": asdict(stats),
        "games_per_s": round(stats.games_seen / seconds),
        "output": str(final_path),
        "output_bytes": final_path.stat().st_size,
        "raw_deleted": delete_raw and bool(checksum_ok),
    }
    if not complete:
        frac = source.offset / total_bytes
        manifest["projected_full_month_hours"] = round(seconds / frac / 3600, 2)
        manifest["projected_full_output_gb"] = round(manifest["output_bytes"] / frac / 1e9, 2)
    (OUT_DIR / f"{tag}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("month", help="YYYY-MM")
    p.add_argument("--rate-bp", type=int, default=1000, help="account sample in basis points")
    p.add_argument("--max-mb", type=int, default=None, help="pilot: stop after N compressed MB")
    p.add_argument("--delete-raw", action="store_true", help="delete raw file if checksum passes")
    args = p.parse_args()
    print(json.dumps(run_month(args.month, args.rate_bp, args.max_mb, args.delete_raw), indent=2))


if __name__ == "__main__":
    main()
