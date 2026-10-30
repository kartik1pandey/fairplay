"""Lichess monthly files: names, published checksums, and a reader that hashes while it reads.

Measured on 2026-10-28: database.lichess.org serves ~0.08 MB/s per HTTP connection (general
internet here: 17 MB/s), i.e. ~4 days per month over HTTP. Files are therefore downloaded by
torrent (Lichess publishes a .torrent per month) and processed from local disk. Hashing during
the single processing pass means the 28 GB file is read once, not twice.
"""

import hashlib
import io
from typing import BinaryIO

import httpx

BASE_URL = "https://database.lichess.org/standard/"


def month_filename(month: str) -> str:
    return f"lichess_db_standard_rated_{month}.pgn.zst"


def torrent_url(month: str) -> str:
    return BASE_URL + month_filename(month) + ".torrent"


def fetch_checksums() -> dict[str, str]:
    """filename -> sha256 hex, from Lichess's published sha256sums.txt."""
    text = httpx.get(BASE_URL + "sha256sums.txt", timeout=60).raise_for_status().text
    return {name: digest for digest, name in (line.split() for line in text.splitlines() if line)}


class HashingReader(io.RawIOBase):
    """Wraps a binary file; every byte read is fed into sha256. Optional byte limit for pilots."""

    def __init__(self, raw: BinaryIO, max_bytes: int | None = None):
        self.raw = raw
        self.max_bytes = max_bytes
        self.offset = 0
        self.sha256 = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        want = len(b)
        if self.max_bytes is not None:
            want = min(want, self.max_bytes - self.offset)
        if want <= 0:
            return 0
        data = self.raw.read(want)
        n = len(data)
        b[:n] = data
        self.offset += n
        self.sha256.update(data)
        return n
