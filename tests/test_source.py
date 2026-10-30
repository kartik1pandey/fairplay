import hashlib
import io

from fairplay.ingest.source import HashingReader


def test_hashing_reader_hashes_everything_read():
    data = b"x" * 5_000_000
    reader = io.BufferedReader(HashingReader(io.BytesIO(data)), buffer_size=1 << 16)
    assert reader.read() == data
    assert reader.raw.sha256.hexdigest() == hashlib.sha256(data).hexdigest()


def test_hashing_reader_respects_byte_limit():
    raw = HashingReader(io.BytesIO(b"abcdefghij"), max_bytes=4)
    assert io.BufferedReader(raw).read() == b"abcd"
    assert raw.offset == 4
