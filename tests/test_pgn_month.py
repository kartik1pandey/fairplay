import io

import httpx
import pyarrow.parquet as pq
import pytest
import zstandard

from fairplay.ingest import pgn_month
from fairplay.ingest.pgn_month import HttpReader
from fairplay.ingest.prefilter import in_sample

GAME = """[Event "Rated Blitz game"]
[Site "https://lichess.org/{gid}"]
[White "{white}"]
[Black "opponent"]
[Result "1-0"]
[UTCDate "2026.10.03"]
[UTCTime "12:00:05"]
[WhiteElo "1500"]
[BlackElo "?"]
[WhiteRatingDiff "+6"]
[BlackRatingDiff "-6"]
[TimeControl "{tc}"]
[Termination "Normal"]

1. e4 {{ [%clk 0:05:00] }} 1... e5 {{ [%clk 0:05:00] }} 2. Qh5 {{ [%clk 0:04:58] }} 1-0

"""


def sampled_name() -> str:
    return next(f"p{i}" for i in range(10_000) if in_sample(f"p{i}".encode(), 1000))


def flaky_server(data: bytes, drop_every: int):
    """Serves ranges, but cuts each response after `drop_every` bytes."""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(data))})
        start = int(request.headers["range"].split("=")[1].rstrip("-"))
        requests.append(start)
        body = data[start:start + drop_every]

        def stream():
            yield body
            if start + drop_every < len(data):
                raise httpx.ReadError("connection reset")

        return httpx.Response(206, content=stream())

    return httpx.MockTransport(handler), requests


def test_http_reader_resumes_at_the_byte_it_stopped():
    data = bytes(range(256)) * 400
    transport, requests = flaky_server(data, drop_every=10_000)
    reader = HttpReader("https://example.test/f", log=lambda m: None, transport=transport)
    assert io.BufferedReader(reader).read() == data
    assert requests == list(range(0, len(data), 10_000))


def test_ingest_keeps_sampled_blitz_games_in_the_hf_schema(tmp_path, monkeypatch):
    name = sampled_name()
    pgn = (GAME.format(gid="aaaaaaaa", white=name, tc="300+0")
           + GAME.format(gid="bbbbbbbb", white=name, tc="60+0")  # bullet: dropped
           + GAME.format(gid="cccccccc", white="zz-not-sampled", tc="300+0"))
    if in_sample(b"zz-not-sampled", 1000) or in_sample(b"opponent", 1000):
        raise AssertionError("fixture names must be outside the sample")
    raw = zstandard.ZstdCompressor().compress(pgn.encode())
    month = "2026-10"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / pgn_month.month_filename(month)).write_bytes(raw)
    import hashlib
    monkeypatch.setattr(pgn_month, "RAW_DIR", raw_dir)
    monkeypatch.setattr(pgn_month, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(pgn_month, "fetch_checksums", lambda: {
        pgn_month.month_filename(month): hashlib.sha256(raw).hexdigest()})
    manifest = pgn_month.ingest(month, 1000, delete_raw=False, log=lambda m: None)
    assert manifest["checksum_verified"] and manifest["totals"] == {
        "games_seen": 3, "dropped_not_blitz": 1, "dropped_not_sampled": 1, "kept": 1}
    t = pq.read_table(tmp_path / "out" / f"month={month}" / "part-00000.parquet").to_pylist()
    assert len(t) == 1
    row = t[0]
    assert row["game_id"] == "aaaaaaaa" and row["WhiteElo"] == 1500 and row["BlackElo"] is None
    assert row["WhiteRatingDiff"] == 6 and str(row["UTCDate"]) == "2026-10-03"
    assert row["movetext"].startswith("1. e4 { [%clk 0:05:00] }") and row["movetext"].endswith(
        "1-0")


def test_checksum_mismatch_deletes_outputs(tmp_path, monkeypatch):
    raw = zstandard.ZstdCompressor().compress(
        GAME.format(gid="aaaaaaaa", white=sampled_name(), tc="300+0").encode())
    month = "2026-10"
    (tmp_path / pgn_month.month_filename(month)).write_bytes(raw)
    monkeypatch.setattr(pgn_month, "RAW_DIR", tmp_path)
    monkeypatch.setattr(pgn_month, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(pgn_month, "fetch_checksums",
                        lambda: {pgn_month.month_filename(month): "0" * 64})
    try:
        pgn_month.ingest(month, 1000, delete_raw=True, log=lambda m: None)
        raise AssertionError("expected a checksum error")
    except RuntimeError:
        pass
    assert not (tmp_path / "out" / f"month={month}").exists()
    assert (tmp_path / pgn_month.month_filename(month)).exists()  # raw kept for inspection


def multi_frame_month(tmp_path, monkeypatch, n_games=60):
    """A raw month of several zstd frames, cut mid-game, like Lichess's files."""
    import hashlib
    name = sampled_name()
    pgn = "".join(GAME.format(gid=f"g{i:07d}", white=name if i % 3 else "zz-not-sampled",
                              tc="300+0" if i % 5 else "60+0") for i in range(n_games)).encode()
    cuts = [0, 333, 2000, 2001, 5555, len(pgn) // 2, len(pgn)]
    raw = b"".join(zstandard.ZstdCompressor().compress(pgn[a:b])
                   for a, b in zip(cuts, cuts[1:], strict=False))
    month = "2026-10"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / pgn_month.month_filename(month)).write_bytes(raw)
    monkeypatch.setattr(pgn_month, "RAW_DIR", raw_dir)
    monkeypatch.setattr(pgn_month, "fetch_checksums", lambda: {
        pgn_month.month_filename(month): hashlib.sha256(raw).hexdigest()})
    monkeypatch.setattr(pgn_month, "PART_GAMES", 5)
    return month


def rows_of(folder):
    return sorted((r["game_id"], r["movetext"]) for f in sorted(folder.glob("part-*.parquet"))
                  for r in pq.read_table(f).to_pylist())


def test_frame_ingest_matches_stream_and_survives_a_kill(tmp_path, monkeypatch):
    month = multi_frame_month(tmp_path, monkeypatch)
    monkeypatch.setattr(pgn_month, "OUT_ROOT", tmp_path / "stream")
    stream = pgn_month.ingest(month, 1000, delete_raw=False, log=lambda m: None)
    expected = rows_of(tmp_path / "stream" / f"month={month}")
    assert len(expected) == stream["totals"]["kept"] > 10

    monkeypatch.setattr(pgn_month, "OUT_ROOT", tmp_path / "frames")
    real_write, calls = pgn_month.write_part, []

    def dies_on_third(con, rows, out):
        calls.append(out)
        if len(calls) == 3:
            raise MemoryError("killed")
        real_write(con, rows, out)

    monkeypatch.setattr(pgn_month, "write_part", dies_on_third)
    with pytest.raises(MemoryError):
        pgn_month.ingest_frames(month, 1000, delete_raw=False, log=lambda m: None)
    monkeypatch.setattr(pgn_month, "write_part", real_write)
    out = pgn_month.ingest_frames(month, 1000, delete_raw=True, log=lambda m: None)
    folder = tmp_path / "frames" / f"month={month}"
    assert rows_of(folder) == expected
    assert out["totals"] == stream["totals"]
    assert not list(folder.glob("_*"))  # checkpoint files removed when complete
    assert not (pgn_month.RAW_DIR / pgn_month.month_filename(month)).exists()


def test_download_resumes_from_the_part_file(tmp_path, monkeypatch):
    data = bytes(range(256)) * 300
    transport, requests = flaky_server(data, drop_every=7_000)
    monkeypatch.setattr(pgn_month, "RAW_DIR", tmp_path)
    name = pgn_month.month_filename("2026-10")
    (tmp_path / (name + ".part")).write_bytes(data[:30_000])  # an earlier, killed session
    path = pgn_month.download("2026-10", log=lambda m: None, transport=transport)
    assert path.read_bytes() == data and requests[0] == 30_000
