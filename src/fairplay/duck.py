"""One DuckDB setup for the batch steps on this 16 GB laptop (DECISIONS 2026-10-29).

Background jobs are stopped when the machine runs short of memory, so batch steps use a small
memory limit and spill to disk instead. Row order is never relied on (every query that needs an
order says ORDER BY), so insertion order need not be preserved, which lets DuckDB stream.
"""

import os
from pathlib import Path

import duckdb

TEMP_DIR = Path("private/tmp/duckdb")


def connect(memory_limit: str | None = None, threads: int | None = None
            ) -> duckdb.DuckDBPyConnection:
    """Defaults are safe with a browser open; FAIRPLAY_DUCK_MEMORY / FAIRPLAY_DUCK_THREADS raise
    them when the machine is free (e.g. 4GB / 4)."""
    memory_limit = memory_limit or os.environ.get("FAIRPLAY_DUCK_MEMORY", "2GB")
    threads = threads or int(os.environ.get("FAIRPLAY_DUCK_THREADS", "2"))
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET enable_progress_bar = false; SET memory_limit = '{memory_limit}'; "
                f"SET threads = {threads}; SET preserve_insertion_order = false; "
                f"SET temp_directory = '{TEMP_DIR.as_posix()}'")
    return con


def copy_atomic(con: duckdb.DuckDBPyConnection, query: str, out: Path) -> Path:
    """COPY to a temp file, then rename: a killed run never leaves a half-written output that a
    later step (or a resume check) would mistake for a finished one."""
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con.execute(f"COPY ({query}) TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    os.replace(tmp, out)
    return out
