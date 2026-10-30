"""Run logging and the evaluation-budget guard (design §0 item 8, §4, §12).

- Every run writes runs/<run_id>/ with config, metrics, git commit (or "uncommitted"), and the
  SHA256 of the input table. Runs are never overwritten.
- Validation (August) may be used for at most 8 logged evaluations; each use is appended to
  runs/evaluation_log.jsonl. A 9th raises. The log is append-only and git-ignored with runs/.
- Test (September) may be scored only with final=True, once. A second final run must state a
  reason (e.g. a bug fix), and then results before and after are both kept (design §4).
"""

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

RUNS = Path("runs")
EVAL_LOG = RUNS / "evaluation_log.jsonl"
VALIDATION_BUDGET = 8


class BudgetExceeded(Exception):
    pass


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def _log_entries(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def claim_evaluation(split: str, what: str, final: bool = False, reason: str = "",
                     log: Path = EVAL_LOG, diagnostic: bool = False) -> dict:
    """Call BEFORE scoring a held-out split. Raises if the budget forbids it.

    diagnostic=True: a pre-registered, parameter-free check on already-frozen models (e.g. the
    fairness audit). Recorded for transparency, not counted against the validation budget.
    """
    entries = _log_entries(log)
    if split == "validation" and not diagnostic:
        used = sum(e["split"] == "validation" and not e.get("diagnostic") for e in entries)
        if used >= VALIDATION_BUDGET:
            raise BudgetExceeded(f"validation used {used}/{VALIDATION_BUDGET} times")
    elif split == "test":
        if not final:
            raise BudgetExceeded("test split is locked; pass final=True for the final run")
        if any(e["split"] == "test" for e in entries) and not reason:
            raise BudgetExceeded("test already scored once; a re-run needs a stated reason")
    entry = {"time": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "split": split,
             "what": what, "final": final, "reason": reason, "diagnostic": diagnostic,
             "commit": git_commit()}
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def new_run(name: str, config: dict, inputs: list[Path], root: Path = RUNS) -> Path:
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + f"_{name}"
    d = root / run_id
    d.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": run_id, "commit": git_commit(), "config": config,
                "inputs": {str(p): file_sha256(p) for p in inputs}}
    (d / "config.json").write_text(json.dumps(manifest, indent=2, default=str))
    return d


def save_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
