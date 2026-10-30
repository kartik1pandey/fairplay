"""One-command reproduction (design §12): the two targets, as ordered steps with a cache.

  uv run python -m fairplay.pipeline reproduce-synthetic   no API, no labels: data + synthetic
  uv run python -m fairplay.pipeline reproduce             + real-label training and evaluation
  uv run python -m fairplay.pipeline reproduce --dry-run   print the plan and expected runtime

A step is skipped when its outputs exist and the fingerprint of its inputs and of its own source
module is unchanged since the last successful run. Fingerprint = (path, size, mtime) of every
input file + sha256 of the step's module: content-hashing 14 GB of Parquet on every run would
cost more than most steps. A changed input or a changed step reruns that step and everything
after it. Durations of the last successful run are printed as the expected runtime.

Steps that call the Lichess API (labels) are never run by this script: a label snapshot is an
input you fetch once yourself (design §12: "make reproduce ... from a label snapshot on disk").
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

STATE = Path("runs/pipeline_state.json")
MONTHS = ["2025-06", "2025-07", "2025-08"]  # Sept dropped (DECISIONS 2026-10-29)
P = "data/processed"


@dataclass(frozen=True)
class Step:
    name: str
    module: str
    args: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()  # globs
    outputs: tuple[str, ...] = ()
    needs_labels: bool = False
    after: tuple[str, ...] = field(default=())


STEPS = [
    Step("ingest", "fairplay.ingest.hf_month", (*MONTHS, "--rate-bp", "1000"),
         outputs=tuple(f"private/games_sampled/month={m}/manifest_r1000.json" for m in MONTHS)),
    Step("clean", "fairplay.ingest.clean", (*MONTHS, "--rate-bp", "1000"),
         ("private/games_sampled/month=*/*.parquet",),
         tuple(f"{P}/clean_manifest_{m}.json" for m in MONTHS)),
    Step("validate_parser", "fairplay.ingest.validate_parser", ("2025-08",),
         (f"{P}/game_moves/month=2025-08.parquet",), (f"{P}/parser_validation_2025-08.json",)),
    Step("game_features", "fairplay.features.game_features", tuple(MONTHS),
         (f"{P}/player_games/*.parquet",), tuple(f"{P}/game_features/month={m}.parquet"
                                                  for m in MONTHS)),
    Step("eligibility", "fairplay.dataset.eligibility", (), (f"{P}/player_games/*.parquet",),
         (f"{P}/eligible_rows.parquet",)),
    Step("account_features", "fairplay.features.account_features", (),
         (f"{P}/game_features/*.parquet", f"{P}/eligible_rows.parquet"),
         (f"{P}/features_tier1.parquet",)),
    Step("assemble", "fairplay.dataset.assemble", ("--snapshot", "S0"),
         (f"{P}/features_tier1.parquet", "private/labels.sqlite"),
         (f"{P}/model_table_S0.parquet",), needs_labels=True),
    Step("baselines", "fairplay.models.run_baselines", (), (f"{P}/model_table_S0.parquet",),
         needs_labels=True),
    Step("tier1", "fairplay.models.run_tier1", (), (f"{P}/model_table_S0.parquet",),
         needs_labels=True),
    Step("fairness", "fairplay.fairness.run_fairness", (), (f"{P}/model_table_S0.parquet",),
         needs_labels=True),
    Step("final", "fairplay.models.run_final", (), (f"{P}/model_table_S0.parquet",),
         needs_labels=True),
    Step("report", "fairplay.report", ("build",), ("runs/*/metrics.json",),
         ("results/results.md",)),
]
TARGETS = {"reproduce-synthetic": [s for s in STEPS if not s.needs_labels],
           "reproduce": STEPS}


def module_hash(module: str) -> str:
    spec = importlib.util.find_spec(module)
    return hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()[:16] if spec else "missing"


def fingerprint(step: Step) -> str:
    files = sorted({f for g in step.inputs for f in Path().glob(g) if f.is_file()})
    stats = [(f.as_posix(), f.stat().st_size, int(f.stat().st_mtime)) for f in files]
    blob = json.dumps({"inputs": stats, "module": module_hash(step.module), "args": step.args})
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def plan(target: str, state: dict) -> list[tuple[Step, str, str]]:
    """[(step, 'run' | 'cached', reason)]; once one step reruns, every later step reruns."""
    out, dirty = [], False
    for s in TARGETS[target]:
        fp = fingerprint(s)
        prev = state.get(s.name, {})
        missing = [o for o in s.outputs if not Path(o).exists()]
        if dirty or missing or prev.get("fingerprint") != fp or not s.outputs:
            reason = ("upstream changed" if dirty else f"missing {missing[0]}" if missing
                      else "no outputs declared" if not s.outputs else "inputs or code changed")
            out.append((s, "run", reason))
            dirty = True
        else:
            out.append((s, "cached", f"unchanged since {prev.get('finished')}"))
    return out


def run(target: str, dry_run: bool = False) -> None:
    state = load_state()
    steps = plan(target, state)
    expected = sum(state.get(s.name, {}).get("seconds", 0) for s, act, _ in steps if act == "run")
    print(f"{target}: {sum(a == 'run' for _, a, _ in steps)} step(s) to run, expected "
          f"~{expected / 60:.0f} min (last measured durations; unmeasured steps count 0)")
    for s, act, why in steps:
        print(f"  {act:6} {s.name:18} {why}")
    if dry_run:
        return
    if target == "reproduce" and not Path("private/labels.sqlite").exists():
        sys.exit("reproduce needs a label snapshot on disk: fetch it with fairplay.labels.cli")
    for s, act, _ in steps:
        if act == "cached":
            continue
        t0 = time.time()
        subprocess.run([sys.executable, "-m", s.module, *s.args], check=True)
        state[s.name] = {"fingerprint": fingerprint(s), "seconds": round(time.time() - t0),
                         "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def adopt(target: str) -> list[str]:
    """Record steps that were run by hand (outputs present) as done with their current
    fingerprint, so the pipeline does not redo them. Stops at the first step without outputs."""
    state, adopted = load_state(), []
    for s in TARGETS[target]:
        if not s.outputs or not all(Path(o).exists() for o in s.outputs):
            break
        state[s.name] = {"fingerprint": fingerprint(s), "seconds": state.get(s.name, {})
                         .get("seconds", 0), "finished": "adopted (run by hand)"}
        adopted.append(s.name)
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return adopted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("target", choices=list(TARGETS))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--adopt", action="store_true", help="mark hand-run steps with outputs as done")
    args = p.parse_args()
    if args.adopt:
        print("adopted:", adopt(args.target))
        return
    run(args.target, args.dry_run)


if __name__ == "__main__":
    main()
