"""Assemble a Hugging Face Space (space_build/) from the package and the demo bundle.

Two forms, same analysis code and bundle (aggregate-only, checked again here; no data/,
private/ or runs/ file is copied):
- static (default, free tier): index.html + app.zip (the fairplay package and demo_bundle/).
  The page runs the Python in the visitor's browser with Pyodide (demo/browser.py). Pyodide
  314.0.7 ships duckdb 1.5.1, lightgbm 4.6.0 and pyarrow 22; the demo tests pass on those
  versions, and scikit-learn is not loaded (DECISIONS 2026-11-02).
- gradio (--gradio): app.py + requirements.txt for a Gradio Space (Hugging Face requires a paid
  plan for these since late 2026).

uv run python -m fairplay.demo.make_space
hf upload <your-user>/fairplay-demo space_build . --repo-type space
"""

import argparse
import importlib.metadata as md
import shutil
import zipfile
from pathlib import Path

from fairplay.demo.bundle import check_aggregate_only
from fairplay.demo.peergrid import OUT as BUNDLE

BUILD = Path("space_build")
PACKAGES = ["gradio", "duckdb", "pyarrow", "numpy", "pydantic", "httpx", "chess", "lightgbm",
            "scikit-learn"]
PAGE = Path(__file__).parent / "static" / "index.html"
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "static")

HEADER = """---
title: FairPlay research demo
emoji: ♟️
colorFrom: gray
colorTo: blue
{sdk}
pinned: false
license: other  # no licence chosen yet for the repo
short_description: Moderator-flag prioritisation for Lichess blitz (research)
---

Research prototype. A score is a prioritisation signal, not evidence of wrongdoing.

Code and full evaluation: https://github.com/kartik1pandey/fairplay
"""

GRADIO_SDK = """sdk: gradio
sdk_version: {gradio}
python_version: "3.11"
app_file: app.py"""

APP = """from fairplay.demo.space_app import build

build().launch()
"""


def fresh(out: Path) -> None:
    check_aggregate_only(BUNDLE)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()


def build_static(out: Path = BUILD) -> Path:
    fresh(out)
    staging = out / "_app"
    shutil.copytree("src/fairplay", staging / "fairplay", ignore=SKIP)
    shutil.copytree(BUNDLE, staging / BUNDLE.name)
    with zipfile.ZipFile(out / "app.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(staging).as_posix())
    shutil.rmtree(staging)
    shutil.copy(PAGE, out / "index.html")
    (out / "README.md").write_text(HEADER.format(sdk="sdk: static"), encoding="utf-8")
    return out


def build(out: Path = BUILD) -> Path:
    fresh(out)
    shutil.copytree("src/fairplay", out / "fairplay", ignore=SKIP)
    shutil.copytree(BUNDLE, out / BUNDLE.name)
    (out / "app.py").write_text(APP, encoding="utf-8")
    (out / "requirements.txt").write_text(
        "".join(f"{p}=={md.version(p)}\n" for p in PACKAGES if p != "gradio"), encoding="utf-8")
    sdk = GRADIO_SDK.format(gradio=md.version("gradio"))
    (out / "README.md").write_text(HEADER.format(sdk=sdk), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gradio", action="store_true", help="Gradio Space (paid HF plan)")
    out = build() if p.parse_args().gradio else build_static()
    print(out, sum(f.stat().st_size for f in out.rglob("*") if f.is_file()), "bytes")
