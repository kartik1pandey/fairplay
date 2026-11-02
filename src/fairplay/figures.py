"""Figures for the docs, deck and README. Aggregates only: every number comes from a run's
metrics.json (or data/tier2/decision.json, which holds no account rows).

uv run --group docs python -m fairplay.figures
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from fairplay.runs import RUNS  # noqa: E402

OUT = Path("assets/figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b4b3ad"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def metrics(name_suffix: str) -> dict:
    run = sorted(p for p in RUNS.glob(f"*_{name_suffix}") if (p / "metrics.json").exists())[-1]
    return json.loads((run / "metrics.json").read_text(encoding="utf-8"))


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def precision(final: dict) -> Path:
    s = final["all_accounts"]["summary"]
    names = [("rule", "Rule baseline\n(performance-rating gap)"),
             ("logreg", "Logistic regression"), ("lgbm", "LightGBM (tier 1)")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (key, _label) in enumerate(names):
        fp = s[key]["flag_precision"]
        lo, hi = fp["ci95"]
        color = BLUE if key == "lgbm" else GRAY
        ax.barh(i, fp["value"], height=0.55, color=color)
        ax.errorbar(fp["value"], i, xerr=[[fp["value"] - lo], [hi - fp["value"]]], fmt="none",
                    ecolor=INK, elinewidth=1.4, capsize=4)
        ax.text(hi + 0.008, i, f"{fp['value']:.3f}  ({s[key]['lift']['value']:.1f}× base)",
                va="center", color=INK, fontsize=10.5)
    base = s["lgbm"]["base_rate"]["value"]
    ax.axvline(base, color=ORANGE, lw=2)
    ax.text(base + 0.006, -0.42, f"background flag rate {base:.3f}", color=INK2, fontsize=9.5)
    ax.set_yticks(range(3), [n[1] for n in names])
    ax.set_xlim(0, 0.55)
    ax.set_ylim(-0.6, 2.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Flag-precision@k: share of each weekly queue that is S0-flagged "
                  "(k = 0.5% of eligible)")
    ax.set_title("August test month, scored once: 95% cluster-bootstrap intervals")
    return save(fig, "01_flag_precision")


def fairness(final: dict) -> Path:
    s = final["fairness"]["lgbm"]["capped"]["selection_rate_among_unflagged"]
    groups = [k for k in s if k != "overall"]
    groups.sort(key=lambda g: s[g]["ratio_to_overall"])
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for i, g in enumerate(groups):
        r = s[g]["ratio_to_overall"]
        fails = s[g]["fails_2x"]
        ax.barh(i, r, height=0.6, color=ORANGE if fails else GRAY)
        ax.text(r + 0.06, i, f"{r:.2f}×" + ("  above 2× limit" if fails else ""), va="center",
                fontsize=9.5, color=INK)
    ax.axvline(1, color=INK2, lw=1.2, ls=":")
    ax.axvline(2, color=ORANGE, lw=1.6)
    ax.text(2.04, len(groups) - 0.6, "audit limit 2×", color=INK2, fontsize=9.5)
    ax.set_yticks(range(len(groups)), [g.replace(":", ": ").replace("_", " ") for g in groups],
                  fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Selection rate among unflagged accounts ÷ overall rate (LightGBM, 2200+ cap)")
    ax.set_title("Who gets queued without a flag? (a proxy: unflagged ≠ honest)")
    return save(fig, "02_fairness_audit")


def tier2_cost() -> Path | None:
    path = Path("data/tier2/decision.json")
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    rows = [r for r in d["table"] if r["nodes"] == 100_000]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = range(len(rows))
    cand = [r["cpu_hours_candidates_only"] for r in rows]
    total = [r["cpu_hours_per_cutoff"] for r in rows]
    ax.bar(x, cand, width=0.55, color=BLUE, label="candidates (top q by tier 1)")
    ax.bar(x, [t - c for t, c in zip(total, cand, strict=True)], width=0.55, bottom=cand,
           color=AQUA, label="2% random audit of non-candidates", edgecolor=SURFACE, lw=2)
    for i, t in enumerate(total):
        ax.text(i, t + 1, f"{t:.1f} h", ha="center", fontsize=10)
    ax.axhline(d["budget_cpu_hours_per_cutoff"], color=ORANGE, lw=2)
    ax.text(len(rows) - 0.55, d["budget_cpu_hours_per_cutoff"] + 1.2, "budget 4 CPU-h",
            color=INK2, fontsize=9.5, ha="right")
    ax.set_xticks(list(x), [f"q = {r['q']:.0%}" for r in rows])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("CPU-hours per weekly cutoff")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    tag = " (provisional: measured under load)" if d.get("provisional") else ""
    ax.set_title(f"Tier 2 engine cost at 100k nodes{tag}")
    return save(fig, "03_tier2_cost")


def monitoring() -> Path:
    m = metrics("monitoring_replay")
    weeks = m["weeks"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    xs = range(len(weeks))
    ax.plot(xs, [w["score_psi"] for w in weeks], color=BLUE, lw=2, marker="o", ms=6,
            label="score PSI vs July")
    ax.axhline(m["psi_alert"], color=ORANGE, lw=1.6)
    ax.text(0, m["psi_alert"] - 0.02, "alert line 0.2 → investigation note", color=INK2,
            fontsize=9.5, va="top")
    ax.set_ylim(0, 0.25)
    ax.set_xticks(list(xs), [w["cutoff"][5:] for w in weeks], fontsize=9.5)
    ax.axvspan(3.5, len(weeks) - 0.5, color=GRID, alpha=0.5, lw=0)
    ax.text(3.6, 0.235, "August (labels not read)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Population stability index")
    ax.set_title("Weekly monitoring replay: score drift stays near zero")
    return save(fig, "04_monitoring_psi")


def calibration(final: dict) -> Path:
    curve = final["calibration"]["curve"]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    hi = max(max(c["mean_predicted"], c["observed_flag_share"]) for c in curve) * 1.1
    ax.plot([0, hi], [0, hi], color=GRAY, lw=1.2, ls="--", label="perfect calibration")
    ax.plot([c["mean_predicted"] for c in curve], [c["observed_flag_share"] for c in curve],
            color=BLUE, lw=2, marker="o", ms=7, label="LightGBM + isotonic")
    ax.set_xlabel("Mean predicted P(flag at S0)")
    ax.set_ylabel("Observed flag share")
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title(f"Reliability on August (Brier {final['calibration']['brier']:.3f})")
    return save(fig, "05_calibration")


def synthetic() -> Path | None:
    m = metrics("synthetic_score")
    if "rule" not in m:
        return None
    names = {"A": "A: blunt\n50% of moves", "B": "B: moderate\n20% of moves",
             "C": "C: careful (held out)\n10%, critical only"}
    strategies = [s for s in names if s in m["auc_vs_control"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.36
    for j, (label, auc, color) in enumerate([("LightGBM", m["auc_vs_control"], BLUE),
                                             ("Rule baseline", m["rule"]["auc_vs_control"],
                                              GRAY)]):
        xs = [i + (j - 0.5) * w for i in range(len(strategies))]
        vals = [auc[s] for s in strategies]
        ax.bar(xs, vals, width=w - 0.04, color=color, label=label)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(0.5, color=ORANGE, lw=1.6)
    ax.text(len(strategies) - 0.5, 0.52, "chance", color=INK2, fontsize=9.5, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(range(len(strategies)), [names[s] for s in strategies], fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("AUC vs synthetic honest players")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5, ncol=2)
    ax.set_title("Simulated engine use: blunt is obvious, careful is nearly invisible")
    return save(fig, "07_synthetic_auc")


def research() -> Path | None:
    """July out-of-fold flag-precision for every model family, from the deep and bake-off runs.
    Exploratory (July only); the figure title says so."""
    rows = {}
    for suffix in ("bakeoff", "deep"):
        runs = sorted(p for p in RUNS.glob(f"*_{suffix}") if (p / "metrics.json").exists())
        for run in runs:  # later runs overwrite earlier ones
            rows |= json.loads((run / "metrics.json").read_text(encoding="utf-8"))["july_oof"]
    if "mil" not in rows:
        return None
    labels = {"rule": "Rule baseline", "logreg": "Logistic regression", "mlp": "PyTorch MLP",
              "catboost": "CatBoost", "xgb": "XGBoost", "lgbm": "LightGBM (tier 1)",
              "mil_seq": "+ move-sequence CNN", "mil_seq_shuf": "+ CNN, moves shuffled",
              "mil": "Attention over games"}
    keys = [k for k in labels if k in rows]
    fig, ax = plt.subplots(figsize=(8, 0.48 * len(keys) + 1.2))
    for i, k in enumerate(keys):
        fp = rows[k]["flag_precision"]
        lo, hi = fp["ci95"]
        color = AQUA if k == "mil" else BLUE if k == "lgbm" else GRAY
        ax.barh(i, fp["value"], height=0.58, color=color)
        ax.errorbar(fp["value"], i, xerr=[[fp["value"] - lo], [hi - fp["value"]]], fmt="none",
                    ecolor=INK, elinewidth=1.2, capsize=3)
        ax.text(hi + 0.008, i, f"{fp['value']:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(keys)), [labels[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("Flag-precision in a 0.5% weekly queue (July, out-of-fold, 95% CI)")
    ax.set_title("Model research: the representation matters more than the tree library")
    return save(fig, "08_model_research")


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=SURFACE, ec=color, lw=2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.48, body, ha="center", va="top", fontsize=8.3, color=INK2,
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "-|>", "color": INK2,
                                                    "lw": 1.4})


def pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    top = [("Lichess CC0 games", "monthly PGN dumps\n(Hugging Face mirror)\nchecksum-verified", GRAY),  # noqa: E501
           ("Stream + filter", "header-only pre-filter\nrated blitz · sampled\nno moves parsed", BLUE),  # noqa: E501
           ("Parse → Parquet", "moves, %clk clocks\nUUID accounts\natomic writes", BLUE),
           ("Features (DuckDB)", "16 tier-1 features\nstrictly before cutoff\npeer z by band × TC", BLUE),  # noqa: E501
           ("Model table", "account × weekly cutoff\nJuly train · Aug test\n525k rows", BLUE)]
    for i, (t, b, c) in enumerate(top):
        box(ax, 0.1 + i * 2.4, 3.4, 2.1, 1.45, t, b, c)
        if i:
            arrow(ax, i * 2.4 - 0.2, 4.12, i * 2.4 + 0.1, 4.12)
    bottom = [("Labels: S0 snapshot", "Lichess API tosViolation\nresumable, 429-aware\n90k accounts", ORANGE),  # noqa: E501
              ("Models", "rule · logistic · LightGBM\ngrouped CV in July\nOptuna, isotonic", AQUA),
              ("Weekly queue", "top k = 0.5% eligible\n2200+ cap, dedup\nfairness audit", AQUA),
              ("Case files", "peer percentiles\nevidence + counter\nno LLM, no claims", AQUA),
              ("Serve + monitor", "FastAPI + Postgres\nweekly PSI drift\nsynthetic-only demo", AQUA)]  # noqa: E501
    for i, (t, b, c) in enumerate(bottom):
        box(ax, 0.1 + i * 2.4, 0.6, 2.1, 1.45, t, b, c)
        if i:
            arrow(ax, i * 2.4 - 0.2, 1.32, i * 2.4 + 0.1, 1.32)
    arrow(ax, 11.0, 3.4, 3.6, 2.05)  # model table feeds the models
    ax.text(6.0, 5.1, "FairPlay v2: data → features → labels → ranked review queue",
            ha="center", fontsize=13, fontweight="bold")
    return save(fig, "00_pipeline")


def timeline() -> Path:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    spans = [(0.2, 1.8, "Jun 2025", "lookback only", GRAY),
             (2.0, 3.6, "Jul 2025", "train + all choices\n(grouped CV)", BLUE),
             (3.8, 5.4, "Aug 2025", "locked test\nscored once", ORANGE)]
    for x0, x1, t, b, c in spans:
        ax.add_patch(FancyBboxPatch((x0, 1.1), x1 - x0, 0.7, boxstyle="round,pad=0.02",
                                    fc=c, ec=SURFACE))
        ax.text((x0 + x1) / 2, 1.45, t, ha="center", va="center", color="white",
                fontweight="bold")
        ax.text((x0 + x1) / 2, 0.95, b, ha="center", va="top", fontsize=9, color=INK2)
    ax.annotate("", (7.2, 1.45), (5.6, 1.45),
                arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1.4})
    ax.add_patch(FancyBboxPatch((7.3, 1.1), 2.5, 0.7, boxstyle="round,pad=0.02", fc=AQUA,
                                ec=SURFACE))
    ax.text(8.55, 1.45, "S0 labels, Oct 2026", ha="center", va="center", color="white",
            fontweight="bold")
    ax.text(8.55, 0.95, "Experiment R outcome\n(flag date unknown)", ha="center", va="top",
            fontsize=9, color=INK2)
    ax.text(0.2, 2.35, "Experiment R: rank accounts by flag status at one snapshot. "
            "Experiment P (forward test) follows flags applied after scoring.",
            fontsize=10.5, color=INK)
    return save(fig, "06_splits_timeline")


def main() -> None:
    final = metrics("final")
    paths = [pipeline(), timeline(), precision(final), fairness(final), calibration(final),
             monitoring(), tier2_cost(), synthetic(), research()]
    for p in paths:
        if p:
            print(p.as_posix())


if __name__ == "__main__":
    main()
