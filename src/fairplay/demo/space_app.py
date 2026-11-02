"""Gradio app for the public demo (Hugging Face Spaces, free CPU tier).

Inputs: Lichess game links, pasted PGN, an uploaded .pgn file, or a random synthetic player.
Output: the queue label, the account score against the frozen queue threshold, the template case
file and a per-game table. One request at a time (queue concurrency 1): the Lichess fetch then
also stays one-at-a-time, as Lichess asks. Inputs are not logged or stored.

uv run --group space python -m fairplay.demo.space_app
"""

import json
import random

import gradio as gr

from fairplay.casefiles.template import DISCLAIMER
from fairplay.demo.analyse import Scorer, analyse, analyse_sample
from fairplay.demo.browser import NOTES
from fairplay.demo.lichess import FetchError, fetch_pgn, game_ids

SIDES = {"Auto: the player in every game": None, "White": "w", "Black": "b"}
GAME_COLUMNS = ["date", "attention", "colour", "time_control", "result", "own_rating",
                "opponent_gap",
                "score_minus_expected", "moves", "think_median_s", "think_cv", "instant_share",
                "termination"]

ABOUT = """
**What this is.** FairPlay ranks accounts for a moderator's review queue by how much their
recent blitz games look like accounts that later carried a Lichess terms-of-service flag.
It was trained on July 2025 and tested once on August 2025 (flag-precision 0.388 at a 0.5%
weekly queue, 23x the base rate).

**What the label means.** *Would enter the review queue*: the score is at or above the score of
the last account admitted to the real August queue (median over weeks). It is a request for a
human to look, never a verdict. A flag can mean any terms-of-service violation, not only engine
use, and the model has not been checked against careful engine use (synthetic strategy C was
near chance).

**Inputs.** Lichess rated blitz games with clocks. Ten games within 30 days give an account
score; fewer give only the per-game table. Profile links are refused: the demo never looks an
account up by name. Player names are dropped on reading and nothing is stored.
"""


def label_card(r: dict) -> str:
    head = {"would enter the review queue": "### Would enter the review queue",
            "would not be queued": "### Would not be queued"}.get(r["label"],
                                                                  f"### {r['label'].capitalize()}")
    lines = [head, ""]
    if r.get("synthetic"):
        lines.append(f"Synthetic player, strategy **{r['strategy']}**: {r['strategy_note']}")
    a = r.get("account")
    if a:
        lines += [f"- Score **{a['score']}** vs queue threshold **{a['queue_threshold']}** "
                  f"(weekly capacity {a['capacity']:.1%} of eligible accounts)",
                  f"- Higher than about **{a['score_percentile']}%** of {a['score_reference']}",
                  f"- Calibrated P(flagged at the S0 snapshot): **{a['p_flag_at_s0']}**"]
    lines.append(f"- Games used: {r['games_used']} of {r['games_read']}"
                 + (f"; {r.get('games_in_30_day_window', 0)} inside the 30-day window"))
    if r.get("research_model"):
        lines.append("- Research model (attention over games, July-only, not yet tested out of "
                     "sample): the **attention** column shows which games its score leaned on")
    for why, n in r["skipped"].items():
        lines.append(f"- Skipped {n}: {why}")
    lines += ["", f"*{DISCLAIMER}*"]
    return "\n".join(lines)


def table(r: dict) -> list[list]:
    return [[g.get(c) for c in GAME_COLUMNS] for g in r["games"]]


def build(scorer: Scorer | None = None) -> gr.Blocks:
    scorer = scorer or Scorer()
    samples = scorer.b["samples"]

    def respond(r: dict):
        return label_card(r), table(r), json.dumps(r.get("case_file"), indent=2)

    def error(msg: str):
        return f"### Could not analyse\n\n{msg}", [], "null"

    def from_text(text: str, side: str):
        if not text or not text.strip():
            return error("paste at least one PGN game")
        return respond(analyse(text, scorer, SIDES[side]))

    def from_links(links: str, side: str):
        try:
            return from_text(fetch_pgn(game_ids(links or "")), side)
        except FetchError as e:
            return error(str(e))

    def from_file(path, side: str):
        if not path:
            return error("choose a .pgn file")
        with open(path, encoding="utf-8", errors="replace") as fh:
            return from_text(fh.read(2_000_000), side)

    def from_sample(strategy: str):
        pool = [s for s in samples if s["strategy"] == strategy]
        s = random.choice(pool)
        r = analyse_sample([tuple(t) for t in s["rows"]], scorer)
        r |= {"strategy": strategy, "strategy_note": NOTES[strategy]}
        return respond(r)

    with gr.Blocks(title="FairPlay research demo") as app:
        gr.Markdown("# FairPlay: moderator-flag prioritisation (research demo)")
        gr.Markdown(f"**{DISCLAIMER}**")
        side = gr.Radio(list(SIDES), value=list(SIDES)[0], label="Whose games to analyse")
        with gr.Tabs():
            with gr.Tab("Lichess game links"):
                links = gr.Textbox(lines=4, label="Game links, one per line or comma-separated",
                                   placeholder="https://lichess.org/AbCdEfGh")
                b_links = gr.Button("Analyse", variant="primary")
            with gr.Tab("Paste PGN"):
                pgn = gr.Textbox(lines=8, label="PGN with [%clk] comments (Lichess export)")
                b_pgn = gr.Button("Analyse", variant="primary")
            with gr.Tab("Upload .pgn"):
                upload = gr.File(file_types=[".pgn"], type="filepath", label="PGN file")
                b_file = gr.Button("Analyse", variant="primary")
            with gr.Tab("Synthetic player"):
                strategy = gr.Radio(list(NOTES), value="A", label="Simulated strategy")
                b_sample = gr.Button("Draw a random synthetic player", variant="primary")
        card = gr.Markdown()
        games = gr.Dataframe(headers=GAME_COLUMNS, label="Games (newest first)", wrap=True)
        case = gr.Code(language="json", label="Case file (template, no LLM)")
        with gr.Accordion("How to read this", open=False):
            gr.Markdown(ABOUT)
        outs = [card, games, case]
        b_links.click(from_links, [links, side], outs)
        b_pgn.click(from_text, [pgn, side], outs)
        b_file.click(from_file, [upload, side], outs)
        b_sample.click(from_sample, [strategy], outs)
    app.queue(default_concurrency_limit=1, max_size=20)
    return app


if __name__ == "__main__":
    build().launch()
