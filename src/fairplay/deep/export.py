"""Final attention-over-games model (variant "mil") trained on all of July, exported to numpy.

Epochs = the median best epoch of the five folds in the comparison run (no early-stopping
holdout here, so every July row trains it). The export holds only weights and the game-feature
scaler (means and standard deviations), so the demo needs no PyTorch and no training rows.
`forward` is the same network in numpy; tests check it against torch.

uv run --group research python -m fairplay.deep.export
"""

import json
from pathlib import Path

import numpy as np

from fairplay.demo.peergrid import OUT as BUNDLE

ATTENTION_FILE = "attention_mil.npz"


def to_numpy(fitted) -> dict[str, np.ndarray]:
    net = fitted.net
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    return {"g_mean": fitted.scaler.g_mean.astype(np.float32),
            "g_sd": fitted.scaler.g_sd.astype(np.float32), **sd}


def _relu(x):
    return np.maximum(x, 0)


def forward(w: dict, games: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """games [B, G, F] raw per-game features, mask [B, G] -> (logit [B], attention [B, G])."""
    x = np.where(mask[..., None], (games - w["g_mean"]) / w["g_sd"], 0).astype(np.float32)
    lin = lambda h, k: h @ w[f"{k}.weight"].T + w[f"{k}.bias"]  # noqa: E731
    h = _relu(lin(_relu(lin(x, "game.0")), "game.2"))
    gate = np.tanh(lin(h, "att_v")) * (1 / (1 + np.exp(-lin(h, "att_u"))))
    a = lin(gate, "att_w")[..., 0]
    a = np.where(mask, a, -1e4)
    a = np.exp(a - a.max(1, keepdims=True))
    a = a / a.sum(1, keepdims=True)
    z = (a[..., None] * h).sum(1)
    return lin(_relu(lin(z, "head.1")), "head.3")[:, 0], a


def main() -> None:
    from fairplay.deep import bags
    from fairplay.deep.mil import VARIANTS, Bags, fit
    from fairplay.models.data import group_codes, load_split
    from fairplay.models.run_baselines import TABLE
    from fairplay.runs import RUNS

    run = sorted(r for r in RUNS.glob("*_deep") if (r / "metrics.json").exists()
                 and "mil" in json.loads((r / "metrics.json").read_text())["variants"])[-1]
    epochs = int(np.median(json.loads((run / "metrics.json").read_text())
                           ["variants"]["mil"]["best_epochs"]))
    train = load_split(TABLE, "train")
    a = bags.load(train)
    b = Bags(a["games"], a["mask"], a["seq"][:, :, :1], train.X)  # seq unused by "mil"
    del a
    m = fit(VARIANTS["mil"], b, train.y, train.weight, group_codes(train.account_id), seed=0,
            epochs=epochs)
    weights = to_numpy(m)
    logit_t, _ = m.score(b.take(np.arange(2000)))
    logit_n, _ = forward(weights, b.games[:2000], b.mask[:2000])
    assert np.allclose(logit_t, logit_n, atol=1e-3), "numpy forward differs from torch"
    out = Path(BUNDLE)
    np.savez(out / ATTENTION_FILE, **weights)
    meta = {"source_run": run.name, "epochs": epochs, "variant": "mil",
            "game_features": list(bags.GAME_FEATURES), "trained_on": "July 2025, all rows",
            "status": "research model: July out-of-fold only, no clean test yet"}
    (out / "attention_mil.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
