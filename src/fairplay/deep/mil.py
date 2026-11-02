"""Attention over games (multiple-instance learning) and the move-sequence variant (research).

An account-cutoff row is a BAG of up to 30 games; only the bag has a label (games are never
labelled, design §4). The model (gated attention MIL, Ilse et al. 2018):

  game i -> encoder -> h_i;  a_i = softmax_i( w . (tanh(V h_i) * sigmoid(U h_i)) )  (masked)
  bag = sum_i a_i h_i  -> [optionally + tier-1 account features] -> head -> logit

a_i says which games the score leaned on. Attention is not an explanation by itself (Jain &
Wallace 2019), so `deletion_test` checks it: removing the top-attention game should move the
score more than removing a random game.

Variants (VARIANTS):
- mil:          per-game summaries only.
- mil_tab:      + the 15 tier-1 account features (same inputs as the MLP).
- mil_seq:      mil + a 1D-CNN over each game's think-time sequence (moves 11-50, log
                seconds). Built on mil, not mil_tab: mil scored higher on July (run
                20261030T192241Z_deep), so the order test starts from the best bag model.
- mil_seq_shuf: the same, with move order shuffled inside every game, in training AND scoring.
                This is the design §7 control: if mil_seq does not beat it, move ORDER carries
                no signal beyond the per-game summaries.

Training: AdamW on account-weighted binary cross-entropy; early stopping on a 10% account
holdout of the training fold (weighted PR-AUC, patience 4). No hyperparameter search: one fixed
configuration (CONFIG), stated as a handicap against the 50-trial tree models.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from fairplay.models.bakeoff import LR_COLS, Fold, mlp_inputs, stop_split
from fairplay.models.data import Split, group_codes

THREADS = 8
CONFIG = {"d": 64, "channels": 16, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
          "batch": 256, "max_epochs": 40, "patience": 4}
VARIANTS = {
    "mil": {"tab": False, "seq": False, "shuffle": False},
    "mil_tab": {"tab": True, "seq": False, "shuffle": False},
    "mil_seq": {"tab": False, "seq": True, "shuffle": False},
    "mil_seq_shuf": {"tab": False, "seq": True, "shuffle": True},
}


@dataclass
class Bags:
    games: np.ndarray  # [n, 30, F] float32
    mask: np.ndarray  # [n, 30] bool
    seq: np.ndarray  # [n, 30, L] float16, NaN = no move
    tab: np.ndarray  # [n, 16] raw tier-1 features (NaN = missing)

    def take(self, idx: np.ndarray) -> "Bags":
        return Bags(self.games[idx], self.mask[idx], self.seq[idx], self.tab[idx])


def shuffle_moves(seq: np.ndarray, seed: int) -> np.ndarray:
    """Permute the moves of every game among its own present moves; padding stays at the end."""
    rng = np.random.default_rng(seed)
    out = seq.copy()
    flat = out.reshape(-1, seq.shape[-1])
    lengths = (~np.isnan(flat.astype(np.float32))).sum(1)
    keys = rng.random(flat.shape)
    keys[np.arange(flat.shape[1])[None, :] >= lengths[:, None]] = np.inf  # padding sorts last
    order = np.argsort(keys, axis=1)
    return np.take_along_axis(flat, order, axis=1).reshape(seq.shape)


class Scaler:
    """Fold-fitted standardisation: game features over present games, sequence values over
    present moves, account features through the MLP's imputer + quantile transform."""

    def fit(self, b: Bags) -> "Scaler":
        g = b.games[b.mask]
        if not np.isfinite(g).all():
            raise ValueError("non-finite game features; rebuild the bags")
        self.g_mean, self.g_sd = g.mean(0), g.std(0) + 1e-6
        present = np.flatnonzero(b.mask.ravel())  # sequence stats from 100k games: same
        pick = np.random.default_rng(0).choice(present, min(100_000, len(present)), replace=False)
        s = b.seq.reshape(-1, b.seq.shape[-1])[np.sort(pick)].astype(np.float32)
        self.s_mean, self.s_sd = np.nanmean(s), np.nanstd(s) + 1e-6
        self.tab = mlp_inputs().fit(b.tab[:, LR_COLS])
        return self

    def tensors(self, b: Bags, with_seq: bool = True):
        """Sequences stay float16 with NaN padding (~0.5 GB for a July fold instead of ~2 GB
        as two float32 channels); the network expands them one batch at a time."""
        import torch

        games = np.where(b.mask[..., None], (b.games - self.g_mean) / self.g_sd, 0)
        seq = (((b.seq.astype(np.float32) - self.s_mean) / self.s_sd).astype(np.float16)
               if with_seq else np.zeros((len(b.mask), 1, 1), dtype=np.float16))
        tab = self.tab.transform(b.tab[:, LR_COLS])
        f = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
        return f(games), torch.tensor(b.mask), torch.tensor(seq), f(tab)


def make_net(n_game: int, n_tab: int, variant: dict, cfg: dict = CONFIG):
    import torch
    from torch import nn

    d, c = cfg["d"], cfg["channels"]

    class MILNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.game = nn.Sequential(nn.Linear(n_game, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
            if variant["seq"]:
                self.conv = nn.Sequential(nn.Conv1d(2, c, 5, padding=2), nn.ReLU(),
                                          nn.Conv1d(c, c, 3, padding=2, dilation=2), nn.ReLU())
                self.seq_out = nn.Linear(2 * c, d)
            self.att_v, self.att_u = nn.Linear(d, d), nn.Linear(d, d)
            self.att_w = nn.Linear(d, 1)
            head_in = d + (d if variant["tab"] else 0)
            if variant["tab"]:
                self.tab = nn.Sequential(nn.Linear(n_tab, d), nn.ReLU())
            self.head = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(head_in, d),
                                      nn.ReLU(), nn.Linear(d, 1))

        def forward(self, games, mask, seq, tab):
            h = self.game(games)  # [B, G, d]
            if variant["seq"]:
                B, G, L = seq.shape
                s = seq.reshape(B * G, 1, L).float()
                present = (~torch.isnan(s)).float()  # [B*G, 1, L]
                x = self.conv(torch.cat([torch.nan_to_num(s), present], 1))  # [B*G, c, L]
                n = present.sum(-1).clamp(min=1)
                mean = (x * present).sum(-1) / n
                mx = x.masked_fill(present == 0, -1e4).amax(-1).clamp(min=0)
                h = h + self.seq_out(torch.cat([mean, mx], 1)).reshape(B, G, -1)
            a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
            a = torch.softmax(a.masked_fill(~mask, -1e4), dim=1)  # [B, G]
            z = (a.unsqueeze(-1) * h).sum(1)
            if variant["tab"]:
                z = torch.cat([z, self.tab(tab)], 1)
            return self.head(z).squeeze(-1), a

    return MILNet()


def predict(net, tensors, batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    import torch

    net.eval()
    logits, atts = [], []
    with torch.no_grad():
        for i in range(0, len(tensors[0]), batch):
            out, a = net(*(t[i:i + batch] for t in tensors))
            logits.append(out.numpy())
            atts.append(a.numpy())
    return np.concatenate(logits), np.concatenate(atts)


@dataclass
class Fitted:
    net: object
    scaler: Scaler
    variant: dict
    epochs: int
    seed: int

    def prepare(self, b: Bags) -> Bags:
        if self.variant["shuffle"]:
            b = Bags(b.games, b.mask, shuffle_moves(b.seq, self.seed + 7), b.tab)
        return b

    def score(self, b: Bags) -> tuple[np.ndarray, np.ndarray]:
        return predict(self.net, self.scaler.tensors(self.prepare(b), self.variant["seq"]))


def fit(variant: dict, b: Bags, y: np.ndarray, w: np.ndarray, groups: np.ndarray, seed: int,
        epochs: int | None = None, cfg: dict = CONFIG) -> Fitted:
    """Early stopping on an account holdout, or a fixed number of `epochs` on everything."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    if variant["shuffle"]:
        b = Bags(b.games, b.mask, shuffle_moves(b.seq, seed + 7), b.tab)
    idx = np.arange(len(y))
    if epochs is None:
        fit_f, stop_f = stop_split(Fold(idx[:, None], y, w, groups), seed)
        fit_i, stop_i = fit_f.X[:, 0], stop_f.X[:, 0]
    else:
        fit_i, stop_i = idx, idx[:0]
    # scaler statistics from <= 50k rows: a full extra copy of the fold costs ~1 GB of RAM
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(fit_i, min(50_000, len(fit_i)), replace=False))
    scaler = Scaler().fit(b.take(sample))
    T = scaler.tensors(b.take(fit_i), variant["seq"])
    S = scaler.tensors(b.take(stop_i), variant["seq"]) if len(stop_i) else None
    yt = torch.tensor(y[fit_i], dtype=torch.float32)
    wt = torch.tensor(w[fit_i] / w[fit_i].mean(), dtype=torch.float32)
    net = make_net(b.games.shape[2], T[3].shape[1], variant, cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, (epochs or cfg["max_epochs"]) + 1):
        net.train()
        for i in torch.randperm(len(yt), generator=gen).split(cfg["batch"]):
            opt.zero_grad()
            out, _ = net(*(t[i] for t in T))
            (loss_fn(out, yt[i]) * wt[i]).mean().backward()
            opt.step()
        if S is None:
            continue
        ap = average_precision_score(y[stop_i], predict(net, S)[0], sample_weight=w[stop_i])
        if ap > best:
            best, best_epoch, stale = ap, epoch, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return Fitted(net, scaler, variant, best_epoch or epochs, seed)


def grouped_oof(variant: dict, b: Bags, split: Split, seed: int, n_folds: int = 5
                ) -> tuple[np.ndarray, np.ndarray, list]:
    """OOF logits and attention, folds grouped by account (the same folds as every model).
    Also returns each fold's (model, scored rows), for the deletion test."""
    oof = np.empty(len(split.y))
    att = np.zeros(b.mask.shape, dtype=np.float16)
    fitted = []
    codes = group_codes(split.account_id)
    folds = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (fit_idx, pred_idx) in enumerate(folds.split(b.games, split.y, groups=split.groups())):
        m = fit(variant, b.take(fit_idx), split.y[fit_idx], split.weight[fit_idx],
                codes[fit_idx], seed)
        oof[pred_idx], a = m.score(b.take(pred_idx))
        att[pred_idx] = a
        fitted.append((m, pred_idx))
        print(f"  fold {k + 1}/{n_folds}: best epoch {m.epochs}", flush=True)
    return oof, att, fitted


def deletion_test(m: Fitted, b: Bags, rows: np.ndarray, seed: int) -> dict:
    """For each row (bags of >= 2 games): drop in logit when its highest-attention game is
    masked out, versus when a random other game is. Attention that points at the games the score
    depends on gives a larger drop for the top game."""
    rows = rows[b.mask[rows].sum(1) >= 2]
    sub = b.take(rows)
    full, att = m.score(sub)
    top = att.argmax(1)
    rng = np.random.default_rng(seed)
    rand = np.array([rng.choice(np.flatnonzero(mk & (np.arange(len(mk)) != t)))
                     for mk, t in zip(sub.mask, top, strict=True)])
    drops = {}
    for name, drop in (("top_attention", top), ("random_game", rand)):
        mask = sub.mask.copy()
        mask[np.arange(len(rows)), drop] = False
        drops[name] = full - m.score(Bags(sub.games, mask, sub.seq, sub.tab))[0]
    diff = drops["top_attention"] - drops["random_game"]
    return {"rows": int(len(rows)),
            "mean_logit_drop_top": float(drops["top_attention"].mean()),
            "mean_logit_drop_random": float(drops["random_game"].mean()),
            "share_top_drop_larger": float((diff > 0).mean()),
            "mean_top_attention": float(att.max(1).mean()),
            "mean_uniform_attention": float((1 / sub.mask.sum(1)).mean())}
