"""Glicko-2 rating update (Glickman 2012), the system Lichess uses.

Synthetic accounts update their rating after every game (design §0 item 12), so features 10 and 11
(rating change, daily gain) behave like real ones. Tested against the paper's worked example.
"""

import math

SCALE = 173.7178
TAU = 0.5


def _g(phi: float) -> float:
    return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)


def update(rating: float, rd: float, vol: float,
           games: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """games = [(opponent rating, opponent RD, score)]. Returns (rating, RD, volatility)."""
    mu, phi = (rating - 1500) / SCALE, rd / SCALE
    if not games:
        return rating, math.sqrt(phi**2 + vol**2) * SCALE, vol
    opp = [((r - 1500) / SCALE, d / SCALE, s) for r, d, s in games]
    E = [1 / (1 + math.exp(-_g(pj) * (mu - mj))) for mj, pj, _ in opp]
    v = 1 / sum(_g(pj) ** 2 * e * (1 - e) for (_, pj, _), e in zip(opp, E, strict=True))
    delta = v * sum(_g(pj) * (s - e) for (_, pj, s), e in zip(opp, E, strict=True))

    a = math.log(vol**2)

    def f(x: float) -> float:
        ex = math.exp(x)
        return (ex * (delta**2 - phi**2 - v - ex) / (2 * (phi**2 + v + ex) ** 2)
                - (x - a) / TAU**2)

    A = a
    if delta**2 > phi**2 + v:
        B = math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU
    fA, fB = f(A), f(B)
    while abs(B - A) > 1e-6:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2
        B, fB = C, fC
    new_vol = math.exp(A / 2)
    phi_star = math.sqrt(phi**2 + new_vol**2)
    new_phi = 1 / math.sqrt(1 / phi_star**2 + 1 / v)
    new_mu = mu + new_phi**2 * sum(_g(pj) * (s - e) for (_, pj, s), e in zip(opp, E, strict=True))
    return new_mu * SCALE + 1500, new_phi * SCALE, new_vol
