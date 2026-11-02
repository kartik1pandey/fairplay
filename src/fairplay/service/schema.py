"""Request schema for POST /v1/games (design §11). Malformed input -> 422, before any write.

Players arrive as UUIDs: usernames never reach the service (design §9). The scope rules mirror
the offline cleaner exactly, so the online table holds the same games the models were built on:
rated blitz by the duration rule, no BOT player, not abandoned, at least 20 plies.
"""

import datetime as dt
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fairplay.ingest.prefilter import BLITZ_MAX_SECONDS, BLITZ_MIN_SECONDS, MIN_PLIES


class PlayerIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_id: UUID
    rating: int = Field(ge=400, le=4000)
    rating_diff: int | None = Field(default=None, ge=-1000, le=1000)
    is_bot: bool = False


class GameIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    game_id: str = Field(pattern=r"^[A-Za-z0-9]{8}$")  # Lichess game id
    version: int = Field(default=1, ge=1)  # a correction re-sends the game with version + 1
    start_ts: dt.datetime  # UTCDate + UTCTime: game START (design §0.19)
    tc_base: int = Field(ge=0, le=10_800)
    tc_inc: int = Field(ge=0, le=180)
    rated: bool = True
    event_kind: Literal["game", "arena", "swiss"] = "game"
    white: PlayerIn
    black: PlayerIn
    result: Literal["1-0", "0-1", "1/2-1/2"]
    termination: Literal["normal", "time_forfeit", "rules_infraction", "abandoned", "other"]
    moves: list[str] = Field(max_length=1000)  # SAN
    clocks: list[int] | None = Field(default=None, max_length=1000)  # %clk after each ply, s

    @model_validator(mode="after")
    def _check(self) -> "GameIn":
        if self.start_ts.tzinfo is None or self.start_ts.utcoffset() != dt.timedelta(0):
            raise ValueError("start_ts must be UTC with an explicit offset")
        if self.white.account_id == self.black.account_id:
            raise ValueError("white and black are the same account")
        if self.clocks is not None and any(c < 0 for c in self.clocks):
            raise ValueError("negative clock")
        return self

    def skip_reason(self) -> str | None:
        """Out of scope (stored nowhere, answered 200): same rules as the offline cleaner."""
        duration = self.tc_base + 40 * self.tc_inc
        if not self.rated:
            return "unrated"
        if not BLITZ_MIN_SECONDS <= duration <= BLITZ_MAX_SECONDS:
            return "not_blitz"
        if self.white.is_bot or self.black.is_bot:
            return "bot_game"
        if self.termination == "abandoned":
            return "abandoned"
        if len(self.moves) < MIN_PLIES:
            return "too_short"
        return None


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queue_id: str = Field(max_length=200)
    account_id: UUID
    decision: Literal["clear", "watch", "escalate"]
    reason: str = Field(min_length=1, max_length=2000)


class GameOut(BaseModel):
    game_id: str
    status: Literal["stored", "duplicate", "stale_version", "skipped", "deleted"]
    reason: str | None = None
    accounts_updated: list[str] = []
    feature_version: str
    model_version: str
