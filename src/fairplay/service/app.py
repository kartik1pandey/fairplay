"""Online service (design §11). Local: writes enabled. Public demo: read-only, synthetic only.

POST /v1/games            store a finished game (idempotent), recompute both accounts' features
DELETE /v1/games/{id}     tombstone (version must increase)
GET  /v1/accounts/{uuid}/features   latest online features (local only: no lookup in public)
GET  /v1/queue?cutoff=YYYY-MM-DD      the stored weekly queue (public demo: synthetic DB only)
POST /v1/queue/decisions              reviewer decision (local only)
GET  /health

Logs carry game keys and UUIDs only, never request bodies (design §9). If the database is down
the request fails with 503 and nothing is half-written (one transaction per write).

FAIRPLAY_DB=sqlite:///private/online.db uv run uvicorn fairplay.service.app:local_app --factory
"""

import datetime as dt
import logging
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from fairplay.casefiles.template import DISCLAIMER
from fairplay.features.game_features import FEATURE_VERSION
from fairplay.ingest.clean import game_key_secret
from fairplay.service import online, store, weekly
from fairplay.service.db import Database
from fairplay.service.schema import DecisionIn, GameIn, GameOut

log = logging.getLogger("fairplay.service")
PEERS = Path("data/processed/peer_stats_clock30.parquet")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def create_app(db: Database, peers: Path = PEERS, secret: str | None = None,
               allow_writes: bool = False, model_version: str = "none",
               clock: Callable[[], dt.datetime] = utcnow) -> FastAPI:
    app = FastAPI(title="FairPlay online service", docs_url=None, redoc_url=None)
    secret = secret if secret is not None else game_key_secret()

    async def db_down(request: Request, exc: Exception) -> JSONResponse:
        log.error("database error on %s", request.url.path)  # no body, no parameters
        return JSONResponse({"detail": "database unavailable"}, status_code=503)

    app.add_exception_handler(sqlite3.OperationalError, db_down)
    if db.pg:
        import psycopg

        app.add_exception_handler(psycopg.OperationalError, db_down)

    def refresh(accounts: set[str], now: dt.datetime) -> list[str]:
        if accounts:
            online.recompute(db, sorted(accounts), now, peers, now)  # one SQL pass for all
        return sorted(accounts)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "feature_version": FEATURE_VERSION, "model_version": model_version}

    @app.get("/v1/queue")
    def get_queue(cutoff: dt.date) -> dict:
        q = weekly.latest_queue(db, str(cutoff))
        if q is None:
            raise HTTPException(404, "no queue for that cutoff")
        return q | {"notice": DISCLAIMER}

    if allow_writes:
        @app.post("/v1/games", response_model=GameOut)
        def post_game(game: GameIn) -> GameOut:
            out = {"game_id": game.game_id, "feature_version": FEATURE_VERSION,
                   "model_version": model_version}
            reason = game.skip_reason()
            if reason:
                return GameOut(status="skipped", reason=reason, **out)
            now = clock()
            status, accounts = store.upsert_game(db, game, secret, now)
            log.info("game %s %s", store.game_key(secret, game.game_id), status)
            return GameOut(status=status, accounts_updated=refresh(accounts, now), **out)

        @app.delete("/v1/games/{game_id}", response_model=GameOut)
        def delete_game(game_id: str, version: int = Query(ge=2)) -> GameOut:
            now = clock()
            status, accounts = store.delete_game(db, game_id, version, secret, now)
            return GameOut(game_id=game_id, status=status, accounts_updated=refresh(accounts, now),
                           feature_version=FEATURE_VERSION, model_version=model_version)

        @app.post("/v1/queue/decisions")
        def decide(d: DecisionIn) -> dict:
            try:
                weekly.record_decision(db, d.queue_id, str(d.account_id), d.decision, d.reason,
                                       clock())
            except KeyError as e:
                raise HTTPException(404, str(e)) from e
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            return {"ok": True}

        @app.get("/v1/accounts/{account_id}/features")
        def features(account_id: UUID) -> dict:
            row = online.stored(db, str(account_id))
            if row is None:
                raise HTTPException(404, "no games stored for this account")
            return row

    return app


def local_app() -> FastAPI:
    """Local, write-enabled instance (never deployed publicly)."""
    return create_app(Database(os.environ.get("FAIRPLAY_DB", "sqlite:///private/online.db")),
                      allow_writes=True)
