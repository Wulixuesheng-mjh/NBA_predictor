"""On-disk cache for built feature datasets.

Building the training set is by far the slowest part of a retrain: every
completed game needs its rolling-window features computed *as of that date*
(~1 minute per 1,000 games on a laptop-class Postgres). The resulting arrays
are a pure function of:

  * the feature schema (``FEATURE_NAMES``),
  * the build parameters (seasons, lookback, min prior games, injury source),
  * the current database state (games, box scores, rosters, injuries).

So we fingerprint all of those and store the arrays as a compressed ``.npz``
under ``saved_models/feature_cache``. Any new game, corrected score, roster
move or injury edit changes the fingerprint and forces a rebuild — the cache
only skips *redundant* work (an MLP retry after an XGBoost-only run, a
hyper-parameter sweep, a backtest, or a manual retrain on unchanged data).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Bump when the *semantics* of the build change (not just feature names),
# e.g. a different rolling-window rule, a new sample filter, new caps, or a
# leakage fix that changes historical feature values.
CACHE_FORMAT_VERSION = 9


def _default_cache_dir() -> Path:
    env = os.getenv("FEATURE_CACHE_DIR")
    if env:
        return Path(env)
    # ml/src/features/dataset_cache.py -> ml/saved_models/feature_cache
    return Path(__file__).resolve().parents[2] / "saved_models" / "feature_cache"


class DatasetCache:
    """Fingerprinted .npz cache for :class:`TrainingDataset`."""

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()

    # ── Fingerprint ─────────────────────────────────────────

    @staticmethod
    def _schema_hash(feature_names: list[str]) -> str:
        payload = f"v{CACHE_FORMAT_VERSION}|{len(feature_names)}|" + "|".join(feature_names)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    async def _data_fingerprint(session: AsyncSession) -> dict[str, str]:
        """Cheap (count, max row id, max updated_at) per table.

        Uses a single query per table; the whole thing costs well under a
        second even on ~700k box-score rows, and any insert/update bumps it.
        """
        from app.models.game import Game
        from app.models.team_game_stats import TeamGameStats
        from app.models.game_player_stats import GamePlayerStats
        from app.models.player import Player
        from app.models.player_injuries import PlayerInjury
        from app.models.team import Team
        from app.models.data_revision import DataRevision
        from app.models.injury_report import InjuryReport
        from app.models.roster_snapshot import RosterSnapshot
        from app.models.box_score_snapshot import BoxScoreSnapshot

        try:
            from app.models.historical_player_injuries import HistoricalPlayerInjury

            hist_model = HistoricalPlayerInjury
        except Exception:  # pragma: no cover - table optional in some checkouts
            hist_model = None

        models = {
            "games": Game,
            "team_game_stats": TeamGameStats,
            "game_player_stats": GamePlayerStats,
            "players": Player,
            "player_injuries": PlayerInjury,
            "teams": Team,
            "injury_reports": InjuryReport,
            "roster_snapshots": RosterSnapshot,
            "box_score_snapshots": BoxScoreSnapshot,
        }
        if hist_model is not None:
            models["historical_player_injuries"] = hist_model

        out: dict[str, str] = {}
        for name, model in models.items():
            count_col = func.count().label("n")
            cols = [count_col]
            if hasattr(model, "id"):
                cols.append(func.max(model.id).label("max_id"))
            if hasattr(model, "updated_at"):
                cols.append(func.max(model.updated_at).label("max_updated"))
            elif hasattr(model, "observed_at"):
                cols.append(func.max(model.observed_at).label("max_observed"))
            elif hasattr(model, "created_at"):
                cols.append(func.max(model.created_at).label("max_created"))

            row = (await session.execute(select(*cols))).one()
            out[name] = "|".join("" if v is None else str(v) for v in row)
        revision = await session.scalar(select(func.max(DataRevision.revision)))
        out["explicit_data_revision"] = str(revision or 0)
        return out

    async def key(
        self,
        session: AsyncSession,
        *,
        feature_names: list[str],
        seasons: list[str] | None,
        lookback_games: int,
        min_games_before: int,
        use_historical: bool,
    ) -> str:
        fingerprint = {
            "format": CACHE_FORMAT_VERSION,
            "schema": self._schema_hash(feature_names),
            "seasons": sorted(seasons) if seasons else "all",
            "lookback_games": lookback_games,
            "min_games_before": min_games_before,
            "use_historical": bool(use_historical),
            "data": await self._data_fingerprint(session),
        }
        blob = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # ── Load / save ─────────────────────────────────────────

    def path_for(self, key: str) -> Path:
        return self.cache_dir / f"train_{key}.npz"

    def load(self, key: str, expected_dim: int):
        """Return a cached TrainingDataset, or None if missing/corrupt."""
        from ml.src.features.feature_builder import TrainingDataset

        path = self.path_for(key)
        if not path.exists():
            return None

        try:
            with np.load(path, allow_pickle=False) as data:
                X = data["X"]
                y_win = data["y_win"]
                y_spread = data["y_spread"]
                y_total = data["y_total"]
                sample_info = json.loads(str(data["sample_info"]))
                meta = json.loads(str(data["meta"]))
        except Exception as e:
            logger.warning("Feature cache %s unreadable (%s) — rebuilding", path.name, e)
            return None

        if X.ndim != 2 or X.shape[1] != expected_dim:
            logger.warning(
                "Feature cache %s has %s features, expected %d — rebuilding",
                path.name, X.shape[1] if X.ndim == 2 else "n/a", expected_dim,
            )
            return None
        if meta.get("format") != CACHE_FORMAT_VERSION:
            logger.warning("Feature cache %s uses incompatible preprocessing semantics", path.name)
            return None
        if not (len(X) == len(y_win) == len(y_spread) == len(y_total) == len(sample_info)):
            logger.warning("Feature cache %s is inconsistent — rebuilding", path.name)
            return None

        logger.info(
            "Loaded training dataset from cache: %d samples (built %s, key %s)",
            len(X), meta.get("built_at", "?"), key,
        )
        return TrainingDataset(
            X=X, y_win=y_win, y_spread=y_spread, y_total=y_total, sample_info=sample_info, cache_key=key,
        )

    def save(self, key: str, dataset) -> None:
        """Persist a dataset atomically (temp file + rename)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        tmp = path.with_suffix(".npz.tmp")

        meta = {
            "key": key,
            "format": CACHE_FORMAT_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": int(len(dataset.X)),
            "n_features": int(dataset.X.shape[1]),
        }

        try:
            # Pass a file object: np.savez_compressed(path) would append ".npz"
            # to the temp name and break the atomic rename.
            with open(tmp, "wb") as fh:
                np.savez_compressed(
                    fh,
                    X=dataset.X.astype(np.float32),
                    y_win=dataset.y_win.astype(np.float32),
                    y_spread=dataset.y_spread.astype(np.float32),
                    y_total=dataset.y_total.astype(np.float32),
                    sample_info=np.array(json.dumps(dataset.sample_info, default=str)),
                    meta=np.array(json.dumps(meta)),
                )
            os.replace(tmp, path)
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(
                "Cached training dataset: %d samples → %s (%.1f MB)",
                len(dataset.X), path.name, size_mb,
            )
        except Exception as e:
            logger.warning("Could not write feature cache %s: %s", path, e)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def clear(self) -> int:
        """Delete every cached dataset. Returns the number of files removed."""
        if not self.cache_dir.exists():
            return 0
        removed = 0
        for f in self.cache_dir.glob("train_*.npz"):
            f.unlink(missing_ok=True)
            removed += 1
        return removed

    def info(self) -> list[dict]:
        """List cached datasets with size and build time (for debugging)."""
        if not self.cache_dir.exists():
            return []
        out = []
        for f in sorted(self.cache_dir.glob("train_*.npz")):
            stat = f.stat()
            out.append(
                {
                    "file": f.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "mtime": stat.st_mtime,
                }
            )
        return out
