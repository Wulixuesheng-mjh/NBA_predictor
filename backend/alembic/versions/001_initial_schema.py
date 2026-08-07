"""Initial schema — all 9 tables for NBA Predictor platform.

Revision ID: 001_initial
Revises: None
Create Date: 2026-07-30
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import JSON

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. teams ──────────────────────────────────────────
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("nba_team_id", sa.Integer(), nullable=False),
        sa.Column("abbreviation", sa.String(3), nullable=False),
        sa.Column("city", sa.String(50), nullable=False),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("full_name", sa.String(100), nullable=False),
        sa.Column("conference", sa.String(10), nullable=True),
        sa.Column("division", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("nba_team_id"),
    )

    # ── 2. players ────────────────────────────────────────
    op.create_table(
        "players",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("nba_player_id", sa.Integer(), nullable=False),
        sa.Column("first_name", sa.String(50), nullable=True),
        sa.Column("last_name", sa.String(50), nullable=True),
        sa.Column("full_name", sa.String(100), nullable=True),
        sa.Column("team_id", sa.Integer(), nullable=True),
        sa.Column("jersey_number", sa.String(5), nullable=True),
        sa.Column("position", sa.String(5), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("nba_player_id"),
    )

    # ── 3. games ──────────────────────────────────────────
    op.create_table(
        "games",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("nba_game_id", sa.String(20), nullable=False),
        sa.Column("season", sa.String(7), nullable=False),
        sa.Column("game_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="Scheduled"),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.Column("quarter", sa.Integer(), nullable=True),
        sa.Column("time_remaining", sa.String(10), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["home_team_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["away_team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("nba_game_id"),
    )

    # ── 4. team_game_stats ────────────────────────────────
    op.create_table(
        "team_game_stats",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("is_home", sa.Boolean(), nullable=False),
        sa.Column("pts", sa.Integer(), default=0),
        sa.Column("reb", sa.Integer(), default=0),
        sa.Column("ast", sa.Integer(), default=0),
        sa.Column("stl", sa.Integer(), default=0),
        sa.Column("blk", sa.Integer(), default=0),
        sa.Column("tov", sa.Integer(), default=0),
        sa.Column("pf", sa.Integer(), default=0),
        sa.Column("fgm", sa.Integer(), default=0),
        sa.Column("fga", sa.Integer(), default=0),
        sa.Column("fg3m", sa.Integer(), default=0),
        sa.Column("fg3a", sa.Integer(), default=0),
        sa.Column("ftm", sa.Integer(), default=0),
        sa.Column("fta", sa.Integer(), default=0),
        sa.Column("off_rtg", sa.Float(), nullable=True),
        sa.Column("def_rtg", sa.Float(), nullable=True),
        sa.Column("pace", sa.Float(), nullable=True),
        sa.Column("ts_pct", sa.Float(), nullable=True),
        sa.Column("efg_pct", sa.Float(), nullable=True),
        sa.Column("reb_pct", sa.Float(), nullable=True),
        sa.Column("ast_pct", sa.Float(), nullable=True),
        sa.Column("tov_pct", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── 5. game_player_stats ──────────────────────────────
    op.create_table(
        "game_player_stats",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("minutes", sa.String(10), nullable=True),
        sa.Column("pts", sa.Integer(), default=0),
        sa.Column("reb", sa.Integer(), default=0),
        sa.Column("ast", sa.Integer(), default=0),
        sa.Column("stl", sa.Integer(), default=0),
        sa.Column("blk", sa.Integer(), default=0),
        sa.Column("tov", sa.Integer(), default=0),
        sa.Column("pf", sa.Integer(), default=0),
        sa.Column("fgm", sa.Integer(), default=0),
        sa.Column("fga", sa.Integer(), default=0),
        sa.Column("fg3m", sa.Integer(), default=0),
        sa.Column("fg3a", sa.Integer(), default=0),
        sa.Column("ftm", sa.Integer(), default=0),
        sa.Column("fta", sa.Integer(), default=0),
        sa.Column("plus_minus", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.ForeignKeyConstraint(["player_id"], ["players.id"]),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("game_id", "player_id", name="uq_game_player"),
    )

    # ── 6. player_injuries ────────────────────────────────
    op.create_table(
        "player_injuries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(10), nullable=False, server_default="nba_api"),
        sa.Column("confirmation", sa.String(15), nullable=False, server_default="pending"),
        sa.Column("injury_status", sa.String(20), nullable=False),
        sa.Column("injury_type", sa.String(100), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("reported_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expected_return", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("raw_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["player_id"], ["players.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_injuries_player", "player_injuries", ["player_id", "is_active"])
    op.create_index("idx_injuries_active", "player_injuries", ["is_active"])

    # ── 7. predictions ────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=True),
        sa.Column("home_team_id", sa.Integer(), nullable=True),
        sa.Column("away_team_id", sa.Integer(), nullable=True),
        sa.Column("model_name", sa.String(20), nullable=False),
        sa.Column("model_version", sa.String(20), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("win_prob_home", sa.Float(), nullable=False),
        sa.Column("win_prob_away", sa.Float(), nullable=False),
        sa.Column("predicted_spread", sa.Float(), nullable=True),
        sa.Column("spread_confidence", sa.Float(), nullable=True),
        sa.Column("predicted_total", sa.Float(), nullable=True),
        sa.Column("total_confidence", sa.Float(), nullable=True),
        sa.Column("calibrated", sa.Boolean(), default=False),
        sa.Column("calibration_factors", JSON(), nullable=True),
        sa.Column("features_json", JSON(), nullable=True),
        sa.Column("lookback_games", sa.Integer(), default=10),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.ForeignKeyConstraint(["home_team_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["away_team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── 8. prediction_results ─────────────────────────────
    op.create_table(
        "prediction_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("actual_spread", sa.Float(), nullable=True),
        sa.Column("actual_total", sa.Integer(), nullable=True),
        sa.Column("actual_winner", sa.String(4), nullable=True),
        sa.Column("spread_correct", sa.Boolean(), nullable=True),
        sa.Column("total_correct", sa.Boolean(), nullable=True),
        sa.Column("win_correct", sa.Boolean(), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prediction_id"),
    )

    # ── 9. data_collection_log ────────────────────────────
    op.create_table(
        "data_collection_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("task_name", sa.String(50), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(10), nullable=False, server_default="running"),
        sa.Column("records_collected", sa.Integer(), default=0),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("data_collection_log")
    op.drop_table("prediction_results")
    op.drop_table("predictions")
    op.drop_index("idx_injuries_active", table_name="player_injuries")
    op.drop_index("idx_injuries_player", table_name="player_injuries")
    op.drop_table("player_injuries")
    op.drop_table("game_player_stats")
    op.drop_table("team_game_stats")
    op.drop_table("games")
    op.drop_table("players")
    op.drop_table("teams")
