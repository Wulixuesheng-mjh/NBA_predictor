"""Add data quality, as-of provenance and immutable prediction ledger fields.

Revision ID: 005_data_quality_ledger
Revises: 004_prediction_moneyline_odds
"""

from alembic import op
import sqlalchemy as sa

revision = "005_data_quality_ledger"
down_revision = "004_prediction_moneyline_odds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("games", sa.Column("source_game_date", sa.Date(), nullable=True))
    op.add_column("games", sa.Column("scheduled_start_utc", sa.DateTime(timezone=True), nullable=True))
    op.add_column("games", sa.Column("time_precision", sa.String(10), nullable=False, server_default="date"))
    op.add_column("games", sa.Column("game_type", sa.String(20), nullable=False, server_default="regular"))
    op.add_column("games", sa.Column("score_complete", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("games", sa.Column("team_stats_complete", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("games", sa.Column("player_stats_complete", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("games", sa.Column("stats_updated_at", sa.DateTime(timezone=True), nullable=True))

    for table in ("team_game_stats", "game_player_stats"):
        op.add_column(table, sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.add_column("team_game_stats", sa.Column("oreb", sa.Integer(), nullable=True))
    op.add_column("team_game_stats", sa.Column("dreb", sa.Integer(), nullable=True))
    op.add_column("team_game_stats", sa.Column("game_minutes", sa.Float(), nullable=True))
    op.add_column("team_game_stats", sa.Column("possessions_estimated", sa.Boolean(), nullable=False, server_default=sa.true()))

    for table in ("player_injuries", "historical_player_injuries"):
        op.add_column(table, sa.Column("game_id", sa.Integer(), sa.ForeignKey("games.id"), nullable=True))
        op.add_column(table, sa.Column("published_at", sa.DateTime(timezone=True), nullable=True))
        op.add_column(table, sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("player_injuries", sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=True))

    op.create_table(
        "data_revisions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("reason", sa.String(200), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.execute("INSERT INTO data_revisions (id, revision, reason) VALUES (1, 1, 'schema-v6')")
    op.create_table(
        "injury_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("game_id", sa.Integer(), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("source", sa.String(40), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_complete", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("game_id", "team_id", "source", "observed_at", name="uq_injury_report_snapshot"),
    )
    op.create_index("idx_injury_report_game_team", "injury_reports", ["game_id", "team_id", "observed_at"])
    op.create_table(
        "roster_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("player_id", sa.Integer(), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.String(40), nullable=False, server_default="collector"),
        sa.Column("observed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_roster_snapshot_asof", "roster_snapshots", ["team_id", "effective_from", "effective_until"])
    op.create_index("idx_roster_snapshot_player", "roster_snapshots", ["player_id", "effective_from"])

    prediction_columns = [
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("prediction_mode", sa.String(30), nullable=False, server_default="hypothetical_preview"),
        sa.Column("market_home_handicap", sa.Float(), nullable=True),
        sa.Column("market_total", sa.Float(), nullable=True),
        sa.Column("raw_output", sa.JSON(), nullable=True),
        sa.Column("final_output", sa.JSON(), nullable=True),
        sa.Column("baseline_output", sa.JSON(), nullable=True),
        sa.Column("injury_snapshot", sa.JSON(), nullable=True),
        sa.Column("data_quality", sa.JSON(), nullable=True),
        sa.Column("feature_names", sa.JSON(), nullable=True),
        sa.Column("feature_vector", sa.JSON(), nullable=True),
        sa.Column("feature_semantic_version", sa.String(40), nullable=True),
        sa.Column("feature_hash", sa.String(64), nullable=True),
    ]
    for column in prediction_columns:
        op.add_column("predictions", column)


def downgrade() -> None:
    for name in (
        "feature_hash", "feature_semantic_version", "feature_vector", "feature_names",
        "data_quality", "injury_snapshot", "baseline_output", "final_output", "raw_output",
        "market_total", "market_home_handicap", "prediction_mode", "as_of",
    ):
        op.drop_column("predictions", name)
    op.drop_table("roster_snapshots")
    op.drop_table("injury_reports")
    op.drop_table("data_revisions")
    op.drop_column("player_injuries", "team_id")
    for table in ("historical_player_injuries", "player_injuries"):
        for name in ("observed_at", "published_at", "game_id"):
            op.drop_column(table, name)
    for name in ("possessions_estimated", "game_minutes", "dreb", "oreb", "updated_at"):
        op.drop_column("team_game_stats", name)
    op.drop_column("game_player_stats", "updated_at")
    for name in (
        "stats_updated_at", "player_stats_complete", "team_stats_complete", "score_complete",
        "game_type", "time_precision", "scheduled_start_utc", "source_game_date",
    ):
        op.drop_column("games", name)
