"""Add historical_player_injuries table for training-only injury snapshots.

Revision ID: 003_historical_player_injuries
Revises: 002_betting_lines
Create Date: 2026-08-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "003_historical_player_injuries"
down_revision: Union[str, None] = "002_betting_lines"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "historical_player_injuries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("game_date", sa.Date(), nullable=False),
        sa.Column("injury_status", sa.String(20), nullable=False),
        sa.Column("injury_type", sa.String(100), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="nbainjuries"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["player_id"], ["players.id"]),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("player_id", "team_id", "game_date", name="uq_hist_inj_player_team_date"),
    )
    op.create_index("idx_hist_inj_game_date", "historical_player_injuries", ["game_date"])
    op.create_index("idx_hist_inj_team_game", "historical_player_injuries", ["team_id", "game_date"])
    op.create_index("idx_hist_inj_player_game", "historical_player_injuries", ["player_id", "game_date"])


def downgrade() -> None:
    op.drop_index("idx_hist_inj_player_game", table_name="historical_player_injuries")
    op.drop_index("idx_hist_inj_team_game", table_name="historical_player_injuries")
    op.drop_index("idx_hist_inj_game_date", table_name="historical_player_injuries")
    op.drop_table("historical_player_injuries")
