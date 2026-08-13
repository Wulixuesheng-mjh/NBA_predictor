"""Add betting_lines table for point spread & over/under.

Revision ID: 002_betting_lines
Revises: 001_initial
Create Date: 2026-08-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "002_betting_lines"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "betting_lines",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("season", sa.String(7), nullable=False),
        sa.Column("spread_line", sa.Float(), nullable=False),
        sa.Column("total_line", sa.Float(), nullable=False),
        sa.Column("home_spread", sa.Float(), nullable=True),
        sa.Column("away_spread", sa.Float(), nullable=True),
        sa.Column("source", sa.String(30), nullable=False, server_default="manual"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("game_id"),
    )
    op.create_index("idx_betting_lines_season", "betting_lines", ["season"])


def downgrade() -> None:
    op.drop_index("idx_betting_lines_season", table_name="betting_lines")
    op.drop_table("betting_lines")
