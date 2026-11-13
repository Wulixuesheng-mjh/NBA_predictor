"""Add 体彩胜负赔率 (moneyline decimal odds) to predictions.

Revision ID: 004_prediction_moneyline_odds
Revises: 003_historical_player_injuries
Create Date: 2026-10-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "004_prediction_moneyline_odds"
down_revision: Union[str, None] = "003_historical_player_injuries"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("home_odds", sa.Float(), nullable=True))
    op.add_column("predictions", sa.Column("away_odds", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("predictions", "away_odds")
    op.drop_column("predictions", "home_odds")
