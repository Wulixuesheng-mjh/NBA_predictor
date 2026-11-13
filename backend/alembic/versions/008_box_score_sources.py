"""Preserve immutable source evidence for exact rotation repairs."""
from alembic import op
import sqlalchemy as sa

revision='008_box_score_sources'
down_revision='007_injury_payload_dedup'
branch_labels=None
depends_on=None


def upgrade():
    op.create_table('box_score_snapshots',
        sa.Column('id',sa.Integer(),primary_key=True),
        sa.Column('game_id',sa.Integer(),sa.ForeignKey('games.id'),nullable=False),
        sa.Column('source',sa.String(40),nullable=False),
        sa.Column('source_url',sa.Text(),nullable=False),
        sa.Column('source_file',sa.Text(),nullable=False),
        sa.Column('sha256',sa.String(64),nullable=False),
        sa.Column('observed_at',sa.DateTime(timezone=True),nullable=False),
        sa.Column('published_at',sa.DateTime(timezone=True)),
        sa.Column('payload',sa.JSON(),nullable=False),
        sa.Column('validation',sa.JSON(),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),server_default=sa.func.now()),
        sa.UniqueConstraint('game_id','sha256',name='uq_box_score_source_payload'))
    op.create_index('ix_box_score_snapshots_game_id','box_score_snapshots',['game_id'])


def downgrade():
    op.drop_table('box_score_snapshots')
