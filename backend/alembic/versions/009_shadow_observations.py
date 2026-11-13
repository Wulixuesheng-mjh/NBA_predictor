"""Separate diagnostic pregame observations from formal 200-game acceptance."""
from alembic import op
import sqlalchemy as sa
revision='009_shadow_observations'
down_revision='008_box_score_sources'
branch_labels=None
depends_on=None


def upgrade():
    op.create_table('shadow_observation_runs',
        sa.Column('id',sa.Integer(),primary_key=True),
        sa.Column('name',sa.String(100),unique=True,nullable=False),
        sa.Column('protocol',sa.JSON(),nullable=False),
        sa.Column('protocol_hash',sa.String(64),nullable=False),
        sa.Column('enabled',sa.Boolean(),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),server_default=sa.func.now()))
    op.create_table('shadow_observations',
        sa.Column('id',sa.Integer(),primary_key=True),
        sa.Column('run_id',sa.Integer(),sa.ForeignKey('shadow_observation_runs.id'),nullable=False),
        sa.Column('game_id',sa.Integer(),sa.ForeignKey('games.id'),nullable=False),
        sa.Column('as_of',sa.DateTime(timezone=True),nullable=False),
        sa.Column('scheduled_start_utc',sa.DateTime(timezone=True),nullable=False),
        sa.Column('payload',sa.JSON(),nullable=False),
        sa.Column('payload_hash',sa.String(64),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),server_default=sa.func.now()),
        sa.UniqueConstraint('run_id','game_id',name='uq_shadow_observation_run_game'))


def downgrade():
    op.drop_table('shadow_observations')
    op.drop_table('shadow_observation_runs')
