"""Verified injury payloads and locked prospective comparisons.

Revision ID: 006_snapshot_prospective
Revises: 005_data_quality_ledger
"""
from alembic import op
import sqlalchemy as sa

revision = '006_snapshot_prospective'
down_revision = '005_data_quality_ledger'
branch_labels = None
depends_on = None


def upgrade():
    op.create_unique_constraint('uq_team_game_stats_game_team', 'team_game_stats', ['game_id', 'team_id'])
    for column in (
        sa.Column('source_url', sa.Text()),
        sa.Column('source_sha256', sa.String(64)),
        sa.Column('entries_json', sa.JSON()),
        sa.Column('parse_issues', sa.JSON()),
        sa.Column('publication_verified', sa.Boolean(), nullable=False, server_default=sa.false()),
    ):
        op.add_column('injury_reports', column)
    op.create_table(
        'prospective_runs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('protocol', sa.JSON(), nullable=False),
        sa.Column('protocol_hash', sa.String(64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('target_games', sa.Integer(), nullable=False, server_default='200'),
    )
    op.create_table(
        'prospective_forecasts',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('run_id', sa.Integer(), sa.ForeignKey('prospective_runs.id'), nullable=False),
        sa.Column('game_id', sa.Integer(), sa.ForeignKey('games.id'), nullable=False),
        sa.Column('as_of', sa.DateTime(timezone=True), nullable=False),
        sa.Column('scheduled_start_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('feature_names', sa.JSON(), nullable=False),
        sa.Column('feature_vector', sa.JSON(), nullable=False),
        sa.Column('feature_hash', sa.String(64), nullable=False),
        sa.Column('baseline_output', sa.JSON(), nullable=False),
        sa.Column('candidate_output', sa.JSON(), nullable=False),
        sa.Column('data_quality', sa.JSON(), nullable=False),
        sa.Column('market_snapshot', sa.JSON()),
        sa.UniqueConstraint('run_id', 'game_id', name='uq_prospective_run_game'),
    )


def downgrade():
    op.drop_table('prospective_forecasts')
    op.drop_table('prospective_runs')
    for name in ('publication_verified', 'parse_issues', 'entries_json', 'source_sha256', 'source_url'):
        op.drop_column('injury_reports', name)
    op.drop_constraint('uq_team_game_stats_game_team', 'team_game_stats', type_='unique')
