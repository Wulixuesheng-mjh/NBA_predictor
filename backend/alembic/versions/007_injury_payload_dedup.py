"""Deduplicate immutable source payloads even under concurrent ingestion."""
from alembic import op

revision = '007_injury_payload_dedup'
down_revision = '006_snapshot_prospective'
branch_labels = None
depends_on = None


def upgrade():
    op.create_unique_constraint('uq_injury_report_payload','injury_reports',
                                ['game_id','team_id','source_sha256'])


def downgrade():
    op.drop_constraint('uq_injury_report_payload','injury_reports',type_='unique')
