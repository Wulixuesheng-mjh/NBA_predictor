"""Preserve complete, traceable MLP input compatibility version identifiers."""
from alembic import op
import sqlalchemy as sa

revision='010_mlp_compat_version'
down_revision='009_shadow_observations'
branch_labels=None
depends_on=None


def upgrade():
    op.alter_column('predictions','model_version',existing_type=sa.String(20),type_=sa.String(128),existing_nullable=False)


def downgrade():
    count=op.get_bind().scalar(sa.text('SELECT count(*) FROM predictions WHERE length(model_version)>20'))
    if count:
        raise RuntimeError('Cannot shrink model_version without truncating existing identifiers; retain widened schema')
    op.alter_column('predictions','model_version',existing_type=sa.String(128),type_=sa.String(20),existing_nullable=False)
