"""Batches with runs started from code (qaboard.start_run()) say so, in data.sdk: projects with some get the Runs page,
found with a small partial index (backend/runs_list.py has_runs_from_code). Also declared in the models.

The server marks new batches when it gets their runs (api/outputs.py). Here, the batches that already have such runs:
start_run() shipped in November 2026, we only read the runs created since October (outputs.created_date is indexed),
not the whole table.

Revision ID: c4e6a8b0d2f3
Revises: e3a5c7d9f1b2
Create Date: 2026-11-17 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c4e6a8b0d2f3'
down_revision = 'e3a5c7d9f1b2'
branch_labels = None
depends_on = None


name = 'ix_batches_sdk'


def upgrade():
  op.execute(sa.text("""
    UPDATE batches SET data = coalesce(data, '{}'::jsonb) || '{"sdk": true}'::jsonb
    WHERE id IN (
      SELECT DISTINCT batch_id FROM outputs
      WHERE created_date >= '2026-10-01' AND (data->>'sdk') IS NOT NULL
    )
    AND NOT (coalesce(data, '{}'::jsonb) ? 'sdk')
  """))
  # batches is small next to outputs, still: CONCURRENTLY, and we build again an index a failed build left invalid
  with op.get_context().autocommit_block():
    is_invalid = op.get_bind().execute(sa.text(
      "SELECT 1 FROM pg_index JOIN pg_class ON pg_class.oid = pg_index.indexrelid WHERE pg_class.relname = :name AND NOT pg_index.indisvalid"
    ), {"name": name}).first()
    if is_invalid:
      op.drop_index(name, table_name='batches', postgresql_concurrently=True, if_exists=True)
    op.create_index(name, 'batches', ['ci_commit_id'], postgresql_where=sa.text("data ? 'sdk'"),
                    postgresql_concurrently=True, if_not_exists=True)


def downgrade():
  with op.get_context().autocommit_block():
    op.drop_index(name, table_name='batches', postgresql_concurrently=True, if_exists=True)
