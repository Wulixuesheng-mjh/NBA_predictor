#!/bin/bash
# First-time database initialization script.
# This runs inside the postgres container on first startup.

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create extensions
    CREATE EXTENSION IF NOT EXISTS "pg_trgm";

    -- Verify database is ready
    SELECT 'NBA Predictor database initialized successfully' AS status;
EOSQL
