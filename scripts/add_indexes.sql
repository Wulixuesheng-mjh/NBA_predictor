-- ── Performance indexes for NBA Predictor ──────────────
-- Enables full-dataset feature building (30K+ games)
-- Run: docker compose exec postgres psql -U nba_admin -d nba_predictor -f /scripts/add_indexes.sql

-- Games table — most heavily queried
CREATE INDEX IF NOT EXISTS idx_games_game_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON games(home_team_id);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON games(away_team_id);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);

-- Team game stats — joined with games for team features
CREATE INDEX IF NOT EXISTS idx_team_game_stats_game ON team_game_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_team_game_stats_team ON team_game_stats(team_id);

-- Player game stats — player form features
CREATE INDEX IF NOT EXISTS idx_game_player_stats_game ON game_player_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_game_player_stats_player ON game_player_stats(player_id);

-- Composite: most common query pattern in feature building
CREATE INDEX IF NOT EXISTS idx_games_home_date ON games(home_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_games_away_date ON games(away_team_id, game_date);

-- Players — roster lookups and search
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_players_name ON players(full_name);

-- Predictions — history queries and feedback loop
CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_at ON predictions(predicted_at);

-- Prediction results — accuracy trend computations
CREATE INDEX IF NOT EXISTS idx_prediction_results_eval ON prediction_results(evaluated_at);

-- Data collection log — status dashboard
CREATE INDEX IF NOT EXISTS idx_collection_log_started ON data_collection_log(started_at);
CREATE INDEX IF NOT EXISTS idx_collection_log_status ON data_collection_log(status);
