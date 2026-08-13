/** Core TypeScript types — matches backend Pydantic schemas (Phase 2). */

// ── Team ──

export interface Team {
  id: number;
  nba_team_id: number;
  abbreviation: string;
  city: string;
  name: string;
  full_name: string;
  conference: string | null;
  division: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface TeamDetail extends Team {
  recent_record: string | null; // e.g. "7-3"
  avg_pts: number | null;
  avg_pts_allowed: number | null;
  net_rating: number | null;
}

export interface TeamRoster {
  team_id: number;
  team_name: string;
  players: PlayerBrief[];
  total: number;
}

// ── Player ──

export interface Player {
  id: number;
  nba_player_id: number;
  first_name: string | null;
  last_name: string | null;
  full_name: string | null;
  team_id: number | null;
  team_abbreviation?: string | null;
  jersey_number: string | null;
  position: string | null;
  created_at?: string;
}

export interface PlayerBrief {
  id: number;
  nba_player_id: number;
  full_name: string | null;
  position: string | null;
  jersey_number: string | null;
}

export interface PlayerSeasonStats {
  games_played: number;
  avg_pts: number;
  avg_reb: number;
  avg_ast: number;
  avg_stl: number;
  avg_blk: number;
  avg_tov: number;
  avg_minutes: number;
  fg_pct: number;
  fg3_pct: number;
  ft_pct: number;
}

export interface PlayerDetail extends Player {
  season_stats: PlayerSeasonStats | null;
  active_injuries: PlayerInjuryBrief[];
  team_abbreviation: string | null;
}

export interface PlayerInjuryBrief {
  id: number;
  injury_status: string;
  injury_type: string | null;
  description: string | null;
  confirmation: string;
  source: string;
  is_active: boolean;
}

// ── Game ──

export type GameStatus = "Scheduled" | "PreGame" | "InProgress" | "Final" | "Postponed" | "Cancelled";

export interface Game {
  id: number;
  nba_game_id: string;
  season: string;
  game_date: string;
  home_team_id: number;
  away_team_id: number;
  home_team_name: string | null;
  away_team_name: string | null;
  home_team_abbr: string | null;
  away_team_abbr: string | null;
  status: GameStatus;
  home_score: number | null;
  away_score: number | null;
  quarter: number | null;
  time_remaining: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface TeamGameStatsBrief {
  team_id: number;
  team_name: string | null;
  is_home: boolean;
  pts: number;
  reb: number;
  ast: number;
  stl: number;
  blk: number;
  tov: number;
  fgm: number;
  fga: number;
  fg3m: number;
  fg3a: number;
  ftm: number;
  fta: number;
  fg_pct: number | null;
  fg3_pct: number | null;
  ft_pct: number | null;
}

export interface PlayerGameStatsBrief {
  player_id: number;
  player_name: string | null;
  team_id: number;
  minutes: string | null;
  pts: number;
  reb: number;
  ast: number;
  stl: number;
  blk: number;
  tov: number;
  pf: number;
  fgm: number;
  fga: number;
  fg3m: number;
  fg3a: number;
  ftm: number;
  fta: number;
  plus_minus: number | null;
}

export interface GameDetail extends Game {
  home_team_name: string | null;
  away_team_name: string | null;
  home_team_abbr: string | null;
  away_team_abbr: string | null;
  team_stats: TeamGameStatsBrief[];
  player_stats: PlayerGameStatsBrief[];
}

export interface UpcomingGame {
  id: number;
  nba_game_id: string;
  game_date: string;
  home_team_id: number;
  away_team_id: number;
  home_team_name: string | null;
  away_team_name: string | null;
  home_team_abbr: string | null;
  away_team_abbr: string | null;
  status: string;
}

// ── Injury ──

export interface PlayerInjury {
  id: number;
  player_id: number;
  player_name: string | null;
  team_abbreviation: string | null;
  source: "nba_api" | "manual";
  confirmation: "pending" | "confirmed" | "modified";
  injury_status: string;
  injury_type: string | null;
  description: string | null;
  reported_date: string;
  expected_return: string | null;
  is_active: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface InjuryCreatePayload {
  player_id: number;
  injury_status: string;
  injury_type?: string | null;
  description?: string | null;
  expected_return?: string | null;
}

export interface InjuryUpdatePayload {
  injury_status?: string;
  injury_type?: string | null;
  description?: string | null;
  expected_return?: string | null;
  is_active?: boolean;
}

// ── Prediction ──

export interface ModelPredictionOutput {
  model_version: string;
  win_prob_home: number;
  win_prob_away: number;
  predicted_spread: number | null;
  spread_confidence: number | null;
  predicted_total: number | null;
  total_confidence: number | null;
}

export interface ConsensusOutput {
  win_agree: boolean;
  spread_direction_agree: boolean;
  total_agree: boolean;
  agreement_score: number;
}

export interface DataQualityOutput {
  home_games_available: number;
  away_games_available: number;
  warnings: string[];
}

export interface ConfidenceOutput {
  level: "high" | "medium" | "low" | "not_recommended";
  score: number;
  factors: Record<string, number>;
  label_cn: string;
}

export interface PredictionResponse {
  home_team_id: number;
  away_team_id: number;
  home_team_name: string | null;
  away_team_name: string | null;
  home_team_abbr: string | null;
  away_team_abbr: string | null;
  lookback_games: number;
  actual_games_home: number;
  actual_games_away: number;
  predicted_at: string;
  xgboost: ModelPredictionOutput | null;
  mlp: ModelPredictionOutput | null;
  consensus: ConsensusOutput | null;
  recommended: string;
  data_quality: DataQualityOutput;
  calibration_applied: boolean;
  confidence?: ConfidenceOutput | null;
  explanation?: ExplanationOutput | null;
  cover_over?: CoverOverOutput | null;
}

export interface ExplanationFeature {
  feature: string;
  display_name: string;
  shap_value: number;
  contribution: "positive" | "negative";
  direction: "favors_home" | "favors_away";
}

export interface ExplanationOutput {
  top_features: ExplanationFeature[];
  explained_model: string;
}

export interface PredictionRequest {
  home_team_id: number;
  away_team_id: number;
  lookback_games: number;
  spread_line?: number;
  total_line?: number;
}

export interface CoverOverExplanationFeature {
  feature: string;
  display_name: string;
  shap_value: number;
  contribution: "positive" | "negative";
  direction: string;
}

export interface CoverOverExplanation {
  cover: CoverOverExplanationFeature[] | null;
  over: CoverOverExplanationFeature[] | null;
}

export interface CoverOverOutput {
  model_version: string;
  spread_line: number | null;
  total_line: number | null;
  cover_prob_home: number;
  cover_prob_away: number;
  cover_direction: "home" | "away";
  cover_direction_cn: string;
  over_prob: number;
  under_prob: number;
  over_direction: "over" | "under";
  over_direction_cn: string;
  explanation: CoverOverExplanation | null;
}

export interface PredictionHistoryItem {
  id: number;
  game_id: number;
  model_name: string;
  model_version: string;
  predicted_at: string | null;
  win_prob_home: number;
  predicted_spread: number | null;
  predicted_total: number | null;
  lookback_games: number;
  home_team_name: string | null;
  away_team_name: string | null;
  game_date: string | null;
  has_result: boolean;
}

export interface PredictionResultOutput {
  prediction_id: number;
  actual_spread: number | null;
  actual_total: number | null;
  actual_winner: string | null;
  spread_correct: boolean | null;
  total_correct: boolean | null;
  win_correct: boolean | null;
  evaluated_at: string | null;
}

export interface PredictionDetail extends PredictionHistoryItem {
  win_prob_away: number;
  spread_confidence: number | null;
  total_confidence: number | null;
  calibrated: boolean;
  home_team_id: number | null;
  away_team_id: number | null;
  result: PredictionResultOutput | null;
}

// ── Admin ──

export interface CollectionLogItem {
  id: number;
  source: string;
  task_name: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  records_collected: number;
  error_message: string | null;
}

export interface SystemStats {
  teams_count: number;
  players_count: number;
  games_count: number;
  completed_games_count: number;
  active_injuries_count: number;
  pending_injuries_count: number;
  predictions_count: number;
  evaluated_predictions_count: number;
  last_collection: string | null;
  active_model_version: string | null;
  model_last_trained: string | null;
}

// ── API Response wrappers (backend returns flat with meta, some return data directly) ──

export interface ListResponse<T> {
  // Matches backend list response patterns
  [key: string]: T[] | number; // e.g. { teams: [...], total: 50 }
  total: number;
}

export interface GameListResponse {
  games: Game[];
  total: number;
  offset: number;
  limit: number;
}

export interface InjuryListResponse {
  injuries: PlayerInjury[];
  total: number;
  pending_count: number;
}

export interface PredictionHistoryListResponse {
  predictions: PredictionHistoryItem[];
  total: number;
  offset: number;
  limit: number;
}

// ── Accuracy Trends ──

export interface AccuracyTrendPoint {
  date: string;
  win_accuracy: number | null;
  spread_mae: number | null;
  total_mae: number | null;
  sample_count: number;
  window_days: number;
}

export interface AccuracyTrendSummary {
  win_accuracy: number;
  spread_mae: number | null;
  total_mae: number | null;
  total_evaluated: number;
}

export interface AccuracyTrendResponse {
  trends: AccuracyTrendPoint[];
  overall: AccuracyTrendSummary;
  model_version: string | null;
}
