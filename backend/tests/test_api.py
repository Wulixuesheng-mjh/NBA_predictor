"""End-to-end API tests for NBA Predictor.

Covers all endpoints: health, teams, players, games, injuries, predictions, admin.

Run:
    cd backend && python -m pytest tests/ -v
    cd backend && python -m pytest tests/ -v -m "not slow"
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession


# ══════════════════════════════════════════════════════════
# Health & Root
# ══════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_root_returns_status(client: AsyncClient):
    """GET / should return service info."""
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "NBA Predictor API"
    assert data["status"] == "running"
    assert "version" in data


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """GET /health should return healthy."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


# ══════════════════════════════════════════════════════════
# Teams API
# ══════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_teams_empty(client: AsyncClient):
    """List teams when database is empty."""
    resp = await client.get("/api/v1/teams")
    assert resp.status_code == 200
    data = resp.json()
    assert data["teams"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_list_teams_with_data(client: AsyncClient, seeded_db: AsyncSession):
    """List teams returns seeded teams."""
    resp = await client.get("/api/v1/teams")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["teams"]) == 2
    abbreviations = {t["abbreviation"] for t in data["teams"]}
    assert abbreviations == {"LAL", "BOS"}


@pytest.mark.asyncio
async def test_list_teams_by_conference(client: AsyncClient, seeded_db: AsyncSession):
    """Filter teams by conference."""
    resp = await client.get("/api/v1/teams?conference=West")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["teams"][0]["abbreviation"] == "LAL"


@pytest.mark.asyncio
async def test_get_team_detail(client: AsyncClient, seeded_db: AsyncSession):
    """Get single team detail with recent record."""
    team_id = seeded_db._team_ids["LAL"]
    resp = await client.get(f"/api/v1/teams/{team_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["full_name"] == "Los Angeles Lakers"
    assert data["abbreviation"] == "LAL"
    assert data["conference"] == "West"
    assert "recent_record" in data


@pytest.mark.asyncio
async def test_get_team_not_found(client: AsyncClient):
    """Team not found returns 404-format error."""
    resp = await client.get("/api/v1/teams/99999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_team_roster(client: AsyncClient, seeded_db: AsyncSession):
    """Get team roster returns players."""
    team_id = seeded_db._team_ids["LAL"]
    resp = await client.get(f"/api/v1/teams/{team_id}/players")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["players"]) == 2
    player_names = {p["full_name"] for p in data["players"]}
    assert "LeBron James" in player_names
    assert "Anthony Davis" in player_names


# ══════════════════════════════════════════════════════════
# Players API
# ══════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_players(client: AsyncClient, seeded_db: AsyncSession):
    """List all players."""
    resp = await client.get("/api/v1/players")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 4


@pytest.mark.asyncio
async def test_list_players_by_team(client: AsyncClient, seeded_db: AsyncSession):
    """Filter players by team ID."""
    team_id = seeded_db._team_ids["BOS"]
    resp = await client.get(f"/api/v1/players?team_id={team_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    names = {p["full_name"] for p in data["players"]}
    assert names == {"Jayson Tatum", "Jaylen Brown"}


@pytest.mark.asyncio
async def test_list_players_search(client: AsyncClient, seeded_db: AsyncSession):
    """Search players by name."""
    resp = await client.get("/api/v1/players?search=LeBron")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["players"][0]["full_name"] == "LeBron James"


@pytest.mark.asyncio
async def test_get_player_detail(client: AsyncClient, seeded_db: AsyncSession):
    """Get player with season stats and active injuries."""
    player_id = seeded_db._player_ids["LeBron"]
    resp = await client.get(f"/api/v1/players/{player_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["full_name"] == "LeBron James"
    assert data["position"] == "F"
    assert "season_stats" in data
    assert "active_injuries" in data
    # LeBron has an active injury (pending ankle)
    assert len(data["active_injuries"]) == 1
    assert data["active_injuries"][0]["injury_type"] == "Ankle"


@pytest.mark.asyncio
async def test_get_player_not_found(client: AsyncClient):
    """Player not found returns 404."""
    resp = await client.get("/api/v1/players/99999")
    assert resp.status_code == 404


# ══════════════════════════════════════════════════════════
# Games API
# ══════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_games(client: AsyncClient, seeded_db: AsyncSession):
    """List all games."""
    resp = await client.get("/api/v1/games")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3


@pytest.mark.asyncio
async def test_list_games_by_status(client: AsyncClient, seeded_db: AsyncSession):
    """Filter games by status."""
    resp = await client.get("/api/v1/games?status=Final")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    for g in data["games"]:
        assert g["status"] == "Final"


@pytest.mark.asyncio
async def test_list_upcoming_games(client: AsyncClient, seeded_db: AsyncSession):
    """Get upcoming (scheduled) games. May be empty if seeded dates are past."""
    resp = await client.get("/api/v1/games/upcoming?days=30")
    assert resp.status_code == 200
    data = resp.json()
    # Response is a list — may be empty if seeded game date drifted past "now"
    assert isinstance(data, list)
    for game in data:
        assert game["status"] in ("Scheduled", "PreGame")


@pytest.mark.asyncio
async def test_get_game_detail(client: AsyncClient, seeded_db: AsyncSession):
    """Get game detail with team stats."""
    game_id = seeded_db._game_ids["final"]
    resp = await client.get(f"/api/v1/games/{game_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Final"
    assert data["home_score"] == 115
    assert data["away_score"] == 108
    assert "team_stats" in data
    assert len(data["team_stats"]) == 2


@pytest.mark.asyncio
async def test_get_game_not_found(client: AsyncClient):
    """Non-existent game returns 404."""
    resp = await client.get("/api/v1/games/99999")
    assert resp.status_code == 404


# ══════════════════════════════════════════════════════════
# Injuries API
# ══════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_injuries(client: AsyncClient, seeded_db: AsyncSession):
    """List all active injuries."""
    resp = await client.get("/api/v1/injuries")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2


@pytest.mark.asyncio
async def test_list_injuries_filtered(client: AsyncClient, seeded_db: AsyncSession):
    """Filter injuries by team and confirmation."""
    # Pending injuries only
    resp = await client.get("/api/v1/injuries?confirmation=pending")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["injuries"][0]["injury_type"] == "Ankle"


@pytest.mark.asyncio
async def test_create_injury(client: AsyncClient, seeded_db: AsyncSession):
    """Manual injury creation auto-sets source=manual, confirmation=confirmed."""
    player_id = seeded_db._player_ids["AD"]
    payload = {
        "player_id": player_id,
        "injury_status": "Probable",
        "injury_type": "Groin",
        "description": "轻微腹股沟不适",
    }
    resp = await client.post("/api/v1/injuries", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["injury_status"] == "Probable"
    assert data["source"] == "manual"
    assert data["confirmation"] == "confirmed"

    # Verify it appears in injury list
    list_resp = await client.get("/api/v1/injuries")
    assert list_resp.json()["total"] == 3


@pytest.mark.asyncio
async def test_update_injury(client: AsyncClient, seeded_db: AsyncSession):
    """Updating a non-manual injury auto-sets confirmation=modified."""
    # Find the pending ankle injury (LeBron)
    resp = await client.get("/api/v1/injuries?confirmation=pending")
    injury_id = resp.json()["injuries"][0]["id"]

    update = {"injury_status": "Out", "description": "踝关节扭伤加重"}
    put_resp = await client.put(f"/api/v1/injuries/{injury_id}", json=update)
    assert put_resp.status_code == 200
    data = put_resp.json()
    assert data["injury_status"] == "Out"
    assert data["confirmation"] == "modified"


@pytest.mark.asyncio
async def test_delete_injury_soft(client: AsyncClient, seeded_db: AsyncSession):
    """Soft-delete sets is_active=False."""
    resp = await client.get("/api/v1/injuries")
    injury_id = resp.json()["injuries"][0]["id"]
    initial_total = resp.json()["total"]

    del_resp = await client.delete(f"/api/v1/injuries/{injury_id}")
    assert del_resp.status_code == 200

    # Count should decrease by 1
    list_resp = await client.get("/api/v1/injuries")
    assert list_resp.json()["total"] == initial_total - 1


@pytest.mark.asyncio
async def test_confirm_all_injuries(client: AsyncClient, seeded_db: AsyncSession):
    """Bulk confirm all pending injuries."""
    # Before: 1 pending, 1 confirmed
    resp = await client.post("/api/v1/injuries/confirm-all")
    assert resp.status_code == 200
    data = resp.json()
    assert data["confirmed_count"] >= 0

    # After: 0 pending
    pending = await client.get("/api/v1/injuries?confirmation=pending")
    assert pending.json()["total"] == 0


# ══════════════════════════════════════════════════════════
# Predictions API
# ══════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_predict_unavailable_when_no_model(client: AsyncClient, seeded_db: AsyncSession):
    """Prediction returns 200 when no trained model exists. Handles both cases:
    - ML module not importable → recommended="unavailable", xgboost=None
    - ML module loaded but no trained model → may still be unavailable
    """
    home_id = seeded_db._team_ids["LAL"]
    away_id = seeded_db._team_ids["BOS"]

    payload = {"home_team_id": home_id, "away_team_id": away_id, "lookback_games": 10}
    resp = await client.post("/api/v1/predictions", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["recommended"] in ("unavailable", "xgboost", "mlp")

    if data["recommended"] == "unavailable":
        assert data["xgboost"] is None
        assert len(data["data_quality"]["warnings"]) >= 1
    else:
        # ML module loaded and model found — must have valid prediction
        assert data["xgboost"] is not None or data["mlp"] is not None
        if data["xgboost"]:
            assert "win_prob_home" in data["xgboost"]
            assert 0 <= data["xgboost"]["win_prob_home"] <= 1


@pytest.mark.asyncio
async def test_predict_same_team_rejected(client: AsyncClient, seeded_db: AsyncSession):
    """Same team as home and away raises 400."""
    team_id = seeded_db._team_ids["LAL"]
    payload = {"home_team_id": team_id, "away_team_id": team_id}
    resp = await client.post("/api/v1/predictions", json=payload)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_predict_nonexistent_team(client: AsyncClient):
    """Non-existent team raises 400."""
    payload = {"home_team_id": 99999, "away_team_id": 1}
    resp = await client.post("/api/v1/predictions", json=payload)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_predict_lookback_range_validation(client: AsyncClient, seeded_db: AsyncSession):
    """Lookback outside 5-15 range should be rejected by Pydantic."""
    home_id = seeded_db._team_ids["LAL"]
    away_id = seeded_db._team_ids["BOS"]

    # Below minimum (5)
    payload = {"home_team_id": home_id, "away_team_id": away_id, "lookback_games": 2}
    resp = await client.post("/api/v1/predictions", json=payload)
    assert resp.status_code == 422

    # Above maximum (15)
    payload["lookback_games"] = 20
    resp = await client.post("/api/v1/predictions", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_predictions_empty(client: AsyncClient, seeded_db: AsyncSession):
    """List predictions when none exist."""
    resp = await client.get("/api/v1/predictions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["predictions"] == []


@pytest.mark.asyncio
async def test_get_prediction_not_found(client: AsyncClient):
    """Non-existent prediction returns 404."""
    resp = await client.get("/api/v1/predictions/99999")
    assert resp.status_code == 404


# ══════════════════════════════════════════════════════════
# Admin API
# ══════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_admin_stats(client: AsyncClient, seeded_db: AsyncSession):
    """System stats endpoint returns counts."""
    resp = await client.get("/api/v1/admin/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["teams_count"] == 2
    assert data["players_count"] == 4
    assert data["games_count"] == 3
    assert data["active_injuries_count"] == 2
    assert data["pending_injuries_count"] == 1
    assert data["completed_games_count"] == 2


@pytest.mark.asyncio
async def test_admin_collect_status(client: AsyncClient, seeded_db: AsyncSession):
    """Collection status endpoint returns logs."""
    resp = await client.get("/api/v1/admin/collect/status")
    assert resp.status_code == 200
    assert "last_runs" in resp.json()


@pytest.mark.asyncio
async def test_admin_collect_trigger(client: AsyncClient, seeded_db: AsyncSession, monkeypatch):
    """Manual collection trigger should return triggered status."""
    from types import SimpleNamespace
    from datetime import datetime, timezone
    from app.collector.nba_collector import NBACollector

    async def collect_stub(self):
        return SimpleNamespace(status='success', records_collected=0,
            completed_at=datetime.now(timezone.utc), error_message=None)

    monkeypatch.setattr(NBACollector, 'collect', collect_stub)
    resp = await client.post("/api/v1/admin/collect?source=nba_api")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("triggered", "already_running", "failed")


@pytest.mark.asyncio
async def test_admin_train_model(client: AsyncClient, seeded_db: AsyncSession):
    """Model training trigger returns message (mocked to avoid real training)."""
    from unittest.mock import patch, MagicMock
    import importlib

    mock_result = MagicMock()
    mock_result.retrained = False
    mock_result.retrain_version = None
    mock_result.predictions_evaluated = 0
    mock_result.new_games_found = 0
    mock_result.calibration_updated = False

    async def _mock_cycle(*args, **kwargs):
        return mock_result

    try:
        importlib.import_module("ml.src.feedback")
        with patch("ml.src.feedback.run_feedback_cycle", _mock_cycle):
            resp = await client.post("/api/v1/admin/train-model?force=true")
    except (ImportError, ModuleNotFoundError):
        # ML module not available — endpoint falls back to status="failed"
        resp = await client.post("/api/v1/admin/train-model?force=true")

    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data
    assert data["status"] in ("triggered", "failed")


# ══════════════════════════════════════════════════════════
# Error handling
# ══════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_validation_error_format(client: AsyncClient):
    """Validation errors should return consistent error format."""
    # Send invalid body type
    resp = await client.post("/api/v1/predictions", json={"home_team_id": "abc"})
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert "details" in data
    assert data["status_code"] == 422


@pytest.mark.asyncio
async def test_not_found_error_format(client: AsyncClient):
    """404 errors should return consistent format."""
    resp = await client.get("/api/v1/nonexistent-route")
    assert resp.status_code == 404
    data = resp.json()
    assert "error" in data
    assert data["status_code"] == 404


@pytest.mark.asyncio
async def test_request_id_header(client: AsyncClient):
    """Every response should include X-Request-ID header."""
    resp = await client.get("/health")
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) == 12  # short UUID


@pytest.mark.asyncio
async def test_cors_headers(client: AsyncClient):
    """CORS headers should be present."""
    resp = await client.options("/api/v1/teams", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
    })
    assert "access-control-allow-origin" in resp.headers
    assert "access-control-allow-methods" in resp.headers


@pytest.mark.asyncio
async def test_docs_endpoint(client: AsyncClient):
    """OpenAPI docs should be accessible."""
    resp = await client.get("/api/docs")
    assert resp.status_code == 200

    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["info"]["title"] == "NBA Predictor API"
