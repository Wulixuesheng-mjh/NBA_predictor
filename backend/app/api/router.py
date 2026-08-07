"""API router aggregator."""

from fastapi import APIRouter

from app.api.teams import router as teams_router
from app.api.players import router as players_router
from app.api.games import router as games_router
from app.api.injuries import router as injuries_router
from app.api.predictions import router as predictions_router
from app.api.admin import router as admin_router

api_router = APIRouter()
api_router.include_router(teams_router, prefix="/teams", tags=["teams"])
api_router.include_router(players_router, prefix="/players", tags=["players"])
api_router.include_router(games_router, prefix="/games", tags=["games"])
api_router.include_router(injuries_router, prefix="/injuries", tags=["injuries"])
api_router.include_router(predictions_router, prefix="/predictions", tags=["predictions"])
api_router.include_router(admin_router, prefix="/admin", tags=["admin"])
