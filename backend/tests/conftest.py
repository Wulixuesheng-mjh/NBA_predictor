"""Pytest fixtures and seed data helpers for backend tests.

Uses SQLite in-memory database — no external PostgreSQL needed.
"""

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_engine():
    """Create test database engine with all tables."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for a test."""
    test_session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with test_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP test client with DB override."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", follow_redirects=True) as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Seed data fixtures ──────────────────────────────────

@pytest_asyncio.fixture
async def seeded_db(db_session: AsyncSession) -> AsyncSession:
    """Pre-seed the database with teams, players, games, and injuries."""
    from app.models.team import Team
    from app.models.player import Player
    from app.models.game import Game
    from app.models.team_game_stats import TeamGameStats
    from app.models.player_injuries import PlayerInjury

    # Teams
    lakers = Team(
        nba_team_id=1610612747, full_name="Los Angeles Lakers",
        abbreviation="LAL", city="Los Angeles", name="Lakers",
        conference="West", division="Pacific",
    )
    celtics = Team(
        nba_team_id=1610612738, full_name="Boston Celtics",
        abbreviation="BOS", city="Boston", name="Celtics",
        conference="East", division="Atlantic",
    )
    db_session.add_all([lakers, celtics])
    await db_session.flush()

    # Players
    players = [
        Player(nba_player_id=2544, full_name="LeBron James", position="F",
               team_id=lakers.id, jersey_number="23"),
        Player(nba_player_id=203076, full_name="Anthony Davis", position="F-C",
               team_id=lakers.id, jersey_number="3"),
        Player(nba_player_id=1628369, full_name="Jayson Tatum", position="F",
               team_id=celtics.id, jersey_number="0"),
        Player(nba_player_id=1627759, full_name="Jaylen Brown", position="G-F",
               team_id=celtics.id, jersey_number="7"),
    ]
    db_session.add_all(players)
    await db_session.flush()

    # Games
    games = [
        Game(nba_game_id="0022500001", season="2025-26",
             game_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
             home_team_id=lakers.id, away_team_id=celtics.id,
             home_score=115, away_score=108, status="Final"),
        Game(nba_game_id="0022500002", season="2025-26",
             game_date=datetime(2026, 7, 31, tzinfo=timezone.utc),
             home_team_id=lakers.id, away_team_id=celtics.id,
             status="Scheduled"),
        Game(nba_game_id="0022500003", season="2025-26",
             game_date=datetime(2026, 6, 1, tzinfo=timezone.utc),
             home_team_id=lakers.id, away_team_id=celtics.id,
             home_score=102, away_score=110, status="Final"),
    ]
    db_session.add_all(games)
    await db_session.flush()

    # Team game stats
    db_session.add(TeamGameStats(
        game_id=games[0].id, team_id=lakers.id, is_home=True,
        pts=115, reb=45, ast=28, stl=8, blk=5, tov=12,
        fgm=42, fga=88, fg3m=12, fg3a=33, ftm=19, fta=23,
    ))
    db_session.add(TeamGameStats(
        game_id=games[0].id, team_id=celtics.id, is_home=False,
        pts=108, reb=42, ast=24, stl=6, blk=3, tov=14,
        fgm=38, fga=86, fg3m=11, fg3a=33, ftm=21, fta=27,
    ))
    await db_session.flush()

    # Injuries
    db_session.add(PlayerInjury(
        player_id=players[0].id, source="nba_api",
        confirmation="pending", injury_status="Questionable",
        injury_type="Ankle", description="左脚踝扭伤",
        reported_date=datetime(2026, 7, 28, tzinfo=timezone.utc),
        is_active=True,
    ))
    db_session.add(PlayerInjury(
        player_id=players[2].id, source="manual",
        confirmation="confirmed", injury_status="Out",
        injury_type="Knee", description="右膝酸痛",
        reported_date=datetime(2026, 7, 27, tzinfo=timezone.utc),
        is_active=True,
    ))
    await db_session.flush()

    # Store IDs on db_session for tests to access
    db_session._team_ids = {"LAL": lakers.id, "BOS": celtics.id}
    db_session._player_ids = {"LeBron": players[0].id, "AD": players[1].id,
                               "Tatum": players[2].id, "Brown": players[3].id}
    db_session._game_ids = {"final": games[0].id, "scheduled": games[1].id, "final2": games[2].id}

    await db_session.commit()
    return db_session
