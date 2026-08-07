"""Injury service — business logic for injury CRUD and confirmation workflow."""

import logging
from datetime import datetime, timezone
from app.utils.logging import get_logger

from sqlalchemy import select, func, and_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.player_injuries import PlayerInjury
from app.models.player import Player
from app.models.team import Team

logger = logging.getLogger(__name__)


class InjuryService:
    """Service for injury record management with confirmation workflow."""

    @staticmethod
    async def list_injuries(
        session: AsyncSession,
        team_id: int | None = None,
        confirmation: str | None = None,
        is_active: bool = True,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[dict], int, int]:
        """List injuries with optional filters.

        Returns:
            (injury_dicts, total_count, pending_count)
        """
        query = (
            select(PlayerInjury, Player.full_name, Team.abbreviation)
            .join(Player, PlayerInjury.player_id == Player.id)
            .outerjoin(Team, Player.team_id == Team.id)
        )
        count_query = select(func.count(PlayerInjury.id))
        pending_count_query = select(func.count(PlayerInjury.id)).where(
            and_(PlayerInjury.confirmation == "pending", PlayerInjury.is_active == True)
        )

        if team_id is not None:
            query = query.where(Player.team_id == team_id)
            count_query = count_query.join(Player, PlayerInjury.player_id == Player.id).where(
                Player.team_id == team_id
            )
            pending_count_query = pending_count_query.join(
                Player, PlayerInjury.player_id == Player.id
            ).where(Player.team_id == team_id)

        if confirmation:
            query = query.where(PlayerInjury.confirmation == confirmation)
            count_query = count_query.where(PlayerInjury.confirmation == confirmation)

        query = query.where(PlayerInjury.is_active == is_active)
        count_query = count_query.where(PlayerInjury.is_active == is_active)

        query = query.order_by(PlayerInjury.reported_date.desc()).offset(offset).limit(limit)

        result = await session.execute(query)
        rows = result.all()

        count_result = await session.execute(count_query)
        total = count_result.scalar() or 0

        pending_result = await session.execute(pending_count_query)
        pending_count = pending_result.scalar() or 0

        injuries = []
        for inj, player_name, team_abbr in rows:
            injuries.append({
                "id": inj.id,
                "player_id": inj.player_id,
                "player_name": player_name,
                "team_abbreviation": team_abbr,
                "source": inj.source,
                "confirmation": inj.confirmation,
                "injury_status": inj.injury_status,
                "injury_type": inj.injury_type,
                "description": inj.description,
                "reported_date": inj.reported_date,
                "expected_return": inj.expected_return,
                "is_active": inj.is_active,
                "created_at": inj.created_at,
                "updated_at": inj.updated_at,
            })

        return injuries, total, pending_count

    @staticmethod
    async def get_injury(
        session: AsyncSession, injury_id: int
    ) -> PlayerInjury | None:
        """Get a single injury record."""
        query = select(PlayerInjury).where(PlayerInjury.id == injury_id)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def create_injury(
        session: AsyncSession,
        player_id: int,
        injury_status: str,
        injury_type: str | None = None,
        description: str | None = None,
        expected_return: datetime | None = None,
    ) -> PlayerInjury:
        """Create a manual injury record (auto-confirmed, source='manual')."""
        injury = PlayerInjury(
            player_id=player_id,
            source="manual",
            confirmation="confirmed",
            injury_status=injury_status,
            injury_type=injury_type,
            description=description,
            reported_date=datetime.now(timezone.utc),
            expected_return=expected_return,
            is_active=True,
        )
        session.add(injury)
        await session.flush()
        await session.refresh(injury)
        logger.info("Created manual injury #%d for player %d", injury.id, player_id)
        return injury

    @staticmethod
    async def update_injury(
        session: AsyncSession,
        injury_id: int,
        injury_status: str | None = None,
        injury_type: str | None = None,
        description: str | None = None,
        expected_return: datetime | None = None,
        is_active: bool | None = None,
    ) -> PlayerInjury | None:
        """Update an injury record. Sets confirmation to 'modified' unless source is 'manual'."""
        injury = await InjuryService.get_injury(session, injury_id)
        if injury is None:
            return None

        if injury_status is not None:
            injury.injury_status = injury_status
        if injury_type is not None:
            injury.injury_type = injury_type
        if description is not None:
            injury.description = description
        if expected_return is not None:
            injury.expected_return = expected_return
        if is_active is not None:
            injury.is_active = is_active

        # Only mark as modified if not manual
        if injury.source != "manual":
            injury.confirmation = "modified"

        injury.updated_at = datetime.now(timezone.utc)
        await session.flush()
        await session.refresh(injury)
        logger.info("Updated injury #%d (confirmation=%s)", injury.id, injury.confirmation)
        return injury

    @staticmethod
    async def delete_injury(
        session: AsyncSession, injury_id: int
    ) -> bool:
        """Soft-delete: set is_active=False."""
        injury = await InjuryService.get_injury(session, injury_id)
        if injury is None:
            return False
        injury.is_active = False
        injury.updated_at = datetime.now(timezone.utc)
        await session.flush()
        logger.info("Deactivated injury #%d", injury_id)
        return True

    @staticmethod
    async def confirm_all(
        session: AsyncSession
    ) -> int:
        """One-click confirm all pending injuries. Returns count of confirmed records."""
        pending_query = (
            select(PlayerInjury)
            .where(
                and_(
                    PlayerInjury.confirmation == "pending",
                    PlayerInjury.is_active == True,
                )
            )
        )
        result = await session.execute(pending_query)
        pending_injuries = result.scalars().all()

        now = datetime.now(timezone.utc)
        for inj in pending_injuries:
            inj.confirmation = "confirmed"
            inj.updated_at = now

        count = len(pending_injuries)
        await session.flush()
        logger.info("Confirmed %d pending injuries", count)
        return count
