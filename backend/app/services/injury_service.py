"""Injury service — business logic for injury CRUD and confirmation workflow."""

import logging
from datetime import datetime, timezone
from app.utils.logging import get_logger

from sqlalchemy import select, func, and_, update, or_
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
        game_id: int | None = None,
    ) -> tuple[list[dict], int, int]:
        """List injuries with optional filters.

        Returns:
            (injury_dicts, total_count, pending_count)
        """
        query = (
            select(PlayerInjury, Player.full_name, Team.abbreviation)
            .join(Player, PlayerInjury.player_id == Player.id)
            .outerjoin(Team, func.coalesce(PlayerInjury.team_id, Player.team_id) == Team.id)
        )
        count_query = select(func.count(PlayerInjury.id))
        pending_count_query = select(func.count(PlayerInjury.id)).where(
            and_(PlayerInjury.confirmation == "pending", PlayerInjury.is_active == True)
        )
        if game_id is not None:
            query = query.where(PlayerInjury.game_id == game_id)
            count_query = count_query.where(PlayerInjury.game_id == game_id)
            pending_count_query = pending_count_query.where(PlayerInjury.game_id == game_id)

        if team_id is not None:
            team_filter = or_(PlayerInjury.team_id == team_id,
                and_(PlayerInjury.team_id.is_(None), Player.team_id == team_id))
            query = query.where(team_filter)
            count_query = count_query.join(Player, PlayerInjury.player_id == Player.id).where(
                team_filter
            )
            pending_count_query = pending_count_query.join(
                Player, PlayerInjury.player_id == Player.id
            ).where(team_filter)

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
                "game_id": inj.game_id,
                "team_id": inj.team_id,
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
        game_id: int | None = None,
        team_id: int | None = None,
        published_at: datetime | None = None,
    ) -> PlayerInjury:
        """Create a manual injury record (auto-confirmed, source='manual')."""
        if game_id is not None:
            from app.models.game import Game
            game = await session.get(Game, game_id)
            player = await session.get(Player, player_id)
            team_id = team_id or (player.team_id if player else None)
            if game is None or team_id not in (game.home_team_id, game.away_team_id):
                raise ValueError('Manual injury must bind a team participating in the target game')
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
            game_id=game_id,
            team_id=team_id,
            published_at=published_at,
            observed_at=datetime.now(timezone.utc),
        )
        session.add(injury)
        await session.flush()
        await InjuryService.archive_bound_snapshot(session, game_id, team_id)
        from app.services.data_quality_service import bump_data_revision
        await bump_data_revision(session, f"manual-injury-create:{injury.id}")
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
        await InjuryService.archive_bound_snapshot(session, injury.game_id, injury.team_id)
        from app.services.data_quality_service import bump_data_revision
        await bump_data_revision(session, f"injury-update:{injury.id}")
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
        await InjuryService.archive_bound_snapshot(session, injury.game_id, injury.team_id)
        from app.services.data_quality_service import bump_data_revision
        await bump_data_revision(session, f"injury-delete:{injury.id}")
        logger.info("Deactivated injury #%d", injury_id)
        return True

    @staticmethod
    async def archive_bound_snapshot(session, game_id, team_id, *, complete=False):
        """Append the current game-bound manual payload; never rewrite a PDF."""
        if game_id is None or team_id is None:
            return None
        from app.models.injury_report import InjuryReport
        now = datetime.now(timezone.utc)
        official = await session.scalar(select(InjuryReport).where(
            InjuryReport.game_id == game_id, InjuryReport.team_id == team_id,
            InjuryReport.source != 'manual', InjuryReport.published_at <= now,
            InjuryReport.observed_at <= now, InjuryReport.entries_json.isnot(None),
        ).order_by(InjuryReport.published_at.desc(),InjuryReport.observed_at.desc()).limit(1))
        rows = (await session.execute(select(PlayerInjury).where(
            PlayerInjury.game_id == game_id, PlayerInjury.team_id == team_id,
            PlayerInjury.is_active.is_(True),
        ).order_by(PlayerInjury.updated_at.asc(), PlayerInjury.id.asc()))).scalars().all()
        entries = {e['player_id']: dict(e) for e in (official.entries_json or [])
                   if e.get('player_id') is not None} if official else {}
        entries.update({r.player_id: dict(player_id=r.player_id, status=r.injury_status,
                                    reason=r.description or r.injury_type or '') for r in rows})
        unresolved = [issue for issue in (official.parse_issues or [])
                      if issue != 'not_yet_submitted'] if official else []
        if unresolved:
            complete = False
        valid_statuses = {'Out', 'Doubtful', 'Questionable', 'Probable', 'Available'}
        if any(row['status'] not in valid_statuses for row in entries.values()):
            complete = False
        report = InjuryReport(game_id=game_id, team_id=team_id, source='manual',
            published_at=now, observed_at=now, publication_verified=False,
            entries_json=list(entries.values()), is_complete=complete,
            parse_issues=[] if complete else unresolved+['manual_coverage_not_confirmed'])
        session.add(report)
        await session.flush()
        return report

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
        if count:
            from app.services.data_quality_service import bump_data_revision
            await bump_data_revision(session, f"injury-confirm-all:{count}")
        logger.info("Confirmed %d pending injuries", count)
        return count
