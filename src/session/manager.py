"""Session management via Redis."""

import json
from datetime import datetime
from typing import Any

import structlog
import redis.asyncio as redis

from src.config import get_settings
from src.models.state import SessionState, ConversationTurn, SessionEntities

logger = structlog.get_logger()
settings = get_settings()


class SessionManager:
    """Manage conversation sessions in Redis."""

    def __init__(self, client: redis.Redis | None = None):
        self.client = client
        self._prefix = "session:"

    async def connect(self) -> None:
        """Connect to Redis."""
        if self.client is None:
            self.client = redis.from_url(settings.redis_url)
            logger.info("session_manager_connected")

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self.client = None

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self._prefix}{session_id}"

    async def get_session(self, session_id: str) -> SessionState | None:
        """Get session from Redis.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if exists, None otherwise
        """
        if self.client is None:
            await self.connect()

        key = self._key(session_id)

        try:
            data = await self.client.get(key)
            if data is None:
                return None

            session_dict = json.loads(data)
            return SessionState.model_validate(session_dict)

        except Exception as e:
            logger.error("get_session_error", session_id=session_id, error=str(e))
            return None

    async def create_session(self, session_id: str) -> SessionState:
        """Create a new session.

        Args:
            session_id: Session identifier

        Returns:
            New SessionState
        """
        if self.client is None:
            await self.connect()

        session = SessionState(
            session_id=session_id,
            ttl_seconds=settings.session_ttl_seconds,
        )

        await self.save_session(session)
        logger.info("session_created", session_id=session_id)

        return session

    async def save_session(self, session: SessionState) -> None:
        """Save session to Redis.

        Args:
            session: SessionState to save
        """
        if self.client is None:
            await self.connect()

        session.last_activity = datetime.utcnow()
        key = self._key(session.session_id)

        try:
            data = session.model_dump_json()
            await self.client.setex(key, session.ttl_seconds, data)
            logger.debug("session_saved", session_id=session.session_id)

        except Exception as e:
            logger.error("save_session_error", session_id=session.session_id, error=str(e))
            raise

    async def get_or_create_session(self, session_id: str) -> SessionState:
        """Get existing session or create new one.

        Args:
            session_id: Session identifier

        Returns:
            SessionState (existing or new)
        """
        session = await self.get_session(session_id)
        if session is None:
            session = await self.create_session(session_id)
        return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        if self.client is None:
            await self.connect()

        key = self._key(session_id)

        try:
            result = await self.client.delete(key)
            deleted = result > 0
            if deleted:
                logger.info("session_deleted", session_id=session_id)
            return deleted

        except Exception as e:
            logger.error("delete_session_error", session_id=session_id, error=str(e))
            return False

    async def add_user_message(
        self,
        session_id: str,
        content: str,
    ) -> SessionState:
        """Add a user message to the session.

        Args:
            session_id: Session identifier
            content: User message content

        Returns:
            Updated SessionState
        """
        session = await self.get_or_create_session(session_id)
        session.add_user_turn(content)
        await self.save_session(session)
        return session

    async def add_assistant_response(
        self,
        session_id: str,
        content: str,
        result_ids: list[str] | None = None,
    ) -> SessionState:
        """Add an assistant response to the session.

        Args:
            session_id: Session identifier
            content: Assistant response content
            result_ids: Optional list of result document IDs

        Returns:
            Updated SessionState
        """
        session = await self.get_or_create_session(session_id)
        session.add_assistant_turn(content, result_ids)

        # Update previous query
        if session.conversation:
            for turn in reversed(session.conversation):
                if turn.role == "user":
                    session.previous_query = turn.content
                    break

        await self.save_session(session)
        return session

    async def update_entities(
        self,
        session_id: str,
        entities: dict[str, Any],
    ) -> SessionState:
        """Update tracked entities for a session.

        Args:
            session_id: Session identifier
            entities: Entity updates to apply

        Returns:
            Updated SessionState
        """
        session = await self.get_or_create_session(session_id)
        session.entities.update_from_filters(entities)
        await self.save_session(session)
        return session

    async def get_session_context(self, session_id: str) -> dict[str, Any]:
        """Get session context for search pipeline.

        Args:
            session_id: Session identifier

        Returns:
            Context dict with entities, previous results, etc.
        """
        session = await self.get_or_create_session(session_id)

        return {
            "session_id": session.session_id,
            "entities": session.entities.to_filters(),
            "previous_results": session.previous_results,
            "previous_query": session.previous_query,
            "conversation_length": len(session.conversation),
            "recent_conversation": [
                {"role": t.role, "content": t.content}
                for t in session.get_recent_conversation(3)
            ],
        }

    async def list_sessions(self, limit: int = 100) -> list[str]:
        """List active session IDs.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session IDs
        """
        if self.client is None:
            await self.connect()

        pattern = f"{self._prefix}*"
        cursor = 0
        session_ids = []

        while True:
            cursor, keys = await self.client.scan(cursor, match=pattern, count=100)
            for key in keys:
                session_id = key.decode().replace(self._prefix, "")
                session_ids.append(session_id)
                if len(session_ids) >= limit:
                    return session_ids
            if cursor == 0:
                break

        return session_ids
