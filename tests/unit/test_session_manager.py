"""Tests for Redis session manager."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.session.manager import SessionManager
from src.models.state import SessionState, SessionEntities


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.setex = AsyncMock()
        client.delete = AsyncMock(return_value=1)
        client.scan = AsyncMock(return_value=(0, []))
        client.close = AsyncMock()
        return client

    @pytest.fixture
    def manager(self, mock_redis):
        """Create a SessionManager with mock Redis."""
        mgr = SessionManager(client=mock_redis)
        return mgr

    @pytest.mark.asyncio
    async def test_create_session(self, manager, mock_redis):
        """Test creating a new session."""
        session = await manager.create_session("sess-1")

        assert session.session_id == "sess-1"
        assert session.conversation == []
        assert session.previous_results == []
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_exists(self, manager, mock_redis):
        """Test retrieving an existing session."""
        session_data = SessionState(session_id="sess-1")
        mock_redis.get.return_value = session_data.model_dump_json()

        result = await manager.get_session("sess-1")

        assert result is not None
        assert result.session_id == "sess-1"
        mock_redis.get.assert_called_once_with("session:sess-1")

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, manager, mock_redis):
        """Test retrieving a non-existent session."""
        mock_redis.get.return_value = None

        result = await manager.get_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self, manager, mock_redis):
        """Test get_or_create returns existing session."""
        session_data = SessionState(session_id="sess-1")
        mock_redis.get.return_value = session_data.model_dump_json()

        result = await manager.get_or_create_session("sess-1")

        assert result.session_id == "sess-1"
        # Should NOT call setex (no create)
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_new(self, manager, mock_redis):
        """Test get_or_create creates new when not found."""
        mock_redis.get.return_value = None

        result = await manager.get_or_create_session("new-sess")

        assert result.session_id == "new-sess"
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_session(self, manager, mock_redis):
        """Test deleting a session."""
        mock_redis.delete.return_value = 1

        result = await manager.delete_session("sess-1")

        assert result is True
        mock_redis.delete.assert_called_once_with("session:sess-1")

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, manager, mock_redis):
        """Test deleting a non-existent session."""
        mock_redis.delete.return_value = 0

        result = await manager.delete_session("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_add_user_message(self, manager, mock_redis):
        """Test adding a user message to session."""
        session_data = SessionState(session_id="sess-1")
        mock_redis.get.return_value = session_data.model_dump_json()

        result = await manager.add_user_message("sess-1", "find Italian food")

        assert len(result.conversation) == 1
        assert result.conversation[0].role == "user"
        assert result.conversation[0].content == "find Italian food"

    @pytest.mark.asyncio
    async def test_add_assistant_response(self, manager, mock_redis):
        """Test adding an assistant response with result IDs."""
        session_data = SessionState(session_id="sess-1")
        session_data.add_user_turn("find Italian food")
        mock_redis.get.return_value = session_data.model_dump_json()

        result = await manager.add_assistant_response(
            "sess-1", "Here are Italian options...", ["doc-1", "doc-2"]
        )

        assert len(result.conversation) == 2
        assert result.conversation[1].role == "assistant"
        assert result.previous_results == ["doc-1", "doc-2"]
        assert result.previous_query == "find Italian food"

    @pytest.mark.asyncio
    async def test_update_entities(self, manager, mock_redis):
        """Test updating tracked entities."""
        session_data = SessionState(session_id="sess-1")
        mock_redis.get.return_value = session_data.model_dump_json()

        result = await manager.update_entities(
            "sess-1", {"city": "Boston", "cuisine": ["Italian"]}
        )

        assert result.entities.city == "Boston"
        assert result.entities.cuisine == ["Italian"]

    @pytest.mark.asyncio
    async def test_get_session_context(self, manager, mock_redis):
        """Test getting pipeline context from session."""
        session_data = SessionState(session_id="sess-1")
        session_data.entities.city = "Boston"
        session_data.entities.cuisine = ["Italian"]
        session_data.previous_results = ["doc-1"]
        session_data.previous_query = "Italian in Boston"
        session_data.add_user_turn("Italian in Boston")
        mock_redis.get.return_value = session_data.model_dump_json()

        context = await manager.get_session_context("sess-1")

        assert context["entities"]["city"] == "Boston"
        assert context["previous_results"] == ["doc-1"]
        assert context["previous_query"] == "Italian in Boston"
        assert context["conversation_length"] == 1

    @pytest.mark.asyncio
    async def test_list_sessions(self, manager, mock_redis):
        """Test listing active sessions."""
        mock_redis.scan.return_value = (
            0,
            [b"session:sess-1", b"session:sess-2"],
        )

        result = await manager.list_sessions()

        assert "sess-1" in result
        assert "sess-2" in result

    @pytest.mark.asyncio
    async def test_session_ttl(self, manager, mock_redis):
        """Test that sessions are saved with TTL."""
        await manager.create_session("sess-1")

        call_args = mock_redis.setex.call_args
        # Second arg is TTL
        assert call_args[0][1] == 86400

    @pytest.mark.asyncio
    async def test_redis_error_get_session(self, manager, mock_redis):
        """Test graceful handling of Redis errors on get."""
        mock_redis.get.side_effect = Exception("Redis connection refused")

        result = await manager.get_session("sess-1")
        assert result is None


class TestSessionEntities:
    """Tests for SessionEntities model."""

    def test_to_filters_populated(self):
        """Test converting populated entities to filters."""
        entities = SessionEntities(
            city="Boston",
            cuisine=["Italian"],
            dietary_labels=["vegetarian"],
            price_max=100.0,
            serves_min=20,
        )
        filters = entities.to_filters()

        assert filters["city"] == "Boston"
        assert filters["cuisine"] == ["Italian"]
        assert filters["dietary_labels"] == ["vegetarian"]
        assert filters["price_max"] == 100.0
        assert filters["serves_min"] == 20

    def test_to_filters_empty(self):
        """Test converting empty entities to filters."""
        entities = SessionEntities()
        filters = entities.to_filters()
        assert filters == {}

    def test_update_from_filters(self):
        """Test updating entities from filters dict."""
        entities = SessionEntities(city="Boston")
        entities.update_from_filters({
            "cuisine": ["Italian"],
            "dietary_labels": ["vegetarian"],
        })

        assert entities.city == "Boston"  # Preserved
        assert entities.cuisine == ["Italian"]
        assert entities.dietary_labels == ["vegetarian"]

    def test_update_from_filters_overwrites(self):
        """Test that updates overwrite existing values."""
        entities = SessionEntities(city="Boston", cuisine=["Italian"])
        entities.update_from_filters({"city": "Cambridge"})

        assert entities.city == "Cambridge"
        assert entities.cuisine == ["Italian"]  # Unchanged


class TestSessionState:
    """Tests for SessionState model."""

    def test_add_user_turn(self):
        """Test adding user conversation turn."""
        session = SessionState(session_id="sess-1")
        session.add_user_turn("hello")

        assert len(session.conversation) == 1
        assert session.conversation[0].role == "user"
        assert session.conversation[0].content == "hello"

    def test_add_assistant_turn_with_results(self):
        """Test adding assistant turn updates previous_results."""
        session = SessionState(session_id="sess-1")
        session.add_assistant_turn("Here are results", ["doc-1", "doc-2"])

        assert session.previous_results == ["doc-1", "doc-2"]

    def test_get_recent_conversation(self):
        """Test getting recent conversation turns."""
        session = SessionState(session_id="sess-1")
        for i in range(10):
            session.add_user_turn(f"msg-{i}")
            session.add_assistant_turn(f"reply-{i}")

        recent = session.get_recent_conversation(3)
        # max_turns=3 → last 6 entries (3 user + 3 assistant)
        assert len(recent) == 6

    def test_get_recent_conversation_empty(self):
        """Test recent conversation on empty session."""
        session = SessionState(session_id="sess-1")
        recent = session.get_recent_conversation()
        assert recent == []
