"""Tests for Neo4j indexer."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.neo4j_indexer import Neo4jIndexer, NEO4J_SCHEMA


class TestNeo4jIndexerSchema:
    """Tests for Neo4j schema definitions."""

    def test_schema_includes_constraints(self):
        """Test that schema includes uniqueness constraints."""
        assert "CREATE CONSTRAINT restaurant_id" in NEO4J_SCHEMA
        assert "CREATE CONSTRAINT menu_item_id" in NEO4J_SCHEMA
        assert "CREATE CONSTRAINT cuisine_name" in NEO4J_SCHEMA

    def test_schema_includes_indexes(self):
        """Test that schema includes indexes for common queries."""
        assert "CREATE INDEX restaurant_city" in NEO4J_SCHEMA
        assert "CREATE INDEX menu_item_price" in NEO4J_SCHEMA


class TestNeo4jIndexerConnection:
    """Tests for Neo4j connection management."""

    @pytest.mark.asyncio
    async def test_connect_creates_driver(self):
        """Test that connect creates a Neo4j driver."""
        mock_session = AsyncMock()
        mock_session.run = AsyncMock()

        @asynccontextmanager
        async def mock_session_ctx():
            yield mock_session

        with patch("src.ingestion.neo4j_indexer.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_driver.session = mock_session_ctx
            mock_db.driver.return_value = mock_driver

            indexer = Neo4jIndexer()
            await indexer.connect()

            mock_db.driver.assert_called_once()
            assert indexer._connected is True

    @pytest.mark.asyncio
    async def test_close_disconnects(self):
        """Test that close releases the connection."""
        mock_driver = AsyncMock()
        mock_driver.close = AsyncMock()

        indexer = Neo4jIndexer(driver=mock_driver)
        await indexer.close()

        mock_driver.close.assert_called_once()
        assert indexer._connected is False

    def test_driver_property_raises_when_not_connected(self):
        """Test that driver property raises if not connected."""
        indexer = Neo4jIndexer()

        with pytest.raises(RuntimeError, match="not connected"):
            _ = indexer.driver


def create_mock_driver_with_session(session):
    """Helper to create a mock driver with proper async context manager."""
    @asynccontextmanager
    async def mock_session_ctx():
        yield session

    driver = MagicMock()
    driver.session = mock_session_ctx
    return driver


class TestNeo4jIndexerOperations:
    """Tests for Neo4j indexing operations."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver with session support."""
        session = AsyncMock()
        session.run = AsyncMock(return_value=AsyncMock(single=AsyncMock(return_value=None)))
        driver = create_mock_driver_with_session(session)
        return driver, session

    @pytest.mark.asyncio
    async def test_create_schema(self, mock_driver):
        """Test schema creation executes statements."""
        driver, session = mock_driver

        indexer = Neo4jIndexer(driver=driver)
        await indexer.create_schema()

        # Should execute multiple statements
        assert session.run.call_count > 0

    @pytest.mark.asyncio
    async def test_clear_all(self, mock_driver):
        """Test clearing all data."""
        driver, session = mock_driver

        indexer = Neo4jIndexer(driver=driver)
        await indexer.clear_all()

        # Check DETACH DELETE was called
        session.run.assert_called_with("MATCH (n) DETACH DELETE n")

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_driver):
        """Test getting database statistics."""
        driver, session = mock_driver

        # Mock the stats query result
        mock_record = {
            "restaurants": 10,
            "menus": 20,
            "groups": 50,
            "items": 200,
            "cuisines": 5,
        }
        session.run.return_value.single = AsyncMock(return_value=mock_record)

        indexer = Neo4jIndexer(driver=driver)
        stats = await indexer.get_stats()

        assert stats == mock_record

    @pytest.mark.asyncio
    async def test_index_restaurants_empty_input(self, mock_driver):
        """Test that empty restaurant list returns early."""
        driver, _ = mock_driver

        indexer = Neo4jIndexer(driver=driver)
        result = await indexer.index_restaurants([])

        assert result["restaurants"] == 0
        assert result["items"] == 0


class TestNeo4jIndexerRelationships:
    """Tests for relationship creation."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        session = AsyncMock()
        driver = create_mock_driver_with_session(session)
        return driver, session

    @pytest.mark.asyncio
    async def test_create_pairing_relationships(self, mock_driver):
        """Test creating PAIRS_WITH relationships."""
        driver, session = mock_driver

        # Mock the result
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"count": 42})
        session.run.return_value = mock_result

        indexer = Neo4jIndexer(driver=driver)
        count = await indexer.create_pairing_relationships()

        assert count == 42
        # Verify PAIRS_WITH is in the query
        call_args = session.run.call_args
        assert "PAIRS_WITH" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_similar_restaurant_relationships(self, mock_driver):
        """Test creating SIMILAR_TO relationships."""
        driver, session = mock_driver

        # Mock the result
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"count": 15})
        session.run.return_value = mock_result

        indexer = Neo4jIndexer(driver=driver)
        count = await indexer.create_similar_restaurant_relationships(max_distance_km=5.0)

        assert count == 15
        # Verify SIMILAR_TO is in the query
        call_args = session.run.call_args
        assert "SIMILAR_TO" in call_args[0][0]
        # Check the params dict contains max_distance_km
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params.get("max_distance_km") == 5.0
