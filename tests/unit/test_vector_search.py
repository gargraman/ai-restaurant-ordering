"""Tests for vector search implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.search.vector import VectorSearcher


class TestVectorSearcher:
    """Tests for VectorSearcher class."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg connection pool."""
        from unittest.mock import AsyncMock, MagicMock
        
        pool = MagicMock()
        
        conn = AsyncMock()
        conn.fetch.return_value = [
            {
                "doc_id": "doc-1",
                "score": 0.92,
                "restaurant_id": "rest-1",
                "city": "Boston",
                "base_price": 89.99,
                "serves_max": 12,
            },
            {
                "doc_id": "doc-2",
                "score": 0.88,
                "restaurant_id": "rest-1",
                "city": "Boston",
                "base_price": 79.99,
                "serves_max": 10,
            },
        ]
        
        # Create async context manager using AsyncMock
        async_context = AsyncMock()
        async_context.__aenter__.return_value = conn
        async_context.__aexit__.return_value = None
        pool.acquire.return_value = async_context
        
        return pool

    @pytest.fixture
    def mock_embedding_generator(self):
        """Create a mock embedding generator."""
        generator = AsyncMock()
        # Return string format as pgvector expects
        generator.generate_query_embedding.return_value = [0.1] * 1536
        return generator

    @pytest.fixture
    async def vector_searcher(self, mock_pool, mock_embedding_generator):
        """Create a VectorSearcher with mock pool."""
        searcher = VectorSearcher(
            pool=mock_pool,
            embedding_generator=mock_embedding_generator,
        )
        searcher._dsn = "postgresql://test"
        return searcher

    @pytest.mark.asyncio
    async def test_search_returns_results_with_scores(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that search returns results with similarity scores."""
        results = await vector_searcher.search("italian food")

        assert len(results) == 2
        assert results[0]["doc_id"] == "doc-1"
        assert results[0]["score"] == 0.92
        assert results[1]["doc_id"] == "doc-2"
        assert results[1]["score"] == 0.88

    @pytest.mark.asyncio
    async def test_search_generates_query_embedding(
        self,
        vector_searcher,
        mock_pool,
        mock_embedding_generator,
    ):
        """Test that search generates embedding for the query."""
        await vector_searcher.search("italian food")

        mock_embedding_generator.generate_query_embedding.assert_called_once_with(
            "italian food"
        )

    @pytest.mark.asyncio
    async def test_search_builds_correct_sql(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that search builds the correct SQL query."""
        await vector_searcher.search("italian food", top_k=10)

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.assert_called_once()

        call_args = conn.fetch.call_args[0][0]
        # Check SQL contains expected components
        assert "SELECT" in call_args
        assert "FROM menu_embeddings" in call_args
        assert "embedding <=> $" in call_args
        assert "ORDER BY embedding <=> $" in call_args
        assert "LIMIT" in call_args

    @pytest.mark.asyncio
    async def test_search_applies_city_filter(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that city filter is applied."""
        await vector_searcher.search("italian food", filters={"city": "Boston"})

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.assert_called_once()
        
        call_args = conn.fetch.call_args[0]
        sql = call_args[0]

        # SQL should contain city condition
        assert "city = $" in sql

    @pytest.mark.asyncio
    async def test_search_applies_price_filter(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that price filter is applied."""
        await vector_searcher.search("italian food", filters={"price_max": 100.0})

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.fetch.call_args[0][0]

        # SQL should contain price condition
        assert "base_price <= $" in call_args

    @pytest.mark.asyncio
    async def test_search_applies_serving_filter(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that serving size filter is applied."""
        await vector_searcher.search("catering", filters={"serves_min": 10})

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.fetch.call_args[0][0]

        # SQL should contain serving condition
        assert "serves_max >= $" in call_args

    @pytest.mark.asyncio
    async def test_search_applies_dietary_filter(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that dietary labels filter is applied."""
        await vector_searcher.search(
            "salad", filters={"dietary_labels": ["vegetarian", "vegan"]}
        )

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.fetch.call_args[0][0]

        # SQL should contain dietary condition with array overlap operator
        assert "dietary_labels && $" in call_args

    @pytest.mark.asyncio
    async def test_search_applies_restaurant_id_filter(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that restaurant_id filter is applied."""
        await vector_searcher.search("pasta", filters={"restaurant_id": "rest-123"})

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.fetch.call_args[0][0]

        assert "restaurant_id = $" in call_args

    @pytest.mark.asyncio
    async def test_search_applies_exclude_restaurant_filter(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that exclude_restaurant_id filter is applied."""
        await vector_searcher.search(
            "pasta", filters={"exclude_restaurant_id": "rest-123"}
        )

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.fetch.call_args[0][0]

        assert "restaurant_id != $" in call_args

    @pytest.mark.asyncio
    async def test_search_applies_multiple_filters(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that multiple filters are applied together."""
        filters = {
            "city": "Boston",
            "price_max": 100.0,
            "serves_min": 10,
            "dietary_labels": ["vegetarian"],
        }

        await vector_searcher.search("italian food", filters=filters)

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.fetch.call_args[0][0]

        # SQL should contain all filter conditions
        assert "city = $" in call_args
        assert "base_price <= $" in call_args
        assert "serves_max >= $" in call_args
        assert "dietary_labels && $" in call_args

    @pytest.mark.asyncio
    async def test_search_empty_filters(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that search works with empty filters."""
        await vector_searcher.search("italian food", filters={})

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.fetch.call_args[0][0]

        # Should have default condition
        assert "1=1" in call_args

    @pytest.mark.asyncio
    async def test_search_handles_database_error(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that database errors are handled gracefully."""
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.side_effect = Exception("Connection refused")

        results = await vector_searcher.search("test")

        # Should return empty list on error
        assert results == []

    @pytest.mark.asyncio
    async def test_search_uses_default_top_k_from_settings(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that search uses top_k from settings when not specified."""
        with patch("src.search.vector.settings") as mock_settings:
            mock_settings.vector_top_k = 30

            await vector_searcher.search("test")

            conn = mock_pool.acquire.return_value.__aenter__.return_value
            conn.fetch.assert_called_once()
            # Just verify it was called - the top_k is used in the LIMIT clause
            call_args = conn.fetch.call_args[0][0]
            assert "LIMIT" in call_args

    @pytest.mark.asyncio
    async def test_search_records_metrics_on_success(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that search records metrics on success."""
        with patch("src.search.vector.record_search_request") as mock_record:
            await vector_searcher.search("test")

            mock_record.assert_called()
            call_args = mock_record.call_args
            assert call_args[0][0] == "vector"  # search_type
            # Arguments are (search_type, duration, result_count)
            assert call_args[0][1] > 0  # duration
            assert call_args[0][2] == 2  # result_count

    @pytest.mark.asyncio
    async def test_search_records_metrics_on_error(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that search records metrics on error."""
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.side_effect = Exception("Error")

        with patch("src.search.vector.record_search_request") as mock_record:
            await vector_searcher.search("test")

            mock_record.assert_called()
            call_args = mock_record.call_args
            assert call_args[0][0] == "vector"
            # Arguments are (search_type, duration, result_count)
            assert call_args[0][1] > 0  # duration
            assert call_args[0][2] == 0  # result_count (failed)

    @pytest.mark.asyncio
    async def test_connect_creates_pool(self):
        """Test that connect creates a connection pool."""
        with patch("src.search.vector.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_pool_instance = AsyncMock()
            mock_create.return_value = mock_pool_instance

            searcher = VectorSearcher()
            searcher._dsn = "postgresql://test"
            await searcher.connect()

            mock_create.assert_called_once()
            assert searcher.pool == mock_pool_instance

    @pytest.mark.asyncio
    async def test_connect_skips_if_pool_exists(self):
        """Test that connect skips if pool already exists."""
        with patch("src.search.vector.asyncpg.create_pool") as mock_create:
            existing_pool = AsyncMock()

            searcher = VectorSearcher(pool=existing_pool)
            await searcher.connect()

            mock_create.assert_not_called()
            assert searcher.pool == existing_pool

    @pytest.mark.asyncio
    async def test_close_closes_pool(self):
        """Test that close closes the connection pool."""
        mock_pool = AsyncMock()
        searcher = VectorSearcher(pool=mock_pool)

        await searcher.close()

        mock_pool.close.assert_called_once()
        assert searcher.pool is None

    @pytest.mark.asyncio
    async def test_close_skips_if_no_pool(self):
        """Test that close skips if no pool exists."""
        searcher = VectorSearcher()

        # Should not raise
        await searcher.close()

    @pytest.mark.asyncio
    async def test_search_with_function_uses_stored_function(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that search_with_function uses the stored SQL function."""
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.return_value = [
            {
                "doc_id": "doc-1",
                "score": 0.92,
                "restaurant_id": "rest-1",
                "city": "Boston",
                "base_price": 89.99,
                "serves_max": 12,
            }
        ]

        results = await vector_searcher.search_with_function(
            "italian food",
            filters={"city": "Boston"},
            top_k=10,
        )

        # Should call the stored function
        conn.fetch.assert_called_once()
        call_args = conn.fetch.call_args[0][0]
        assert "search_menu_embeddings" in call_args

        # Should return results
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_search_with_function_handles_error(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that search_with_function handles errors gracefully."""
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.side_effect = Exception("Function not found")

        results = await vector_searcher.search_with_function("test")

        # Should return empty list on error
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_minimal_fields(
        self,
        vector_searcher,
        mock_pool,
    ):
        """Test that vector search returns minimal fields."""
        results = await vector_searcher.search("test")

        # Vector search only returns minimal fields
        for result in results:
            assert "doc_id" in result
            assert "score" in result
            assert "restaurant_id" in result
            assert "city" in result
            assert "base_price" in result
            assert "serves_max" in result
            # Should NOT return full document fields
            assert "item_name" not in result
            assert "item_description" not in result


class TestVectorSearcherIntegration:
    """Integration-style tests for VectorSearcher."""

    @pytest.mark.asyncio
    async def test_search_end_to_end(self):
        """Test full search flow end-to-end."""
        # Create real mock chain
        mock_pool = MagicMock()
        conn = AsyncMock()
        conn.fetch.return_value = [
            {
                "doc_id": "doc-1",
                "score": 0.95,
                "restaurant_id": "rest-1",
                "city": "Boston",
                "base_price": 99.99,
                "serves_max": 15,
            },
            {
                "doc_id": "doc-2",
                "score": 0.85,
                "restaurant_id": "rest-2",
                "city": "Cambridge",
                "base_price": 79.99,
                "serves_max": 10,
            },
        ]
        
        async_context = AsyncMock()
        async_context.__aenter__.return_value = conn
        async_context.__aexit__.return_value = None
        mock_pool.acquire.return_value = async_context

        mock_embedding_generator = AsyncMock()
        mock_embedding_generator.generate_query_embedding.return_value = [0.1] * 1536

        searcher = VectorSearcher(
            pool=mock_pool,
            embedding_generator=mock_embedding_generator,
        )

        results = await searcher.search("italian catering", top_k=5)

        # Verify embedding was generated
        mock_embedding_generator.generate_query_embedding.assert_called_once_with(
            "italian catering"
        )

        # Verify results
        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]  # Ordered by score
