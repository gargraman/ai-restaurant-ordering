"""Integration tests for the API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


@pytest.fixture
def mock_session_manager():
    """Mock session manager."""
    with patch("src.api.main.session_manager") as mock:
        mock.get_or_create_session = AsyncMock()
        mock.save_session = AsyncMock()
        yield mock


@pytest.fixture
def mock_search_pipeline():
    """Mock search pipeline."""
    with patch("src.api.main.get_search_pipeline") as mock:
        pipeline = AsyncMock()
        pipeline.ainvoke = AsyncMock(return_value={
            "session_id": "test-session",
            "resolved_query": "italian catering boston",
            "intent": "search",
            "is_follow_up": False,
            "confidence": 0.9,
            "filters": {"city": "Boston", "cuisine": ["Italian"]},
            "final_context": [
                {
                    "doc_id": "doc-1",
                    "restaurant_name": "Test Restaurant",
                    "city": "Boston",
                    "state": "MA",
                    "item_name": "Pasta Tray",
                    "display_price": 89.99,
                    "rrf_score": 0.05,
                }
            ],
            "answer": "Found 1 Italian catering option in Boston.",
            "sources": ["doc-1"],
        })
        mock.return_value = pipeline
        yield mock


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self):
        """Test health check returns healthy status."""
        # Import here to avoid initialization issues
        from src.api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data


class TestSearchEndpoint:
    """Tests for search endpoint."""

    @pytest.mark.skip(reason="Requires full integration setup")
    def test_search_basic(self, mock_session_manager, mock_search_pipeline):
        """Test basic search request."""
        from src.api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/chat/search",
                json={
                    "session_id": "test-session",
                    "user_input": "Italian catering in Boston",
                    "max_results": 10,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "test-session"
            assert data["intent"] == "search"
            assert len(data["results"]) > 0

    @pytest.mark.skip(reason="Requires full integration setup")
    def test_search_with_follow_up(self, mock_session_manager, mock_search_pipeline):
        """Test search with follow-up query."""
        from src.api.main import app

        with TestClient(app) as client:
            # Initial search
            client.post(
                "/chat/search",
                json={
                    "session_id": "test-session",
                    "user_input": "Italian catering in Boston",
                },
            )

            # Follow-up
            response = client.post(
                "/chat/search",
                json={
                    "session_id": "test-session",
                    "user_input": "vegetarian options",
                },
            )

            assert response.status_code == 200
