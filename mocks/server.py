"""Mock API server for development and testing."""

import asyncio
import json
import time
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mocks.data_generator import MockDataGenerator
from mocks.scenarios import ScenarioManager, ScenarioType


class MockServer:
    """Mock server that simulates OpenSearch, pgvector, and Redis responses."""

    def __init__(self):
        self.data_generator = MockDataGenerator()
        self.scenario_manager = ScenarioManager()
        self.request_log: list[dict] = []
        self.sessions: dict[str, dict] = {}
        self.indexed_documents: dict[str, dict] = {}

    def track_request(self, request_data: dict) -> None:
        """Track a request for verification."""
        self.request_log.append({
            **request_data,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_requests(self, path_filter: str | None = None) -> list[dict]:
        """Get tracked requests, optionally filtered by path."""
        if path_filter:
            return [r for r in self.request_log if path_filter in r.get("path", "")]
        return self.request_log

    def reset(self) -> None:
        """Reset mock server state."""
        self.request_log = []
        self.sessions = {}
        self.scenario_manager.reset()


def create_mock_app() -> FastAPI:
    """Create the mock FastAPI application."""
    app = FastAPI(
        title="Mock API Server",
        description="Mock server for hybrid search development",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    mock_server = MockServer()

    # Middleware for request tracking and latency simulation
    @app.middleware("http")
    async def mock_middleware(request: Request, call_next):
        # Track request
        mock_server.track_request({
            "method": request.method,
            "path": str(request.url.path),
            "query": str(request.url.query),
        })

        # Add response headers
        response = await call_next(request)
        response.headers["X-Mock-Server"] = "true"
        response.headers["X-Mock-Scenario"] = mock_server.scenario_manager.current_scenario.value

        return response

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "healthy", "mock": True}

    # Scenario management
    @app.get("/mock/scenarios")
    async def list_scenarios():
        return mock_server.scenario_manager.list_scenarios()

    @app.post("/mock/scenarios/{scenario}")
    async def set_scenario(scenario: str):
        try:
            mock_server.scenario_manager.set_scenario(scenario)
            return {"status": "ok", "scenario": scenario}
        except ValueError:
            raise HTTPException(400, f"Unknown scenario: {scenario}")

    @app.post("/mock/reset")
    async def reset_mock():
        mock_server.reset()
        return {"status": "ok"}

    @app.get("/mock/requests")
    async def get_requests(path: str | None = None):
        return mock_server.get_requests(path)

    # Mock OpenSearch endpoints
    @app.get("/{index}/_search")
    @app.post("/{index}/_search")
    async def opensearch_search(index: str, request: Request):
        """Mock OpenSearch search endpoint."""
        # Simulate latency
        delay = mock_server.scenario_manager.get_response_delay()
        if delay > 0:
            await asyncio.sleep(delay / 1000)

        # Check for errors
        if mock_server.scenario_manager.should_return_error("opensearch_search"):
            scenario = mock_server.scenario_manager.get_scenario()
            raise HTTPException(scenario.error_code, scenario.error_message)

        # Check for empty results
        if mock_server.scenario_manager.should_return_empty():
            return {
                "took": 5,
                "hits": {"total": {"value": 0}, "hits": []},
            }

        # Generate mock results
        count = mock_server.scenario_manager.get_result_count()
        results = mock_server.data_generator.generate_search_results(count)

        return {
            "took": 15,
            "timed_out": False,
            "hits": {
                "total": {"value": len(results), "relation": "eq"},
                "max_score": 1.0,
                "hits": [
                    {
                        "_index": index,
                        "_id": r["doc_id"],
                        "_score": 1.0 - i * 0.05,
                        "_source": r,
                    }
                    for i, r in enumerate(results)
                ],
            },
        }

    @app.post("/{index}/_bulk")
    async def opensearch_bulk(index: str, request: Request):
        """Mock OpenSearch bulk indexing."""
        body = await request.body()
        lines = body.decode().strip().split("\n")
        doc_count = len(lines) // 2

        return {
            "took": 50,
            "errors": False,
            "items": [
                {"index": {"_index": index, "_id": str(uuid4()), "status": 201}}
                for _ in range(doc_count)
            ],
        }

    @app.put("/{index}")
    async def opensearch_create_index(index: str):
        """Mock OpenSearch index creation."""
        return {"acknowledged": True, "index": index}

    @app.head("/{index}")
    async def opensearch_index_exists(index: str):
        """Mock OpenSearch index exists check."""
        return Response(status_code=200)

    @app.get("/{index}/_count")
    async def opensearch_count(index: str):
        """Mock OpenSearch document count."""
        return {"count": len(mock_server.indexed_documents)}

    # Mock pgvector/PostgreSQL endpoints (simulated via HTTP for testing)
    @app.post("/pgvector/search")
    async def pgvector_search(request: Request):
        """Mock vector similarity search."""
        delay = mock_server.scenario_manager.get_response_delay()
        if delay > 0:
            await asyncio.sleep(delay / 1000)

        if mock_server.scenario_manager.should_return_error("pgvector_search"):
            scenario = mock_server.scenario_manager.get_scenario()
            raise HTTPException(scenario.error_code, scenario.error_message)

        count = mock_server.scenario_manager.get_result_count()
        results = mock_server.data_generator.generate_search_results(count)

        return {
            "results": [
                {"doc_id": r["doc_id"], "score": 0.9 - i * 0.05}
                for i, r in enumerate(results)
            ]
        }

    # Mock Redis session endpoints
    @app.get("/session/{session_id}")
    async def get_session(session_id: str):
        """Get mock session."""
        if session_id not in mock_server.sessions:
            return {"session_id": session_id, "exists": False}

        return {
            "session_id": session_id,
            "exists": True,
            "data": mock_server.sessions[session_id],
        }

    @app.post("/session/{session_id}")
    async def create_session(session_id: str, request: Request):
        """Create/update mock session."""
        data = await request.json()
        mock_server.sessions[session_id] = {
            **data,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
        }
        return {"status": "ok", "session_id": session_id}

    @app.delete("/session/{session_id}")
    async def delete_session(session_id: str):
        """Delete mock session."""
        if session_id in mock_server.sessions:
            del mock_server.sessions[session_id]
        return {"status": "ok"}

    # Mock chat/search endpoint (main API)
    class ChatRequest(BaseModel):
        session_id: str
        user_input: str
        max_results: int = 10

    @app.post("/chat/search")
    async def chat_search(req: ChatRequest):
        """Mock chat search endpoint."""
        delay = mock_server.scenario_manager.get_response_delay()
        if delay > 0:
            await asyncio.sleep(delay / 1000)

        if mock_server.scenario_manager.should_return_error("chat_search"):
            scenario = mock_server.scenario_manager.get_scenario()
            raise HTTPException(scenario.error_code, scenario.error_message)

        count = min(req.max_results, mock_server.scenario_manager.get_result_count())
        results = mock_server.data_generator.generate_search_results(count)

        # Determine intent based on input
        input_lower = req.user_input.lower()
        is_follow_up = any(kw in input_lower for kw in ["cheaper", "more", "vegetarian", "same"])
        intent = "filter" if is_follow_up else "search"

        return {
            "session_id": req.session_id,
            "resolved_query": req.user_input,
            "intent": intent,
            "is_follow_up": is_follow_up,
            "filters": {},
            "results": results,
            "answer": f"Found {len(results)} catering options matching your request.",
            "confidence": 0.85,
            "processing_time_ms": delay + 50,
        }

    return app


# Run with: uvicorn mocks.server:app --reload --port 3001
app = create_mock_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
