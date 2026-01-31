"""FastAPI application entry point."""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.config import get_settings
from src.models.api import (
    SearchRequest,
    SearchResponse,
    MenuItemResult,
    SessionResponse,
    FeedbackRequest,
)
from src.models.state import GraphState
from src.session.manager import SessionManager
from src.search.hybrid import HybridSearcher
from src.langgraph.graph import get_search_pipeline
from src.langgraph.nodes import set_session_manager
from src.monitoring.middleware import MetricsMiddleware, ErrorTrackingMiddleware, metrics_endpoint
from src.monitoring.system_metrics import system_metrics_collector
from src.monitoring.database_monitor import stop_db_metrics_collector
from src.monitoring.tracing import setup_tracing
from src.metrics import increment_active_sessions, decrement_active_sessions, collect_system_metrics

logger = structlog.get_logger()
settings = get_settings()

# Global instances
session_manager: SessionManager | None = None
hybrid_searcher: HybridSearcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global session_manager, hybrid_searcher

    logger.info("application_starting", app_name=settings.app_name)

    # Initialize connections
    session_manager = SessionManager()
    await session_manager.connect()
    set_session_manager(session_manager)

    hybrid_searcher = HybridSearcher()
    await hybrid_searcher.vector_searcher.connect()

    # Start system metrics collection
    system_metrics_collector.start()

    # Start system metrics collection task
    system_metrics_task = asyncio.create_task(collect_system_metrics())

    # Database metrics collector is started when vector searcher connects
    # No need to start it separately here

    yield

    # Cleanup
    logger.info("application_shutting_down")
    if session_manager:
        await session_manager.close()
    if hybrid_searcher:
        await hybrid_searcher.close()

    # Stop system metrics collection
    system_metrics_collector.stop()

    # Cancel system metrics collection task
    system_metrics_task.cancel()

    # Stop database metrics collection if initialized
    await stop_db_metrics_collector()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Hybrid Search API",
        description="Conversational Hybrid Search & RAG System for Catering Menus",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Tracing (Phase 2)
    setup_tracing(app)

    # Add monitoring middleware
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(ErrorTrackingMiddleware)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        logger.info(
            "request_completed",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            process_time_ms=round(process_time, 2),
        )

        return response

    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "0.1.0",
        }

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return metrics_endpoint()

    # Chat search endpoint
    @app.post("/chat/search", response_model=SearchResponse)
    async def chat_search(request: SearchRequest):
        """Execute conversational search.

        This endpoint:
        1. Loads/creates session
        2. Runs the LangGraph pipeline
        3. Updates session with results
        4. Returns response

        Examples:
        - "Find Italian catering in Boston for 20 people"
        - "Show me vegetarian options"
        - "Cheaper ones under $15 per person"
        """
        start_time = time.time()

        try:
            # Get or create session
            session = await session_manager.get_or_create_session(request.session_id)

            # Add user message to session
            session.add_user_turn(request.user_input)

            # Build initial state
            state: GraphState = {
                "session_id": request.session_id,
                "user_input": request.user_input,
                "timestamp": datetime.utcnow().isoformat(),
                "intent": "search",
                "is_follow_up": False,
                "follow_up_type": None,
                "confidence": 0.0,
                "resolved_query": "",
                "filters": {},
                "expanded_query": "",
                "candidate_doc_ids": [],
                "bm25_results": [],
                "vector_results": [],
                "merged_results": [],
                "final_context": [],
                "answer": "",
                "sources": [],
                "error": None,
            }

            # Run the search pipeline
            pipeline = get_search_pipeline()
            result = await pipeline.ainvoke(state)

            # Update session with results
            result_ids = result.get("sources", [])
            session.add_assistant_turn(result.get("answer", ""), result_ids)
            session.entities.update_from_filters(result.get("filters", {}))
            session.previous_query = result.get("resolved_query") or request.user_input
            session.previous_results = result_ids
            await session_manager.save_session(session)

            # Build response
            processing_time = (time.time() - start_time) * 1000

            results = [
                MenuItemResult(
                    doc_id=doc.get("doc_id", ""),
                    restaurant_name=doc.get("restaurant_name", "Unknown"),
                    city=doc.get("city", ""),
                    state=doc.get("state", ""),
                    item_name=doc.get("item_name", "Unknown"),
                    item_description=doc.get("item_description"),
                    display_price=doc.get("display_price"),
                    price_per_person=doc.get("price_per_person"),
                    serves_min=doc.get("serves_min"),
                    serves_max=doc.get("serves_max"),
                    dietary_labels=doc.get("dietary_labels", []),
                    tags=doc.get("tags", []),
                    rrf_score=doc.get("rrf_score", 0.0),
                )
                for doc in result.get("final_context", [])[:request.max_results]
            ]

            return SearchResponse(
                session_id=request.session_id,
                resolved_query=result.get("resolved_query", request.user_input),
                intent=result.get("intent", "search"),
                is_follow_up=result.get("is_follow_up", False),
                filters=result.get("filters", {}),
                results=results,
                answer=result.get("answer", ""),
                confidence=result.get("confidence", 0.0),
                processing_time_ms=round(processing_time, 2),
            )

        except Exception as e:
            logger.error("search_error", error=str(e), session_id=request.session_id)
            raise HTTPException(status_code=500, detail=str(e))

    # Session endpoints
    @app.get("/session/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str):
        """Get session state."""
        session = await session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            last_activity=session.last_activity.isoformat(),
            entities=session.entities.to_filters(),
            conversation_length=len(session.conversation),
            previous_results_count=len(session.previous_results),
        )

    @app.delete("/session/{session_id}")
    async def delete_session(session_id: str):
        """Clear session and start fresh."""
        deleted = await session_manager.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted", "session_id": session_id}

    @app.post("/session/{session_id}/feedback")
    async def submit_feedback(session_id: str, feedback: FeedbackRequest):
        """Submit relevance feedback for a result."""
        logger.info(
            "feedback_received",
            session_id=session_id,
            doc_id=feedback.doc_id,
            rating=feedback.rating,
        )

        # Record user feedback in metrics
        from src.metrics import record_user_feedback
        record_user_feedback(result_type="search_result", rating=feedback.rating)

        # In production, this would be stored for model improvement
        return {"status": "received", "doc_id": feedback.doc_id, "rating": feedback.rating}

    return app


# Create app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
