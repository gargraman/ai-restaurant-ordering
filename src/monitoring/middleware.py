"""Middleware for monitoring HTTP requests."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
import time
from typing import Callable, Awaitable
from src.metrics import REQUEST_COUNT, REQUEST_DURATION, APPLICATION_ERRORS
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect metrics for HTTP requests."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> StarletteResponse:
        start_time = time.time()

        response = await call_next(request)

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(time.time() - start_time)

        return response


class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to capture unhandled errors and track error metrics."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> StarletteResponse:
        try:
            response = await call_next(request)

            if response.status_code >= 500:
                APPLICATION_ERRORS.labels(
                    type="http_5xx",
                    endpoint=request.url.path,
                ).inc()

            return response
        except Exception:
            APPLICATION_ERRORS.labels(
                type="unhandled_exception",
                endpoint=request.url.path,
            ).inc()
            raise


def metrics_endpoint():
    """Endpoint to expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)