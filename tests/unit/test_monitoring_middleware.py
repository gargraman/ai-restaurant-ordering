"""Unit tests for monitoring middleware."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.testclient import TestClient
from datetime import datetime
from io import StringIO
import sys
from unittest.mock import patch

from src.monitoring.middleware import MetricsMiddleware
from src.monitoring.system_metrics import collect_system_metrics


@pytest.fixture
def mock_app():
    """Mock FastAPI app for testing middleware."""
    app = FastAPI()
    
    @app.get("/")
    def root():
        return {"message": "Hello World"}
    
    @app.get("/error")
    def error_route():
        raise Exception("Test error")
    
    @app.get("/slow")
    async def slow_route():
        import asyncio
        await asyncio.sleep(0.1)  # Small delay for timing test
        return {"message": "Slow response"}
    
    return app


@pytest.fixture
def mock_request():
    """Mock request object."""
    request = AsyncMock()
    request.method = "GET"
    request.url = "http://testserver/"
    request.headers = {}
    return request


@pytest.mark.asyncio
async def test_metrics_middleware_initialization():
    """Test MetricsMiddleware initialization."""
    middleware = MetricsMiddleware(app=AsyncMock())
    
    assert hasattr(middleware, 'dispatch')
    assert callable(middleware.dispatch)


@pytest.mark.asyncio
async def test_metrics_middleware_request_response_cycle(mock_app):
    """Test that middleware processes request and response correctly."""
    mock_app.add_middleware(MetricsMiddleware)
    client = TestClient(mock_app)
    
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


@pytest.mark.asyncio
async def test_metrics_middleware_error_handling(mock_app):
    """Test that middleware handles errors correctly."""
    mock_app.add_middleware(MetricsMiddleware)
    client = TestClient(mock_app)
    
    with pytest.raises(Exception):
        client.get("/error")


@pytest.mark.asyncio
async def test_metrics_middleware_response_timing(mock_app):
    """Test that middleware captures response timing."""
    mock_app.add_middleware(MetricsMiddleware)
    client = TestClient(mock_app)
    
    # Capture the response time
    import time
    start_time = time.time()
    response = client.get("/slow")
    end_time = time.time()
    
    assert response.status_code == 200
    # Verify the response took at least the minimum time
    assert (end_time - start_time) >= 0.1


@pytest.mark.asyncio
async def test_metrics_middleware_request_size_tracking():
    """Test that middleware tracks request sizes."""
    app = FastAPI()
    
    @app.post("/data")
    async def data_endpoint(data: dict):
        return {"received": len(str(data))}
    
    app.add_middleware(MetricsMiddleware)
    client = TestClient(app)
    
    # Send a request with some data
    test_data = {"key": "value" * 100}  # Large payload
    response = client.post("/data", json=test_data)
    
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_metrics_middleware_status_codes(mock_app):
    """Test that middleware tracks different status codes."""
    mock_app.add_middleware(MetricsMiddleware)
    client = TestClient(mock_app)
    
    # Test 200 status
    response = client.get("/")
    assert response.status_code == 200
    
    # Test 404 status
    response = client.get("/nonexistent")
    assert response.status_code == 404


def test_collect_system_metrics():
    """Test system metrics collection."""
    # This function should return a dictionary with system metrics
    metrics = collect_system_metrics()
    
    # Verify structure of returned metrics
    assert isinstance(metrics, dict)
    assert "cpu_percent" in metrics
    assert "memory_percent" in metrics
    assert "disk_usage_percent" in metrics
    assert "timestamp" in metrics
    
    # Verify values are reasonable
    assert 0 <= metrics["cpu_percent"] <= 100
    assert 0 <= metrics["memory_percent"] <= 100
    assert 0 <= metrics["disk_usage_percent"] <= 100
    assert isinstance(metrics["timestamp"], str)


@pytest.mark.asyncio
async def test_metrics_middleware_calls_dispatch_method():
    """Test that middleware dispatch method is called correctly."""
    app_mock = AsyncMock()
    middleware = MetricsMiddleware(app_mock)
    
    request = AsyncMock()
    request.method = "GET"
    request.url = "http://testserver/"
    request.__getitem__ = lambda _, key: {"content-type": "application/json"}.get(key, "")
    request.headers = {"content-type": "application/json"}
    
    call_next = AsyncMock()
    response_mock = AsyncMock()
    response_mock.status_code = 200
    call_next.return_value = response_mock
    
    # Mock the metrics collection to avoid actual system calls
    with patch('src.monitoring.middleware.collect_system_metrics') as mock_collect:
        mock_collect.return_value = {
            "cpu_percent": 10.0,
            "memory_percent": 20.0,
            "disk_usage_percent": 30.0,
            "timestamp": datetime.now().isoformat()
        }
        
        response = await middleware.dispatch(request, call_next)
        
        # Verify call_next was called
        call_next.assert_called_once()
        # Verify response was returned
        assert response == response_mock


@pytest.mark.asyncio
async def test_metrics_middleware_exception_during_request():
    """Test middleware behavior when exception occurs during request processing."""
    app_mock = AsyncMock()
    middleware = MetricsMiddleware(app_mock)
    
    request = AsyncMock()
    request.method = "GET"
    request.url = "http://testserver/"
    request.__getitem__ = lambda _, key: {"content-type": "application/json"}.get(key, "")
    request.headers = {"content-type": "application/json"}
    
    # Make call_next raise an exception
    call_next = AsyncMock()
    call_next.side_effect = Exception("Request processing failed")
    
    # Mock the metrics collection to avoid actual system calls
    with patch('src.monitoring.middleware.collect_system_metrics') as mock_collect:
        mock_collect.return_value = {
            "cpu_percent": 10.0,
            "memory_percent": 20.0,
            "disk_usage_percent": 30.0,
            "timestamp": datetime.now().isoformat()
        }
        
        with pytest.raises(Exception, match="Request processing failed"):
            await middleware.dispatch(request, call_next)
            
        # Verify call_next was called
        call_next.assert_called_once()


@pytest.mark.asyncio
async def test_metrics_middleware_with_different_http_methods():
    """Test middleware with different HTTP methods."""
    app = FastAPI()
    
    @app.post("/post")
    async def post_endpoint(data: dict = None):
        return {"method": "POST"}
    
    @app.put("/put")
    async def put_endpoint(data: dict = None):
        return {"method": "PUT"}
    
    @app.delete("/delete")
    async def delete_endpoint():
        return {"method": "DELETE"}
    
    app.add_middleware(MetricsMiddleware)
    client = TestClient(app)
    
    # Test POST
    response = client.post("/post", json={})
    assert response.status_code == 200
    
    # Test PUT
    response = client.put("/put", json={})
    assert response.status_code == 200
    
    # Test DELETE
    response = client.delete("/delete")
    assert response.status_code == 200