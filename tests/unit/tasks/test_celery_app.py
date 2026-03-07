"""Unit tests for Celery application configuration."""

import pytest
from unittest.mock import patch, MagicMock

from src.tasks.celery_app import make_celery_app, celery_app


class TestCeleryApp:
    """Test cases for Celery application configuration."""

    def test_make_celery_app_creation(self):
        """Test that Celery app is created with correct configuration."""
        app = make_celery_app()
        
        assert app is not None
        assert app.main == "hybrid_search_tasks"
        assert app.conf.task_serializer == "json"
        assert app.conf.accept_content == ["json"]
        assert app.conf.result_serializer == "json"
        assert app.conf.timezone == "UTC"
        assert app.conf.enable_utc is True
        
    def test_task_routes_configuration(self):
        """Test that task routes are properly configured."""
        app = make_celery_app()
        
        expected_routes = {
            "src.tasks.order_tasks.send_order_to_pos": {"queue": "orders.high"},
            "src.tasks.order_tasks.sync_menu_catalog": {"queue": "catalog.low"},
            "src.tasks.dlq.process_dlq_order": {"queue": "dlq.medium"},
            "src.tasks.dlq.notify_order_failure": {"queue": "notifications.medium"},
        }
        
        for task_name, expected_route in expected_routes.items():
            assert task_name in app.conf.task_routes
            assert app.conf.task_routes[task_name] == expected_route
            
    def test_default_queue_configuration(self):
        """Test that default queue is configured."""
        app = make_celery_app()
        
        assert app.conf.task_default_queue == "default"
        
    def test_worker_settings(self):
        """Test worker configuration settings."""
        app = make_celery_app()
        
        assert app.conf.worker_prefetch_multiplier == 4
        assert app.conf.worker_concurrency == 4
        
    def test_broker_transport_options(self):
        """Test broker transport options."""
        app = make_celery_app()
        
        assert app.conf.broker_transport_options["visibility_timeout"] == 600
        assert app.conf.broker_transport_options["fanout_prefix"] is True
        assert app.conf.broker_transport_options["fanout_patterns"] is True
        
    def test_global_celery_app_instance(self):
        """Test that global celery_app instance is created."""
        assert celery_app is not None
        assert hasattr(celery_app, 'task')