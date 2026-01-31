"""OpenTelemetry tracing setup for the hybrid search application."""

import importlib
import structlog
from fastapi import FastAPI

from src.config import get_settings

logger = structlog.get_logger()


def setup_tracing(app: FastAPI) -> None:
    """Configure OpenTelemetry tracing for the FastAPI app."""
    settings = get_settings()

    if not settings.otel_tracing_enabled:
        logger.info("otel_tracing_disabled")
        return

    try:
        trace = importlib.import_module("opentelemetry.trace")
        exporter_module = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        )
        instrumentation_module = importlib.import_module(
            "opentelemetry.instrumentation.fastapi"
        )
        resources_module = importlib.import_module("opentelemetry.sdk.resources")
        trace_module = importlib.import_module("opentelemetry.sdk.trace")
        trace_export_module = importlib.import_module("opentelemetry.sdk.trace.export")

        OTLPSpanExporter = exporter_module.OTLPSpanExporter
        FastAPIInstrumentor = instrumentation_module.FastAPIInstrumentor
        Resource = resources_module.Resource
        TracerProvider = trace_module.TracerProvider
        BatchSpanProcessor = trace_export_module.BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": settings.otel_service_name,
                "deployment.environment": settings.app_env,
            }
        )

        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info(
            "otel_tracing_initialized",
            service_name=settings.otel_service_name,
            endpoint=settings.otel_exporter_otlp_endpoint,
        )
    except Exception as exc:
        logger.warning("otel_tracing_setup_failed", error=str(exc))
