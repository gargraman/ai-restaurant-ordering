"""API routers for the Order Management system."""

from src.api.routers.auth import router as auth_router
from src.api.routers.orders import router as orders_router
from src.api.routers.restaurants import router as restaurants_router
from src.api.routers.tenants import router as tenants_router
from src.api.routers.webhooks import router as webhooks_router

__all__ = [
    "auth_router",
    "tenants_router",
    "restaurants_router",
    "orders_router",
    "webhooks_router",
]
