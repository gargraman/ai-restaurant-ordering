"""POS (Point of Sale) integration module.

This module provides adapters for various POS systems to:
- Sync menus from POS to local database
- Inject orders into POS systems
- Handle status updates via webhooks
"""

from src.pos.base import POSAdapter
from src.pos.exceptions import (
    POSError,
    POSAuthError,
    POSRateLimitError,
    POSValidationError,
    POSNotFoundError,
    POSConnectionError,
)
from src.pos.models import (
    POSOrderRequest,
    POSOrderResult,
    POSOrderStatus,
    POSCatalogItem,
    POSCatalogSyncResult,
)
from src.pos.registry import POSProvider, POSRegistry, get_adapter

__all__ = [
    # Base class
    "POSAdapter",
    # Exceptions
    "POSError",
    "POSAuthError",
    "POSRateLimitError",
    "POSValidationError",
    "POSNotFoundError",
    "POSConnectionError",
    # Models
    "POSOrderRequest",
    "POSOrderResult",
    "POSOrderStatus",
    "POSCatalogItem",
    "POSCatalogSyncResult",
    # Registry
    "POSProvider",
    "POSRegistry",
    "get_adapter",
]
