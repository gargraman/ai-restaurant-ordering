"""SQLAlchemy models for order management.

All models are imported here for Alembic auto-generation.
"""

from src.db.models.user import User, UserRole
from src.db.models.tenant import Tenant
from src.db.models.restaurant import Restaurant
from src.db.models.stripe_account import StripeAccount, StripeAccountStatus
from src.db.models.pos_connection import POSConnection, POSProvider, POSConnectionStatus
from src.db.models.menu_item import MenuItem
from src.db.models.order import Order, OrderStatus, FulfillmentType
from src.db.models.order_item import OrderItem
from src.db.models.payment import PaymentIntent, PaymentStatus
from src.db.models.refund import Refund, RefundStatus, RefundReason
from src.db.models.webhook_event import WebhookEvent, WebhookProvider, WebhookEventStatus

__all__ = [
    # User
    "User",
    "UserRole",
    # Tenant
    "Tenant",
    # Restaurant
    "Restaurant",
    # Stripe
    "StripeAccount",
    "StripeAccountStatus",
    # POS
    "POSConnection",
    "POSProvider",
    "POSConnectionStatus",
    # Menu
    "MenuItem",
    # Order
    "Order",
    "OrderStatus",
    "FulfillmentType",
    "OrderItem",
    # Payment
    "PaymentIntent",
    "PaymentStatus",
    # Refund
    "Refund",
    "RefundStatus",
    "RefundReason",
    # Webhook
    "WebhookEvent",
    "WebhookProvider",
    "WebhookEventStatus",
]
