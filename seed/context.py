"""Seed context that holds references to all created entities.

This context is passed between seeders so they can reference
each other's created records (e.g., orders reference restaurants,
order_items reference menu_items).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SeedContext:
    """Shared context holding all seeded entity IDs and references.

    Each seeder populates its section, and downstream seeders
    read from upstream sections to build connected data.
    """

    # tenant slug -> tenant record dict (id, name, slug, ...)
    tenants: dict[str, dict[str, Any]] = field(default_factory=dict)

    # user email -> user record dict (id, email, role, tenant_id, ...)
    users: dict[str, dict[str, Any]] = field(default_factory=dict)

    # restaurant slug -> restaurant record dict (id, tenant_id, name, ...)
    restaurants: dict[str, dict[str, Any]] = field(default_factory=dict)

    # (restaurant_slug, item_name) -> menu_item record dict
    menu_items: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    # restaurant slug -> stripe_account record dict
    stripe_accounts: dict[str, dict[str, Any]] = field(default_factory=dict)

    # restaurant slug -> pos_connection record dict
    pos_connections: dict[str, dict[str, Any]] = field(default_factory=dict)

    # order_number -> order record dict
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)

    # (order_number, item_name) -> order_item record dict
    order_items: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    # order_number -> payment_intent record dict
    payment_intents: dict[str, dict[str, Any]] = field(default_factory=dict)

    # stripe_refund_id -> refund record dict
    refunds: dict[str, dict[str, Any]] = field(default_factory=dict)

    # provider_event_id -> webhook_event record dict
    webhook_events: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Counters for generating sequential IDs
    _order_counter: int = 0
    _webhook_counter: int = 0

    def next_order_number(self, tenant_slug: str) -> str:
        """Generate the next order number for a tenant."""
        self._order_counter += 1
        prefix = tenant_slug[:3].upper()
        return f"{prefix}-{self._order_counter:04d}"

    def next_webhook_id(self, provider: str) -> str:
        """Generate the next webhook event ID."""
        self._webhook_counter += 1
        return f"evt_{provider}_{self._webhook_counter:06d}"
