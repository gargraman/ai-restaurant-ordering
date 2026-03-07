"""Seed registry defining entity order and dependencies.

Entities must be seeded in dependency order to satisfy foreign key constraints.
"""

# Ordered list of entity seeders - each entry is (name, module_path)
# Order matters: parents before children
SEED_REGISTRY = [
    "tenants",
    "users",
    "restaurants",
    "menu_items",
    "stripe_accounts",
    "pos_connections",
    "orders",
    "order_items",
    "payment_intents",
    "refunds",
    "webhook_events",
]

# Entity dependencies (for --entities partial seeding)
ENTITY_DEPENDENCIES: dict[str, list[str]] = {
    "tenants": [],
    "users": ["tenants", "restaurants"],
    "restaurants": ["tenants"],
    "menu_items": ["restaurants"],
    "stripe_accounts": ["restaurants"],
    "pos_connections": ["restaurants"],
    "orders": ["tenants", "restaurants", "users"],
    "order_items": ["orders", "menu_items"],
    "payment_intents": ["orders"],
    "refunds": ["orders"],
    "webhook_events": [],
}


def resolve_dependencies(entities: list[str]) -> list[str]:
    """Resolve and return entities in correct dependency order.

    Args:
        entities: List of entity names to seed

    Returns:
        Ordered list including all required dependencies
    """
    resolved: list[str] = []
    seen: set[str] = set()

    def _resolve(entity: str) -> None:
        if entity in seen:
            return
        seen.add(entity)
        for dep in ENTITY_DEPENDENCIES.get(entity, []):
            _resolve(dep)
        resolved.append(entity)

    for entity in entities:
        _resolve(entity)

    return resolved
