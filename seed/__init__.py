"""Seed data module for populating all database entities with realistic, connected data.

Usage:
    python -m seed                          # Seed all entities
    python -m seed --entities tenants users  # Seed specific entities
    python -m seed --clear                   # Clear existing data first
    python -m seed --dry-run                 # Preview without writing
"""

from seed.registry import SEED_REGISTRY

__all__ = ["SEED_REGISTRY"]
