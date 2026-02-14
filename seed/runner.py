"""Seed runner that orchestrates all entity seeders in dependency order."""

import importlib
import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from seed.context import SeedContext
from seed.registry import SEED_REGISTRY, resolve_dependencies


class SeedRunner:
    """Orchestrates seed data generation across all entities.

    Runs seeders in dependency order within a single transaction
    so all data is consistent and connected.
    """

    def __init__(self, session: AsyncSession, dry_run: bool = False):
        self.session = session
        self.dry_run = dry_run
        self.ctx = SeedContext()
        self.stats: dict[str, int] = {}

    async def clear_all(self) -> None:
        """Delete all existing data in reverse dependency order.

        Uses TRUNCATE CASCADE for speed and to handle FK constraints.
        """
        tables = [
            "webhook_events",
            "refunds",
            "payment_intents",
            "order_items",
            "orders",
            "pos_connections",
            "stripe_accounts",
            "menu_items",
            "users",
            "restaurants",
            "tenants",
        ]
        for table in tables:
            await self.session.execute(text(f'TRUNCATE TABLE "{table}" CASCADE'))
        await self.session.flush()
        print("  Cleared all existing data.")

    async def run(self, entities: list[str] | None = None, clear: bool = False) -> dict[str, int]:
        """Run seed data generation.

        Args:
            entities: Optional list of specific entities to seed.
                      If None, seeds all entities.
            clear: Whether to clear existing data first.

        Returns:
            Dict mapping entity name to number of records created.
        """
        if entities:
            ordered = resolve_dependencies(entities)
        else:
            ordered = list(SEED_REGISTRY)

        print(f"\nSeeding {len(ordered)} entities: {', '.join(ordered)}")
        if self.dry_run:
            print("  (DRY RUN - no data will be written)\n")
        print()

        if clear and not self.dry_run:
            await self.clear_all()

        total_start = time.time()

        for entity_name in ordered:
            start = time.time()
            module = importlib.import_module(f"seed.seeders.{entity_name}")
            count = await module.seed(self.session, self.ctx)
            elapsed = time.time() - start
            self.stats[entity_name] = count
            print(f"  {entity_name:<20s} {count:>4d} records  ({elapsed:.2f}s)")

        if not self.dry_run:
            await self.session.commit()
        else:
            await self.session.rollback()

        total_elapsed = time.time() - total_start
        total_records = sum(self.stats.values())
        print(f"\n  Total: {total_records} records in {total_elapsed:.2f}s")

        return self.stats
