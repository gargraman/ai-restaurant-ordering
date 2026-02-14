#!/usr/bin/env python3
"""CLI entry point for seed data generation.

Usage:
    python -m seed                              # Seed all entities
    python -m seed --entities tenants users      # Seed specific entities (+ deps)
    python -m seed --clear                       # Clear existing data first
    python -m seed --dry-run                     # Preview without writing
    python -m seed --clear --entities orders     # Clear all, seed orders + deps
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seed.registry import SEED_REGISTRY


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed database with realistic, connected test data."
    )
    parser.add_argument(
        "--entities",
        nargs="+",
        choices=SEED_REGISTRY,
        help="Specific entities to seed (dependencies auto-included). Default: all.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="TRUNCATE all tables before seeding.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run seeders but rollback instead of committing.",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Create tables before seeding (for fresh databases).",
    )

    args = parser.parse_args()

    # Late imports to avoid import errors when just checking --help
    from src.db.base import AsyncSessionLocal, init_db
    from seed.runner import SeedRunner

    if args.init_db:
        print("Initializing database tables...")
        await init_db()
        print("  Done.\n")

    print("=" * 55)
    print("  SEED DATA GENERATOR")
    print("=" * 55)

    async with AsyncSessionLocal() as session:
        runner = SeedRunner(session, dry_run=args.dry_run)
        stats = await runner.run(entities=args.entities, clear=args.clear)

    print("\n" + "=" * 55)
    if args.dry_run:
        print("  DRY RUN COMPLETE - no data was written.")
    else:
        print("  SEEDING COMPLETE")
    print("=" * 55)

    # Print summary table
    print(f"\n  {'Entity':<20s} {'Records':>8s}")
    print(f"  {'-'*20} {'-'*8}")
    for entity, count in stats.items():
        print(f"  {entity:<20s} {count:>8d}")
    print(f"  {'-'*20} {'-'*8}")
    print(f"  {'TOTAL':<20s} {sum(stats.values()):>8d}")

    print("\n  Default credentials:")
    print("  Email: admin@platform.com")
    print("  Password: SeedPass123!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
