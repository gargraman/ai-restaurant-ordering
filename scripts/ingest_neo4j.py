#!/usr/bin/env python3
"""Script to ingest data into Neo4j graph database."""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.neo4j_indexer import Neo4jIndexer
from src.models.source import RestaurantData


def load_restaurant_data(source_path: Path) -> list[RestaurantData]:
    """Load restaurant data from JSON files.

    Args:
        source_path: Path to JSON file or directory

    Returns:
        List of RestaurantData objects
    """
    restaurants = []

    if source_path.is_file():
        files = [source_path]
    else:
        files = list(source_path.glob("**/*.json"))

    for file_path in files:
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Handle both single restaurant and array of restaurants
            if isinstance(data, list):
                for item in data:
                    restaurants.append(RestaurantData.model_validate(item))
            else:
                restaurants.append(RestaurantData.model_validate(data))

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    return restaurants


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest restaurant data into Neo4j graph database"
    )
    parser.add_argument(
        "source",
        help="Path to JSON file or directory of JSON files",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing Neo4j data before indexing",
    )
    parser.add_argument(
        "--create-relationships",
        action="store_true",
        help="Create PAIRS_WITH and SIMILAR_TO relationships after indexing",
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=10.0,
        help="Maximum distance for similar restaurant relationships (default: 10km)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}")
        sys.exit(1)

    print(f"Loading data from: {source_path}")
    restaurants = load_restaurant_data(source_path)
    print(f"Loaded {len(restaurants)} restaurants")

    if not restaurants:
        print("No restaurant data found")
        sys.exit(1)

    indexer = Neo4jIndexer()

    try:
        print("\nConnecting to Neo4j...")
        await indexer.connect()

        print("Creating schema...")
        await indexer.create_schema()

        if args.clear:
            print("Clearing existing data...")
            await indexer.clear_all()

        print(f"\nIndexing {len(restaurants)} restaurants...")
        stats = await indexer.index_restaurants(restaurants)

        print("\n" + "=" * 50)
        print("INDEXING COMPLETE")
        print("=" * 50)
        print(f"Restaurants: {stats['restaurants']}")
        print(f"Menus: {stats['menus']}")
        print(f"Menu Groups: {stats['groups']}")
        print(f"Menu Items: {stats['items']}")
        print(f"Cuisines: {stats['cuisines']}")

        if args.create_relationships:
            print("\nCreating relationships...")

            print("  Creating PAIRS_WITH relationships...")
            pairing_count = await indexer.create_pairing_relationships()
            print(f"  Created {pairing_count} pairing relationships")

            print(f"  Creating SIMILAR_TO relationships (max {args.max_distance_km}km)...")
            similar_count = await indexer.create_similar_restaurant_relationships(
                max_distance_km=args.max_distance_km
            )
            print(f"  Created {similar_count} similar restaurant relationships")

        # Final stats
        print("\nFinal database stats:")
        final_stats = await indexer.get_stats()
        for key, value in final_stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\nError during ingestion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    finally:
        await indexer.close()


if __name__ == "__main__":
    asyncio.run(main())
