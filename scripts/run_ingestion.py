#!/usr/bin/env python3
"""Script to run the data ingestion pipeline."""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline, DEFAULT_BATCH_SIZE


async def main():
    parser = argparse.ArgumentParser(
        description="Run the data ingestion pipeline for catering menu data"
    )
    parser.add_argument(
        "source",
        help="Path to JSON file or directory of JSON files",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate indexes (delete existing data)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (for testing without OpenAI API)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of documents per batch (default: {DEFAULT_BATCH_SIZE})",
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

    print(f"Starting ingestion from: {source_path}")
    print(f"Options:")
    print(f"  - Recreate indexes: {args.recreate}")
    print(f"  - Skip embeddings: {args.skip_embeddings}")
    print(f"  - Batch size: {args.batch_size}")
    print()

    pipeline = IngestionPipeline(batch_size=args.batch_size)

    try:
        stats = await pipeline.run(
            source_path=source_path,
            recreate_indexes=args.recreate,
            skip_embeddings=args.skip_embeddings,
        )

        print("\n" + "=" * 50)
        print("INGESTION COMPLETE")
        print("=" * 50)
        print(f"Documents transformed: {stats['transform'].get('document_count', 0)}")
        print(f"  - Restaurants: {stats['transform'].get('restaurants_processed', 0)}")
        print(f"  - Menus: {stats['transform'].get('menus_processed', 0)}")
        print(f"  - Menu groups: {stats['transform'].get('groups_processed', 0)}")
        print(f"  - Menu items: {stats['transform'].get('items_processed', 0)}")
        print()
        print(f"Batches processed: {stats.get('batches_processed', 0)}")
        print()
        print(f"OpenSearch indexing:")
        print(f"  - Success: {stats['opensearch'].get('success', 0)}")
        print(f"  - Failed: {stats['opensearch'].get('failed', 0)}")
        print()
        if stats['embeddings'].get('skipped'):
            print("Embeddings: SKIPPED")
        else:
            print(f"Embeddings generated: {stats['embeddings'].get('generated', 0)}")
        print()
        print(f"pgvector indexing:")
        print(f"  - Success: {stats['pgvector'].get('success', 0)}")
        print(f"  - Failed: {stats['pgvector'].get('failed', 0)}")

    except Exception as e:
        print(f"\nError during ingestion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
