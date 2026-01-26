"""Complete ingestion pipeline orchestration."""

import asyncio
from pathlib import Path
from typing import Any, Sequence

import structlog

from src.ingestion.transformer import DocumentTransformer
from src.ingestion.embeddings import EmbeddingGenerator
from src.ingestion.indexer import OpenSearchIndexer, PgVectorIndexer
from src.models.index import IndexDocument

logger = structlog.get_logger()

# Default batch size for processing large datasets
DEFAULT_BATCH_SIZE = 1000


class IngestionPipeline:
    """Orchestrate the complete data ingestion pipeline.

    Pipeline steps:
    1. Transform: JSON files -> IndexDocuments
    2. Generate: Create embeddings for each document
    3. Index: Store in OpenSearch (BM25) and pgvector (vector)

    Supports batched processing for large datasets to manage memory usage.
    """

    def __init__(
        self,
        transformer: DocumentTransformer | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        opensearch_indexer: OpenSearchIndexer | None = None,
        pgvector_indexer: PgVectorIndexer | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Initialize the ingestion pipeline.

        Args:
            transformer: Document transformer instance
            embedding_generator: Embedding generator instance
            opensearch_indexer: OpenSearch indexer instance
            pgvector_indexer: pgvector indexer instance
            batch_size: Number of documents to process per batch (default 1000)
        """
        self.transformer = transformer or DocumentTransformer()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.opensearch_indexer = opensearch_indexer or OpenSearchIndexer()
        self.pgvector_indexer = pgvector_indexer or PgVectorIndexer()
        self.batch_size = batch_size

    async def run(
        self,
        source_path: str | Path,
        recreate_indexes: bool = False,
        skip_embeddings: bool = False,
    ) -> dict[str, Any]:
        """Run the complete ingestion pipeline.

        Args:
            source_path: Path to JSON file or directory of JSON files
            recreate_indexes: If True, delete and recreate indexes
            skip_embeddings: If True, skip embedding generation (for testing)

        Returns:
            Pipeline execution statistics
        """
        source_path = Path(source_path)
        stats: dict[str, Any] = {
            "source_path": str(source_path),
            "transform": {"document_count": 0},
            "embeddings": {"generated": 0, "skipped": False},
            "opensearch": {"success": 0, "failed": 0},
            "pgvector": {"success": 0, "failed": 0},
            "batches_processed": 0,
        }

        logger.info("pipeline_starting", source_path=str(source_path))

        # Step 1: Create/ensure indexes exist
        logger.info("step_1_create_indexes")
        self.opensearch_indexer.create_index(delete_existing=recreate_indexes)

        await self.pgvector_indexer.connect()
        await self.pgvector_indexer.create_schema()

        # Step 2: Transform and process
        logger.info("step_2_transform_and_index")

        if source_path.is_dir():
            # Process directory with batched file processing
            stats = await self._process_directory(
                source_path, stats, skip_embeddings
            )
        else:
            # Process single file
            stats = await self._process_file(
                source_path, stats, skip_embeddings
            )

        # Cleanup
        await self.pgvector_indexer.close()

        # Add transformer stats
        stats["transform"].update(self.transformer.stats)

        logger.info("pipeline_complete", stats=stats)
        return stats

    async def _process_directory(
        self,
        dir_path: Path,
        stats: dict[str, Any],
        skip_embeddings: bool,
    ) -> dict[str, Any]:
        """Process all JSON files in a directory with batching."""
        json_files = [
            f for f in dir_path.glob("*.json")
            if "schema" not in f.name.lower()
        ]

        if not json_files:
            logger.warning("no_json_files_found", path=str(dir_path))
            return stats

        logger.info("processing_directory", file_count=len(json_files))

        all_documents: list[IndexDocument] = []

        # Transform all files first
        for json_file in json_files:
            try:
                docs = self.transformer.transform_file(json_file)
                all_documents.extend(docs)
            except Exception as e:
                logger.error("file_transform_error", file=str(json_file), error=str(e))

        if not all_documents:
            logger.warning("no_documents_after_transform")
            return stats

        stats["transform"]["document_count"] = len(all_documents)

        # Process in batches
        stats = await self._process_batches(all_documents, stats, skip_embeddings)

        return stats

    async def _process_file(
        self,
        file_path: Path,
        stats: dict[str, Any],
        skip_embeddings: bool,
    ) -> dict[str, Any]:
        """Process a single JSON file."""
        documents = self.transformer.transform_file(file_path)

        if not documents:
            logger.warning("no_documents_to_index")
            return stats

        stats["transform"]["document_count"] = len(documents)

        # Process in batches
        stats = await self._process_batches(documents, stats, skip_embeddings)

        return stats

    async def _process_batches(
        self,
        documents: list[IndexDocument],
        stats: dict[str, Any],
        skip_embeddings: bool,
    ) -> dict[str, Any]:
        """Process documents in batches for memory efficiency.

        Args:
            documents: All documents to process
            stats: Statistics dictionary to update
            skip_embeddings: Whether to skip embedding generation

        Returns:
            Updated statistics
        """
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size

        logger.info(
            "batch_processing_starting",
            total_documents=len(documents),
            batch_size=self.batch_size,
            total_batches=total_batches,
        )

        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(documents))
            batch = documents[start_idx:end_idx]

            logger.info(
                "processing_batch",
                batch=batch_num + 1,
                total_batches=total_batches,
                batch_size=len(batch),
            )

            # Generate embeddings for batch
            embeddings: dict[str, list[float]] = {}
            if not skip_embeddings:
                try:
                    embeddings = await self.embedding_generator.generate_document_embeddings(batch)
                    stats["embeddings"]["generated"] += len(embeddings)
                except Exception as e:
                    logger.error(
                        "batch_embedding_generation_failed",
                        batch=batch_num + 1,
                        error=str(e),
                    )
                    # Continue with indexing even if embeddings fail
            else:
                stats["embeddings"]["skipped"] = True

            # Index batch to OpenSearch
            try:
                os_result = self.opensearch_indexer.index_documents(batch)
                stats["opensearch"]["success"] += os_result.get("success", 0)
                failed_count = len(os_result.get("failed", []))
                stats["opensearch"]["failed"] += failed_count
            except Exception as e:
                logger.error(
                    "batch_opensearch_indexing_failed",
                    batch=batch_num + 1,
                    error=str(e),
                )
                stats["opensearch"]["failed"] += len(batch)

            # Index batch to pgvector (if we have embeddings)
            if embeddings:
                try:
                    pg_result = await self.pgvector_indexer.index_documents(batch, embeddings)
                    stats["pgvector"]["success"] += pg_result.get("success", 0)
                    stats["pgvector"]["failed"] += pg_result.get("failed", 0)
                except Exception as e:
                    logger.error(
                        "batch_pgvector_indexing_failed",
                        batch=batch_num + 1,
                        error=str(e),
                    )
                    stats["pgvector"]["failed"] += len(batch)

            stats["batches_processed"] += 1

            logger.info(
                "batch_complete",
                batch=batch_num + 1,
                total_batches=total_batches,
                os_success=stats["opensearch"]["success"],
                pg_success=stats["pgvector"]["success"],
            )

        return stats

    async def ingest_documents(
        self,
        documents: Sequence[IndexDocument],
        generate_embeddings: bool = True,
    ) -> dict[str, Any]:
        """Ingest pre-transformed documents.

        Useful for testing or when documents come from a different source.

        Args:
            documents: Pre-transformed IndexDocuments
            generate_embeddings: Whether to generate embeddings

        Returns:
            Ingestion statistics
        """
        stats: dict[str, Any] = {
            "document_count": len(documents),
            "embeddings": {"generated": 0},
            "opensearch": {"success": 0, "failed": 0},
            "pgvector": {"success": 0, "failed": 0},
        }

        if not documents:
            logger.warning("no_documents_to_ingest")
            return stats

        # Generate embeddings
        embeddings: dict[str, list[float]] = {}
        if generate_embeddings:
            embeddings = await self.embedding_generator.generate_document_embeddings(documents)
            stats["embeddings"]["generated"] = len(embeddings)

        # Index to OpenSearch
        os_result = self.opensearch_indexer.index_documents(documents)
        stats["opensearch"] = {
            "success": os_result.get("success", 0),
            "failed": len(os_result.get("failed", [])),
        }

        # Index to pgvector
        if embeddings:
            await self.pgvector_indexer.connect()
            await self.pgvector_indexer.create_schema()
            pg_result = await self.pgvector_indexer.index_documents(documents, embeddings)
            stats["pgvector"] = {
                "success": pg_result.get("success", 0),
                "failed": pg_result.get("failed", 0),
            }
            await self.pgvector_indexer.close()

        return stats


async def run_ingestion(
    source_path: str,
    recreate: bool = False,
    skip_embeddings: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Convenience function to run ingestion pipeline.

    Args:
        source_path: Path to JSON file or directory
        recreate: Whether to recreate indexes
        skip_embeddings: Whether to skip embedding generation
        batch_size: Number of documents per batch

    Returns:
        Pipeline execution statistics
    """
    pipeline = IngestionPipeline(batch_size=batch_size)
    return await pipeline.run(
        source_path=source_path,
        recreate_indexes=recreate,
        skip_embeddings=skip_embeddings,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.pipeline <source_path> [--recreate] [--skip-embeddings] [--batch-size N]")
        sys.exit(1)

    source = sys.argv[1]
    recreate = "--recreate" in sys.argv
    skip_emb = "--skip-embeddings" in sys.argv

    # Parse batch size
    batch_size = DEFAULT_BATCH_SIZE
    for i, arg in enumerate(sys.argv):
        if arg == "--batch-size" and i + 1 < len(sys.argv):
            try:
                batch_size = int(sys.argv[i + 1])
            except ValueError:
                print(f"Invalid batch size: {sys.argv[i + 1]}")
                sys.exit(1)

    result = asyncio.run(run_ingestion(source, recreate, skip_emb, batch_size))
    print(f"Ingestion complete: {result}")
