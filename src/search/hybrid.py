"""Hybrid search combining BM25 and vector search with RRF fusion."""

import asyncio
import time
from collections import defaultdict
from typing import Any

import structlog

from src.config import get_settings
from src.models.state import SearchFilters
from src.search.bm25 import BM25Searcher
from src.search.vector import VectorSearcher
from src.search.rrf_utils import rrf_merge_2way
from src.metrics import record_search_request

logger = structlog.get_logger()
settings = get_settings()


class HybridSearcher:
    """Hybrid search combining BM25 lexical and vector semantic search."""

    def __init__(
        self,
        bm25_searcher: BM25Searcher | None = None,
        vector_searcher: VectorSearcher | None = None,
    ):
        self.bm25_searcher = bm25_searcher or BM25Searcher()
        self.vector_searcher = vector_searcher or VectorSearcher()

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        """Execute hybrid search with RRF fusion.

        Args:
            query: Search query text
            filters: Optional filters to apply
            top_k: Number of final results to return
            bm25_weight: Weight for BM25 results in RRF
            vector_weight: Weight for vector results in RRF

        Returns:
            List of documents ranked by RRF score
        """
        top_k = top_k or 10
        bm25_weight = bm25_weight or settings.bm25_weight
        vector_weight = vector_weight or settings.vector_weight
        filters = filters or {}

        logger.info(
            "hybrid_search",
            query=query[:50],
            filters=filters,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )

        start_time = time.time()
        try:
            # Run BM25 and vector searches in parallel
            bm25_task = asyncio.create_task(
                asyncio.to_thread(
                    self.bm25_searcher.search,
                    query,
                    filters,
                    settings.bm25_top_k,
                )
            )
            vector_task = asyncio.create_task(
                self.vector_searcher.search(query, filters, settings.vector_top_k)
            )

            bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

            logger.info(
                "search_results",
                bm25_count=len(bm25_results),
                vector_count=len(vector_results),
            )

            # Merge with RRF
            merged = self._rrf_merge(
                bm25_results,
                vector_results,
                bm25_weight,
                vector_weight,
            )

            # Fetch full documents for vector-only results
            merged = await self._enrich_results(merged, bm25_results)

            # Return top_k
            result = merged[:top_k]

            duration = time.time() - start_time
            record_search_request('hybrid', duration, len(result))

            logger.info("hybrid_search_complete", result_count=len(result), duration=duration)
            return result

        except Exception as e:
            duration = time.time() - start_time
            record_search_request('hybrid', duration, 0)  # Record failed search
            logger.error("hybrid_search_error", error=str(e), duration=duration)
            raise

    def _rrf_merge(
        self,
        bm25_results: list[dict],
        vector_results: list[dict],
        bm25_weight: float,
        vector_weight: float,
    ) -> list[dict[str, Any]]:
        """Merge results using Reciprocal Rank Fusion.

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            bm25_weight: Weight for BM25 rankings
            vector_weight: Weight for vector rankings

        Returns:
            Merged and re-ranked results
        """
        k = settings.rrf_k
        merged = rrf_merge_2way(
            bm25_results,
            vector_results,
            k=k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )

        logger.info(
            "rrf_merge_complete",
            total_unique=len(merged),
            both_sources=sum(1 for r in merged if r["in_bm25"] and r["in_vector"]),
        )

        return merged

    async def _enrich_results(
        self,
        merged: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """Enrich vector-only results with full document data.

        Vector search only returns minimal fields. For documents that
        weren't in BM25 results, fetch full data from OpenSearch.
        """
        bm25_ids = {doc.get("doc_id") for doc in bm25_results}
        vector_only_ids = [
            doc["doc_id"] for doc in merged
            if doc.get("doc_id") and doc["doc_id"] not in bm25_ids
        ]

        if not vector_only_ids:
            return merged

        # Fetch full documents for vector-only results
        full_docs = self.bm25_searcher.search_by_ids(vector_only_ids)
        full_doc_map = {doc["doc_id"]: doc for doc in full_docs}

        # Enrich merged results
        enriched = []
        for doc in merged:
            doc_id = doc.get("doc_id")
            if doc_id in full_doc_map:
                # Merge vector result with full document
                full_doc = full_doc_map[doc_id].copy()
                full_doc["rrf_score"] = doc.get("rrf_score", 0)
                full_doc["in_bm25"] = doc.get("in_bm25", False)
                full_doc["in_vector"] = doc.get("in_vector", False)
                enriched.append(full_doc)
            else:
                enriched.append(doc)

        return enriched

    async def close(self) -> None:
        """Close connections."""
        await self.vector_searcher.close()


