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

        RRF(d) = Σ weight_i / (k + rank_i(d))

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            bm25_weight: Weight for BM25 rankings
            vector_weight: Weight for vector rankings

        Returns:
            Merged and re-ranked results
        """
        k = settings.rrf_k
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, dict] = {}

        # Score BM25 results
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc.get("doc_id")
            if doc_id:
                scores[doc_id] += bm25_weight / (k + rank)
                doc_map[doc_id] = doc

        # Score vector results
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = doc.get("doc_id")
            if doc_id:
                scores[doc_id] += vector_weight / (k + rank)
                # Only add to doc_map if not already there (prefer BM25 full doc)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build result list with RRF scores
        results = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = scores[doc_id]
            doc["in_bm25"] = any(d.get("doc_id") == doc_id for d in bm25_results)
            doc["in_vector"] = any(d.get("doc_id") == doc_id for d in vector_results)
            results.append(doc)

        logger.info(
            "rrf_merge_complete",
            total_unique=len(results),
            both_sources=sum(1 for r in results if r["in_bm25"] and r["in_vector"]),
        )

        return results

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


def rrf_merge_2way(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Merge BM25 and vector results using Reciprocal Rank Fusion.

    RRF(d) = sum(weight_i / (k + rank_i(d)))

    Args:
        bm25_results: Results from BM25 search (OpenSearch)
        vector_results: Results from vector search (pgvector)
        k: RRF constant (default 60)
        bm25_weight: Weight for BM25 rankings
        vector_weight: Weight for vector rankings

    Returns:
        Merged results with RRF scores and source indicators
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict] = {}
    sources: dict[str, set] = defaultdict(set)

    # Score BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += bm25_weight / (k + rank)
            doc_map[doc_id] = doc
            sources[doc_id].add("bm25")

    # Score vector results
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += vector_weight / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            sources[doc_id].add("vector")

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build result list
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        doc["sources"] = list(sources[doc_id])
        doc["in_bm25"] = "bm25" in sources[doc_id]
        doc["in_vector"] = "vector" in sources[doc_id]
        doc["in_graph"] = False
        results.append(doc)

    logger.info(
        "rrf_merge_2way_complete",
        total_unique=len(results),
        both_sources=sum(1 for r in results if r["in_bm25"] and r["in_vector"]),
    )

    return results


def rrf_merge_3way(
    bm25_results: list[dict],
    vector_results: list[dict],
    graph_results: list[dict],
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
    graph_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Merge BM25, vector, and graph results using Reciprocal Rank Fusion.

    RRF(d) = sum(weight_i / (k + rank_i(d)))

    This 3-way fusion combines:
    - BM25 (OpenSearch): Lexical/keyword matching
    - Vector (pgvector): Semantic similarity
    - Graph (Neo4j): Relationship-based relevance

    Args:
        bm25_results: Results from BM25 search (OpenSearch)
        vector_results: Results from vector search (pgvector)
        graph_results: Results from graph traversal (Neo4j)
        k: RRF constant (default 60)
        bm25_weight: Weight for BM25 rankings
        vector_weight: Weight for vector rankings
        graph_weight: Weight for graph rankings

    Returns:
        Merged results with RRF scores and source indicators
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict] = {}
    sources: dict[str, set] = defaultdict(set)

    # Score BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += bm25_weight / (k + rank)
            doc_map[doc_id] = doc
            sources[doc_id].add("bm25")

    # Score vector results
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += vector_weight / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            sources[doc_id].add("vector")

    # Score graph results
    for rank, doc in enumerate(graph_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += graph_weight / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            sources[doc_id].add("graph")

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build result list
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        doc["sources"] = list(sources[doc_id])
        doc["in_bm25"] = "bm25" in sources[doc_id]
        doc["in_vector"] = "vector" in sources[doc_id]
        doc["in_graph"] = "graph" in sources[doc_id]
        results.append(doc)

    # Log statistics
    all_three = sum(1 for r in results if r["in_bm25"] and r["in_vector"] and r["in_graph"])
    bm25_vector = sum(1 for r in results if r["in_bm25"] and r["in_vector"] and not r["in_graph"])
    bm25_graph = sum(1 for r in results if r["in_bm25"] and r["in_graph"] and not r["in_vector"])
    vector_graph = sum(1 for r in results if r["in_vector"] and r["in_graph"] and not r["in_bm25"])

    logger.info(
        "rrf_merge_3way_complete",
        total_unique=len(results),
        all_three_sources=all_three,
        bm25_vector_only=bm25_vector,
        bm25_graph_only=bm25_graph,
        vector_graph_only=vector_graph,
        bm25_only=sum(1 for r in results if r["in_bm25"] and not r["in_vector"] and not r["in_graph"]),
        vector_only=sum(1 for r in results if r["in_vector"] and not r["in_bm25"] and not r["in_graph"]),
        graph_only=sum(1 for r in results if r["in_graph"] and not r["in_bm25"] and not r["in_vector"]),
    )

    return results
