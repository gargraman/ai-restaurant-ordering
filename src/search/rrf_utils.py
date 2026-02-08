"""Utilities for Reciprocal Rank Fusion (RRF) merging of search results."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import structlog

from src.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


def compute_rrf_scores(
    results_list: List[List[Dict[str, Any]]],
    weights: List[float],
    source_names: List[str],
    k: int = 60,
) -> tuple[Dict[str, float], Dict[str, Dict], Dict[str, Set[str]]]:
    """Compute RRF scores for multiple result sets.
    
    Args:
        results_list: List of result lists from different sources
        weights: Weights for each source
        source_names: Names of each source (e.g., ["bm25", "vector", "graph"])
        k: RRF constant (default 60)
        
    Returns:
        Tuple of (scores, doc_map, sources)
    """
    scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Dict] = {}
    sources: Dict[str, Set[str]] = defaultdict(set)

    for results, weight, source_name in zip(results_list, weights, source_names):
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get("doc_id")
            if doc_id:
                scores[doc_id] += weight / (k + rank)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
                sources[doc_id].add(source_name)

    return scores, doc_map, sources


def format_rrf_results(
    scores: Dict[str, float],
    doc_map: Dict[str, Dict],
    sources: Dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    """Format RRF results with scores and source information.
    
    Args:
        scores: Dictionary mapping doc_id to RRF score
        doc_map: Dictionary mapping doc_id to document
        sources: Dictionary mapping doc_id to set of sources
        
    Returns:
        List of formatted results with RRF scores and source indicators
    """
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

    return results


def rrf_merge_2way(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> List[Dict[str, Any]]:
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
    results_list = [bm25_results, vector_results]
    weights = [bm25_weight, vector_weight]
    source_names = ["bm25", "vector"]

    scores, doc_map, sources = compute_rrf_scores(results_list, weights, source_names, k)
    merged_results = format_rrf_results(scores, doc_map, sources)

    logger.info(
        "rrf_merge_2way_complete",
        total_unique=len(merged_results),
        both_sources=sum(1 for r in merged_results if r["in_bm25"] and r["in_vector"]),
    )

    return merged_results


def rrf_merge_3way(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    graph_results: List[Dict[str, Any]],
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
    graph_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """Merge BM25, vector, and graph results using Reciprocal Rank Fusion.
    
    RRF(d) = sum(weight_i / (k + rank_i(d)))
    
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
    results_list = [bm25_results, vector_results, graph_results]
    weights = [bm25_weight, vector_weight, graph_weight]
    source_names = ["bm25", "vector", "graph"]

    scores, doc_map, sources = compute_rrf_scores(results_list, weights, source_names, k)
    merged_results = format_rrf_results(scores, doc_map, sources)

    # Log statistics
    all_three = sum(1 for r in merged_results if r["in_bm25"] and r["in_vector"] and r["in_graph"])
    bm25_vector = sum(1 for r in merged_results if r["in_bm25"] and r["in_vector"] and not r["in_graph"])
    bm25_graph = sum(1 for r in merged_results if r["in_bm25"] and r["in_graph"] and not r["in_vector"])
    vector_graph = sum(1 for r in merged_results if r["in_vector"] and r["in_graph"] and not r["in_bm25"])

    logger.info(
        "rrf_merge_3way_complete",
        total_unique=len(merged_results),
        all_three_sources=all_three,
        bm25_vector_only=bm25_vector,
        bm25_graph_only=bm25_graph,
        vector_graph_only=vector_graph,
        bm25_only=sum(1 for r in merged_results if r["in_bm25"] and not r["in_vector"] and not r["in_graph"]),
        vector_only=sum(1 for r in merged_results if r["in_vector"] and not r["in_bm25"] and not r["in_graph"]),
        graph_only=sum(1 for r in merged_results if r["in_graph"] and not r["in_bm25"] and not r["in_vector"]),
    )

    return merged_results


def rrf_merge_simple(
    *result_lists: List[Dict[str, Any]],
    weights: Optional[List[float]] = None,
    source_names: Optional[List[str]] = None,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Generic RRF merge function for any number of result lists.
    
    Args:
        *result_lists: Variable number of result lists to merge
        weights: Optional weights for each source (defaults to 1.0 for each)
        source_names: Optional names for each source (defaults to indexed names)
        k: RRF constant (default 60)
        
    Returns:
        Merged results with RRF scores
    """
    num_sources = len(result_lists)
    
    if weights is None:
        weights = [1.0] * num_sources
    elif len(weights) != num_sources:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of result lists ({num_sources})")
    
    if source_names is None:
        source_names = [f"source_{i}" for i in range(num_sources)]
    elif len(source_names) != num_sources:
        raise ValueError(f"Number of source names ({len(source_names)}) must match number of result lists ({num_sources})")
    
    scores, doc_map, sources = compute_rrf_scores(list(result_lists), weights, source_names, k)
    merged_results = format_rrf_results(scores, doc_map, sources)
    
    logger.info(
        "rrf_merge_generic_complete",
        total_unique=len(merged_results),
        num_sources=num_sources,
    )
    
    return merged_results