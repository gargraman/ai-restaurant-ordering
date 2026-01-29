"""BM25 lexical search via OpenSearch."""

from typing import Any

import structlog
from opensearchpy import OpenSearch

from src.config import get_settings
from src.models.state import SearchFilters

logger = structlog.get_logger()
settings = get_settings()


class BM25Searcher:
    """Execute BM25 lexical search via OpenSearch."""

    def __init__(self, client: OpenSearch | None = None):
        if client:
            self.client = client
        else:
            self.client = OpenSearch(
                hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
                http_auth=(settings.opensearch_user, settings.opensearch_password),
                use_ssl=settings.opensearch_use_ssl,
                verify_certs=False,
                ssl_show_warn=False,
            )
        self.index_name = settings.opensearch_index

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute BM25 search with optional filters.

        Args:
            query: Search query text
            filters: Optional filters to apply
            top_k: Number of results to return

        Returns:
            List of matching documents with scores
        """
        top_k = top_k or settings.bm25_top_k
        filters = filters or {}

        logger.info("bm25_search", query=query, filters=filters, top_k=top_k)

        # Build filter clauses
        must_filters = self._build_filters(filters)

        # Build search query
        body = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "item_name^3",
                                "item_description^2",
                                "text",
                                "restaurant_name",
                                "menu_group_name",
                            ],
                            "type": "best_fields",
                            "fuzziness": "AUTO",
                        }
                    },
                    "filter": must_filters,
                }
            },
            "size": top_k,
        }

        try:
            response = self.client.search(index=self.index_name, body=body)

            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["_score"] = hit["_score"]
                results.append(doc)

            logger.info("bm25_search_complete", result_count=len(results))
            return results

        except Exception as e:
            logger.error("bm25_search_error", error=str(e))
            return []

    def _build_filters(self, filters: SearchFilters) -> list[dict]:
        """Build OpenSearch filter clauses."""
        must_filters = []

        if filters.get("city"):
            must_filters.append({"term": {"city": filters["city"]}})

        if filters.get("state"):
            must_filters.append({"term": {"state": filters["state"]}})

        if filters.get("zip_code"):
            must_filters.append({"term": {"zip_code": filters["zip_code"]}})

        if filters.get("cuisine"):
            must_filters.append({"terms": {"cuisine": filters["cuisine"]}})

        if filters.get("dietary_labels"):
            must_filters.append({"terms": {"dietary_labels": filters["dietary_labels"]}})

        if filters.get("price_max"):
            must_filters.append({"range": {"display_price": {"lte": filters["price_max"]}}})

        if filters.get("price_per_person_max"):
            must_filters.append(
                {"range": {"price_per_person": {"lte": filters["price_per_person_max"]}}}
            )

        if filters.get("serves_min"):
            must_filters.append({"range": {"serves_max": {"gte": filters["serves_min"]}}})

        if filters.get("serves_max"):
            must_filters.append({"range": {"serves_min": {"lte": filters["serves_max"]}}})

        if filters.get("tags"):
            must_filters.append({"terms": {"tags": filters["tags"]}})

        if filters.get("restaurant_name"):
            must_filters.append(
                {"match": {"restaurant_name": {"query": filters["restaurant_name"], "fuzziness": "AUTO"}}}
            )

        if filters.get("restaurant_id"):
            must_filters.append({"term": {"restaurant_id": filters["restaurant_id"]}})

        if filters.get("menu_type"):
            must_filters.append({"term": {"menu_name": filters["menu_type"]}})

        if filters.get("exclude_restaurant_id"):
            must_filters.append({
                "bool": {
                    "must_not": {"term": {"restaurant_id": filters["exclude_restaurant_id"]}}
                }
            })

        return must_filters

    def search_by_ids(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch documents by their IDs."""
        if not doc_ids:
            return []

        body = {"query": {"terms": {"doc_id": doc_ids}}, "size": len(doc_ids)}

        try:
            response = self.client.search(index=self.index_name, body=body)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error("fetch_by_ids_error", error=str(e))
            return []

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Get a single document by ID."""
        results = self.search_by_ids([doc_id])
        return results[0] if results else None
