"""Tests for BM25 search implementation."""

import pytest
from unittest.mock import MagicMock, patch

from src.search.bm25 import BM25Searcher


class TestBM25Searcher:
    """Tests for BM25Searcher class."""

    @pytest.fixture
    def mock_opensearch_client(self):
        """Create a mock OpenSearch client."""
        client = MagicMock()
        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "doc_id": "doc-1",
                            "item_name": "Pasta Tray",
                            "restaurant_name": "Italian Kitchen",
                        },
                        "_score": 5.5,
                    },
                    {
                        "_source": {
                            "doc_id": "doc-2",
                            "item_name": "Pizza Platter",
                            "restaurant_name": "Italian Kitchen",
                        },
                        "_score": 4.2,
                    },
                ]
            }
        }
        return client

    @pytest.fixture
    def bm25_searcher(self, mock_opensearch_client):
        """Create a BM25Searcher with mock client."""
        with patch("src.search.bm25.OpenSearch") as mock_class:
            mock_class.return_value = mock_opensearch_client
            searcher = BM25Searcher()
            # Replace client with our mock
            searcher.client = mock_opensearch_client
            return searcher

    def test_search_returns_results_with_scores(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that search returns results with scores."""
        results = bm25_searcher.search("italian food")

        assert len(results) == 2
        assert results[0]["doc_id"] == "doc-1"
        assert results[0]["_score"] == 5.5
        assert results[1]["doc_id"] == "doc-2"
        assert results[1]["_score"] == 4.2

    def test_search_builds_correct_query(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that search builds the correct OpenSearch query."""
        bm25_searcher.search("italian food", top_k=10)

        mock_opensearch_client.search.assert_called_once()
        call_args = mock_opensearch_client.search.call_args

        # Check index
        assert call_args[1]["index"] == "catering_menus"

        # Check query structure
        body = call_args[1]["body"]
        assert "query" in body
        assert body["size"] == 10

        # Check multi_match query
        query = body["query"]["bool"]["must"]["multi_match"]
        assert query["query"] == "italian food"
        assert "item_name^3" in query["fields"]
        assert "item_description^2" in query["fields"]
        assert "text" in query["fields"]
        assert query["type"] == "best_fields"
        assert query["fuzziness"] == "AUTO"

    def test_search_applies_city_filter(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that city filter is applied."""
        bm25_searcher.search("italian food", filters={"city": "Boston"})

        body = mock_opensearch_client.search.call_args[1]["body"]
        filters = body["query"]["bool"]["filter"]

        assert any(f["term"].get("city") == "Boston" for f in filters)

    def test_search_applies_multiple_filters(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that multiple filters are applied."""
        filters = {
            "city": "Boston",
            "state": "MA",
            "cuisine": ["Italian"],
            "price_max": 100.0,
            "dietary_labels": ["vegetarian"],
        }

        bm25_searcher.search("italian food", filters=filters)

        body = mock_opensearch_client.search.call_args[1]["body"]
        filter_clauses = body["query"]["bool"]["filter"]

        # Check all filters are present
        filter_dict = {}
        for clause in filter_clauses:
            if "term" in clause:
                key = list(clause["term"].keys())[0]
                filter_dict[key] = clause["term"][key]
            elif "terms" in clause:
                key = list(clause["terms"].keys())[0]
                filter_dict[key] = clause["terms"][key]
            elif "range" in clause:
                key = list(clause["range"].keys())[0]
                filter_dict[key] = clause["range"][key]

        assert filter_dict.get("city") == "Boston"
        assert filter_dict.get("state") == "MA"
        assert filter_dict.get("cuisine") == ["Italian"]
        assert "display_price" in filter_dict  # price_max
        assert filter_dict.get("dietary_labels") == ["vegetarian"]

    def test_search_applies_price_range_filter(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that price filter uses range query."""
        bm25_searcher.search("italian food", filters={"price_max": 100.0})

        body = mock_opensearch_client.search.call_args[1]["body"]
        filter_clauses = body["query"]["bool"]["filter"]

        price_filter = next(
            f for f in filter_clauses
            if "range" in f and "display_price" in f["range"]
        )
        assert price_filter["range"]["display_price"]["lte"] == 100.0

    def test_search_applies_serving_size_filters(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that serving size filters are applied."""
        filters = {
            "serves_min": 10,
            "serves_max": 50,
        }

        bm25_searcher.search("catering", filters=filters)

        body = mock_opensearch_client.search.call_args[1]["body"]
        filter_clauses = body["query"]["bool"]["filter"]

        # serves_min should filter serves_max >= value
        serves_min_filter = next(
            f for f in filter_clauses
            if "range" in f and "serves_max" in f["range"]
        )
        assert serves_min_filter["range"]["serves_max"]["gte"] == 10

        # serves_max should filter serves_min <= value
        serves_max_filter = next(
            f for f in filter_clauses
            if "range" in f and "serves_min" in f["range"]
        )
        assert serves_max_filter["range"]["serves_min"]["lte"] == 50

    def test_search_applies_restaurant_name_filter(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that restaurant name filter uses fuzzy match."""
        bm25_searcher.search("pasta", filters={"restaurant_name": "Italian Kitchen"})

        body = mock_opensearch_client.search.call_args[1]["body"]
        filter_clauses = body["query"]["bool"]["filter"]

        restaurant_filter = next(
            f for f in filter_clauses
            if "match" in f and "restaurant_name" in f["match"]
        )
        assert restaurant_filter["match"]["restaurant_name"]["query"] == "Italian Kitchen"
        assert restaurant_filter["match"]["restaurant_name"]["fuzziness"] == "AUTO"

    def test_search_applies_restaurant_id_filter(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that restaurant_id filter uses exact match."""
        bm25_searcher.search("pasta", filters={"restaurant_id": "rest-123"})

        body = mock_opensearch_client.search.call_args[1]["body"]
        filter_clauses = body["query"]["bool"]["filter"]

        restaurant_filter = next(
            f for f in filter_clauses
            if "term" in f and "restaurant_id" in f["term"]
        )
        assert restaurant_filter["term"]["restaurant_id"] == "rest-123"

    def test_search_applies_exclude_restaurant_filter(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that exclude_restaurant_id uses must_not."""
        bm25_searcher.search("pasta", filters={"exclude_restaurant_id": "rest-123"})

        body = mock_opensearch_client.search.call_args[1]["body"]
        filter_clauses = body["query"]["bool"]["filter"]

        exclude_filter = next(
            f for f in filter_clauses
            if "bool" in f and "must_not" in f["bool"]
        )
        assert exclude_filter["bool"]["must_not"]["term"]["restaurant_id"] == "rest-123"

    def test_search_applies_menu_type_filter(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that menu_type filter is applied."""
        bm25_searcher.search("pasta", filters={"menu_type": "Catering"})

        body = mock_opensearch_client.search.call_args[1]["body"]
        filter_clauses = body["query"]["bool"]["filter"]

        menu_filter = next(
            f for f in filter_clauses
            if "term" in f and "menu_name" in f["term"]
        )
        assert menu_filter["term"]["menu_name"] == "Catering"

    def test_search_empty_filters(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that search works with empty filters."""
        bm25_searcher.search("italian food", filters={})

        body = mock_opensearch_client.search.call_args[1]["body"]
        filter_clauses = body["query"]["bool"]["filter"]

        # Should have no filters
        assert filter_clauses == []

    def test_search_handles_opensearch_error(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that OpenSearch errors are handled gracefully."""
        mock_opensearch_client.search.side_effect = Exception("Connection refused")

        results = bm25_searcher.search("test")

        # Should return empty list on error
        assert results == []

    def test_search_by_ids_fetches_documents(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that search_by_ids fetches documents by ID."""
        mock_opensearch_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "doc_id": "doc-1",
                            "item_name": "Pasta",
                        }
                    },
                    {
                        "_source": {
                            "doc_id": "doc-2",
                            "item_name": "Pizza",
                        }
                    },
                ]
            }
        }

        results = bm25_searcher.search_by_ids(["doc-1", "doc-2"])

        assert len(results) == 2
        assert results[0]["doc_id"] == "doc-1"
        assert results[1]["doc_id"] == "doc-2"

    def test_search_by_ids_empty_list(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that search_by_ids returns empty for empty list."""
        results = bm25_searcher.search_by_ids([])

        assert results == []
        mock_opensearch_client.search.assert_not_called()

    def test_search_by_ids_handles_error(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that search_by_ids handles errors gracefully."""
        mock_opensearch_client.search.side_effect = Exception("Connection refused")

        results = bm25_searcher.search_by_ids(["doc-1"])

        # Should return empty list on error
        assert results == []

    def test_get_document_fetches_single_doc(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that get_document fetches a single document."""
        mock_opensearch_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "doc_id": "doc-1",
                            "item_name": "Pasta",
                        }
                    }
                ]
            }
        }

        doc = bm25_searcher.get_document("doc-1")

        assert doc is not None
        assert doc["doc_id"] == "doc-1"

    def test_get_document_not_found(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that get_document returns None for missing doc."""
        mock_opensearch_client.search.return_value = {
            "hits": {"hits": []}
        }

        doc = bm25_searcher.get_document("nonexistent")

        assert doc is None

    def test_build_filters_zip_code(
        self,
        bm25_searcher,
    ):
        """Test that zip_code filter is built correctly."""
        filters = bm25_searcher._build_filters({"zip_code": "02101"})

        assert len(filters) == 1
        assert filters[0]["term"]["zip_code"] == "02101"

    def test_build_filters_tags(
        self,
        bm25_searcher,
    ):
        """Test that tags filter uses terms query."""
        filters = bm25_searcher._build_filters({"tags": ["popular", "spicy"]})

        assert len(filters) == 1
        assert filters[0]["terms"]["tags"] == ["popular", "spicy"]

    def test_search_uses_default_top_k_from_settings(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that search uses top_k from settings when not specified."""
        with patch("src.search.bm25.settings") as mock_settings:
            mock_settings.bm25_top_k = 25

            bm25_searcher.search("test")

            body = mock_opensearch_client.search.call_args[1]["body"]
            assert body["size"] == 25

    def test_search_records_metrics_on_success(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that search records metrics on success."""
        with patch("src.search.bm25.record_search_request") as mock_record:
            bm25_searcher.search("test")

            mock_record.assert_called()
            call_args = mock_record.call_args
            assert call_args[0][0] == "bm25"  # search_type
            # Arguments are (search_type, duration, result_count)
            assert call_args[0][1] > 0  # duration
            assert call_args[0][2] == 2  # result_count

    def test_search_records_metrics_on_error(
        self,
        bm25_searcher,
        mock_opensearch_client,
    ):
        """Test that search records metrics on error."""
        mock_opensearch_client.search.side_effect = Exception("Error")

        with patch("src.search.bm25.record_search_request") as mock_record:
            bm25_searcher.search("test")

            mock_record.assert_called()
            call_args = mock_record.call_args
            assert call_args[0][0] == "bm25"
            # Arguments are (search_type, duration, result_count)
            assert call_args[0][1] > 0  # duration
            assert call_args[0][2] == 0  # result_count (failed)
