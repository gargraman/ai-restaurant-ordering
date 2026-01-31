"""Tests for graph query detection."""

import pytest

from src.langgraph.nodes import _detect_graph_query


class TestGraphQueryDetection:
    """Tests for _detect_graph_query function."""

    def test_detects_restaurant_items_with_previous_results(self):
        """Test detection of 'more from this restaurant' queries."""
        queries = [
            "show me more from this restaurant",
            "what else do they have",
            "other items from this place",
            "show me their full menu",
        ]
        for query in queries:
            requires_graph, query_type = _detect_graph_query(query, has_previous_results=True)
            assert requires_graph is True, f"Failed for: {query}"
            assert query_type == "restaurant_items", f"Wrong type for: {query}"

    def test_no_restaurant_items_without_previous_results(self):
        """Test that restaurant_items requires previous results."""
        requires_graph, query_type = _detect_graph_query(
            "show me more from this restaurant",
            has_previous_results=False,
        )
        assert requires_graph is False

    def test_detects_similar_restaurants(self):
        """Test detection of 'similar restaurants' queries."""
        queries = [
            "similar restaurants nearby",
            "restaurants like this one",
            "other restaurants in the area",
            "alternatives to this place",
        ]
        for query in queries:
            requires_graph, query_type = _detect_graph_query(query, has_previous_results=True)
            assert requires_graph is True, f"Failed for: {query}"
            assert query_type == "similar_restaurants", f"Wrong type for: {query}"

    def test_detects_pairing_queries(self):
        """Test detection of pairing queries."""
        queries = [
            "what pairs well with this",
            "something that goes with the pasta",
            "sides for the chicken",
            "complementary dishes",
        ]
        for query in queries:
            requires_graph, query_type = _detect_graph_query(query, has_previous_results=True)
            assert requires_graph is True, f"Failed for: {query}"
            assert query_type == "pairing", f"Wrong type for: {query}"

    def test_detects_catering_packages(self):
        """Test detection of catering package queries."""
        queries = [
            "complete catering package for 50 people",
            "full meal for a party",
            "appetizer and entree and dessert",
        ]
        for query in queries:
            requires_graph, query_type = _detect_graph_query(query, has_previous_results=False)
            assert requires_graph is True, f"Failed for: {query}"
            assert query_type == "catering_packages", f"Wrong type for: {query}"

    def test_no_graph_for_regular_search(self):
        """Test that regular searches don't trigger graph queries."""
        queries = [
            "italian food in boston",
            "vegetarian options for 20 people",
            "chicken parmesan under $100",
            "catering menus near me",
        ]
        for query in queries:
            requires_graph, query_type = _detect_graph_query(query, has_previous_results=False)
            assert requires_graph is False, f"Should not trigger graph for: {query}"
            assert query_type is None

    def test_case_insensitivity(self):
        """Test that detection is case insensitive."""
        requires_graph, query_type = _detect_graph_query(
            "SHOW ME MORE FROM THIS RESTAURANT",
            has_previous_results=True,
        )
        assert requires_graph is True
        assert query_type == "restaurant_items"


class TestGraphQueryTypes:
    """Test specific graph query type patterns."""

    @pytest.mark.parametrize(
        "query,expected_type",
        [
            ("more from this restaurant", "restaurant_items"),
            ("what else do they have", "restaurant_items"),
            ("similar restaurants", "similar_restaurants"),
            ("restaurants like this", "similar_restaurants"),
            ("what pairs with this", "pairing"),
            ("sides to go with", "pairing"),
            ("full catering package", "catering_packages"),
            ("appetizer and dessert combo", "catering_packages"),
        ],
    )
    def test_query_type_mapping(self, query, expected_type):
        """Test that queries map to correct types."""
        requires_graph, query_type = _detect_graph_query(query, has_previous_results=True)
        assert requires_graph is True
        assert query_type == expected_type
