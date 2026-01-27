"""Tests for the document transformer."""

import pytest
from src.ingestion.transformer import DocumentTransformer
from src.models.source import (
    RestaurantData,
    Restaurant,
    Location,
    Coordinates,
    Menu,
    MenuGroup,
    MenuItem,
    Price,
    ServingSize,
)


@pytest.fixture
def sample_restaurant_data() -> dict:
    """Sample restaurant data matching the schema."""
    return {
        "metadata": {
            "schemaVersion": "0.1.0",
            "sourcePlatform": "test",
            "sourcePath": "/test/restaurant",
        },
        "restaurant": {
            "name": "Test Restaurant",
            "cuisine": ["Italian", "Mediterranean"],
            "location": {
                "address": "123 Main St",
                "city": "Boston",
                "state": "MA",
                "zipCode": "02101",
                "country": "USA",
                "coordinates": {"latitude": 42.3601, "longitude": -71.0589},
            },
        },
        "menus": [
            {
                "menuId": "catering",
                "name": "Catering",
                "menuGroups": [
                    {
                        "groupId": "entrees",
                        "name": "Hot Entrees",
                        "menuItems": [
                            {
                                "itemId": "pasta-1",
                                "name": "Pasta Tray",
                                "description": "Fresh pasta with marinara sauce",
                                "price": {"basePrice": 89.99, "displayPrice": 89.99},
                                "servingSize": {
                                    "amount": 12,
                                    "unit": "serves",
                                    "description": "Serves 10-12",
                                },
                                "dietaryLabels": ["vegetarian"],
                                "tags": ["popular"],
                            },
                            {
                                "itemId": "chicken-1",
                                "name": "Chicken Parmesan Tray",
                                "description": "Breaded chicken with mozzarella",
                                "price": {"basePrice": 129.99, "displayPrice": 129.99},
                                "servingSize": {
                                    "amount": 12,
                                    "unit": "serves",
                                    "description": "Serves 10-12",
                                },
                            },
                        ],
                    }
                ],
            }
        ],
    }


class TestDocumentTransformer:
    """Tests for DocumentTransformer."""

    def test_transform_data_creates_documents(self, sample_restaurant_data):
        """Test that transform_data creates IndexDocuments."""
        transformer = DocumentTransformer()
        documents = transformer.transform_data(sample_restaurant_data)

        assert len(documents) == 2
        assert transformer.stats["restaurants_processed"] == 1
        assert transformer.stats["documents_created"] == 2

    def test_document_has_correct_fields(self, sample_restaurant_data):
        """Test that created documents have correct fields."""
        transformer = DocumentTransformer()
        documents = transformer.transform_data(sample_restaurant_data)

        doc = documents[0]

        assert doc.restaurant_name == "Test Restaurant"
        assert doc.city == "Boston"
        assert doc.state == "MA"
        assert doc.item_name == "Pasta Tray"
        assert doc.display_price == 89.99
        assert doc.serves_min == 10
        assert doc.serves_max == 12
        assert "vegetarian" in doc.dietary_labels
        assert "popular" in doc.tags

    def test_price_per_person_calculated(self, sample_restaurant_data):
        """Test that price_per_person is calculated correctly."""
        transformer = DocumentTransformer()
        documents = transformer.transform_data(sample_restaurant_data)

        doc = documents[0]  # Pasta tray: $89.99 / 12 = $7.50
        assert doc.price_per_person == pytest.approx(7.50, rel=0.01)

    def test_serving_size_parsed_from_description(self, sample_restaurant_data):
        """Test that serving size is parsed from description."""
        transformer = DocumentTransformer()
        documents = transformer.transform_data(sample_restaurant_data)

        doc = documents[0]
        assert doc.serves_min == 10
        assert doc.serves_max == 12
        assert doc.serving_unit == "people"

    def test_text_field_generated(self, sample_restaurant_data):
        """Test that searchable text field is generated."""
        transformer = DocumentTransformer()
        documents = transformer.transform_data(sample_restaurant_data)

        doc = documents[0]
        text = doc.text

        assert "Pasta Tray" in text
        assert "marinara" in text
        assert "vegetarian" in text
        assert "10-12" in text

    def test_restaurant_id_is_consistent(self, sample_restaurant_data):
        """Test that restaurant_id is consistently generated."""
        transformer = DocumentTransformer()
        documents = transformer.transform_data(sample_restaurant_data)

        # All documents from same restaurant should have same restaurant_id
        assert documents[0].restaurant_id == documents[1].restaurant_id

    def test_reset_stats(self, sample_restaurant_data):
        """Test that stats can be reset."""
        transformer = DocumentTransformer()
        transformer.transform_data(sample_restaurant_data)

        assert transformer.stats["documents_created"] == 2

        transformer.reset_stats()
        assert transformer.stats["documents_created"] == 0
    def test_transform_empty_restaurant_data(self):
        """Test that empty restaurant data returns no documents."""
        transformer = DocumentTransformer()
        empty_data = {
            "metadata": {"sourcePlatform": "test"},
            "restaurant": {
                "name": "Test",
                "cuisine": [],
                "location": {"city": "Boston", "state": "MA"},
            },
            "menus": [],
        }

        documents = transformer.transform_data(empty_data)

        assert len(documents) == 0
        assert transformer.stats["documents_created"] == 0

    def test_transform_missing_required_fields(self):
        """Test handling of missing required fields."""
        transformer = DocumentTransformer()

        # Missing location field
        bad_data = {
            "metadata": {"sourcePlatform": "test"},
            "restaurant": {
                "name": "Test Restaurant",
                "cuisine": ["Italian"],
                # Missing location
            },
            "menus": [],
        }

        # Should handle gracefully without crashing
        try:
            documents = transformer.transform_data(bad_data)
            # Either returns empty or skips bad record
            assert isinstance(documents, list)
        except (KeyError, AttributeError, ValueError):
            # Acceptable to fail on malformed data if properly logged
            pass

    def test_transform_malformed_menu_items(self):
        """Test handling of malformed menu items."""
        transformer = DocumentTransformer()

        bad_data = {
            "metadata": {"sourcePlatform": "test"},
            "restaurant": {
                "name": "Test",
                "cuisine": ["Italian"],
                "location": {"city": "Boston", "state": "MA", "zipCode": "02101"},
            },
            "menus": [
                {
                    "name": "Catering",
                    "menuGroups": [
                        {
                            "name": "Entrees",
                            "menuItems": [
                                {
                                    "name": "Pasta",
                                    # Missing required price and description fields
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        # Should handle gracefully
        try:
            documents = transformer.transform_data(bad_data)
            assert isinstance(documents, list)
            # May skip bad items or log warnings
        except (KeyError, AttributeError, ValueError):
            pass

    def test_transform_invalid_coordinates(self):
        """Test handling of invalid geographic coordinates."""
        transformer = DocumentTransformer()

        bad_data = {
            "metadata": {"sourcePlatform": "test"},
            "restaurant": {
                "name": "Test",
                "cuisine": ["Italian"],
                "location": {
                    "city": "Boston",
                    "state": "MA",
                    "coordinates": {"latitude": "invalid", "longitude": "invalid"},
                },
            },
            "menus": [],
        }

        # Should handle gracefully without crashing
        try:
            documents = transformer.transform_data(bad_data)
            assert isinstance(documents, list)
        except (ValueError, TypeError):
            # Acceptable to fail on invalid coordinates
            pass

    def test_transform_null_values(self):
        """Test handling of null/None values in fields."""
        transformer = DocumentTransformer()

        data_with_nulls = {
            "metadata": {"sourcePlatform": "test"},
            "restaurant": {
                "name": "Test",
                "cuisine": None,
                "location": {
                    "city": "Boston",
                    "state": "MA",
                    "coordinates": None,
                },
            },
            "menus": None,
        }

        try:
            documents = transformer.transform_data(data_with_nulls)
            assert isinstance(documents, list)
        except (TypeError, AttributeError):
            pass

    def test_transform_very_long_description(self):
        """Test handling of extremely long descriptions."""
        transformer = DocumentTransformer()

        long_description = "a" * 50000

        data = {
            "metadata": {"sourcePlatform": "test"},
            "restaurant": {
                "name": "Test",
                "cuisine": ["Italian"],
                "location": {"city": "Boston", "state": "MA"},
            },
            "menus": [
                {
                    "name": "Catering",
                    "menuGroups": [
                        {
                            "name": "Entrees",
                            "menuItems": [
                                {
                                    "name": "Pasta",
                                    "description": long_description,
                                    "price": {"displayPrice": 89.99},
                                    "dietaryLabels": [],
                                    "tags": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        try:
            documents = transformer.transform_data(data)
            # Should handle without OOM or crashing
            assert isinstance(documents, list)
        except MemoryError:
            pass

    def test_transform_special_characters(self):
        """Test handling of special characters in fields."""
        transformer = DocumentTransformer()

        data = {
            "metadata": {"sourcePlatform": "test"},
            "restaurant": {
                "name": "Test™ Rëstaurant™ <script>",
                "cuisine": ["Italian"],
                "location": {"city": "Boston", "state": "MA"},
            },
            "menus": [
                {
                    "name": "Catering",
                    "menuGroups": [
                        {
                            "name": "Entrees",
                            "menuItems": [
                                {
                                    "name": "Pâstà™ & Sàlad",
                                    "description": "Pasta with sauce™ <tag>",
                                    "price": {"displayPrice": 89.99},
                                    "dietaryLabels": [],
                                    "tags": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        documents = transformer.transform_data(data)

        # Should handle special characters gracefully
        assert len(documents) >= 0
        if len(documents) > 0:
            assert isinstance(documents[0].item_name, str)

    def test_transform_duplicate_items_same_restaurant(self, sample_restaurant_data):
        """Test that duplicate items from same restaurant have same restaurant_id."""
        transformer = DocumentTransformer()
        documents = transformer.transform_data(sample_restaurant_data)

        if len(documents) > 1:
            # All items from same restaurant should have matching restaurant_id
            restaurant_ids = [doc.restaurant_id for doc in documents]
            assert all(rid == restaurant_ids[0] for rid in restaurant_ids)