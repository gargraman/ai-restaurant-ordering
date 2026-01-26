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
