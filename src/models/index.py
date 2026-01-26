"""Index document model - flattened for search."""

import hashlib
import re
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field

from src.models.source import MenuItem, Menu, MenuGroup, Restaurant, Metadata


class Coordinates(BaseModel):
    """Geo coordinates for OpenSearch."""

    lat: float
    lon: float


class IndexDocument(BaseModel):
    """Flattened document for search indexing.

    Each MenuItem becomes one IndexDocument with denormalized restaurant/menu context.
    """

    # Document identification
    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    restaurant_id: str
    item_id: str | None = None

    # Restaurant context
    restaurant_name: str
    cuisine: list[str] = Field(default_factory=list)

    # Location
    city: str
    state: str
    zip_code: str
    coordinates: Coordinates

    # Menu context
    menu_name: str
    menu_group_name: str

    # Item details
    item_name: str
    item_description: str | None = None

    # Pricing
    base_price: float | None = None
    display_price: float | None = None
    currency: str = "USD"

    # Serving
    serves_min: int | None = None
    serves_max: int | None = None
    serving_unit: str | None = None
    serving_description: str | None = None
    minimum_order_qty: float | None = None
    minimum_order_unit: str | None = None

    # Attributes
    dietary_labels: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    # Portions & Modifiers
    has_portions: bool = False
    portion_options: list[str] = Field(default_factory=list)
    has_modifiers: bool = False
    modifier_groups: list[str] = Field(default_factory=list)

    # Metadata
    source_platform: str | None = None
    source_path: str | None = None
    content_hash: str | None = None
    scraped_at: datetime | None = None
    indexed_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def price_per_person(self) -> float | None:
        """Calculate price per person if serving info available."""
        if self.display_price and self.serves_max and self.serves_max > 0:
            return round(self.display_price / self.serves_max, 2)
        return None

    @computed_field
    @property
    def text(self) -> str:
        """Generate searchable text field."""
        parts = [self.item_name]

        if self.item_description:
            parts.append(self.item_description)

        if self.dietary_labels:
            parts.append(f"Dietary: {', '.join(self.dietary_labels)}")

        if self.serves_min and self.serves_max:
            parts.append(f"Serves {self.serves_min}-{self.serves_max} people")
        elif self.serving_description:
            parts.append(self.serving_description)

        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        return ". ".join(parts)

    @classmethod
    def from_menu_item(
        cls,
        item: MenuItem,
        restaurant: Restaurant,
        menu: Menu,
        menu_group: MenuGroup,
        metadata: Metadata | None = None,
    ) -> "IndexDocument":
        """Create an IndexDocument from a MenuItem with context."""
        # Generate restaurant_id from name + location
        location = restaurant.location
        id_source = f"{restaurant.name}|{location.address}|{location.city}"
        restaurant_id = hashlib.sha256(id_source.encode()).hexdigest()[:16]

        # Parse serving size
        serves_min, serves_max, serving_unit = cls._parse_serving_size(item)

        # Get pricing
        price = item.price
        base_price = price.base_price if price else None
        display_price = price.display_price if price else None
        currency = price.currency if price else "USD"

        # Get minimum order
        min_order = item.minimum_order
        min_qty = min_order.quantity if min_order else None
        min_unit = min_order.unit if min_order else None

        # Portions
        portion_options = [p.name for p in item.portions] if item.portions else []

        # Modifier groups
        modifier_group_names = [mg.name for mg in item.modifier_groups] if item.modifier_groups else []

        return cls(
            restaurant_id=restaurant_id,
            item_id=item.item_id,
            restaurant_name=restaurant.name,
            cuisine=restaurant.cuisine,
            city=location.city,
            state=location.state,
            zip_code=location.zip_code,
            coordinates=Coordinates(
                lat=location.coordinates.latitude,
                lon=location.coordinates.longitude,
            ),
            menu_name=menu.name,
            menu_group_name=menu_group.name,
            item_name=item.name,
            item_description=item.description,
            base_price=base_price,
            display_price=display_price,
            currency=currency,
            serves_min=serves_min,
            serves_max=serves_max,
            serving_unit=serving_unit,
            serving_description=item.serving_size.description if item.serving_size else None,
            minimum_order_qty=min_qty,
            minimum_order_unit=min_unit,
            dietary_labels=list(item.dietary_labels),
            tags=item.tags,
            has_portions=len(item.portions) > 0,
            portion_options=portion_options,
            has_modifiers=len(item.modifier_groups) > 0,
            modifier_groups=modifier_group_names,
            source_platform=metadata.source_platform if metadata else None,
            source_path=metadata.source_path if metadata else None,
            content_hash=metadata.content_hash if metadata else None,
            scraped_at=metadata.scraped_at if metadata else None,
        )

    @staticmethod
    def _parse_serving_size(item: MenuItem) -> tuple[int | None, int | None, str | None]:
        """Parse serving size from item to extract min/max people served."""
        if not item.serving_size:
            return None, None, None

        serving = item.serving_size
        unit = serving.unit

        # If amount is given and unit indicates people
        if serving.amount and unit in ("person", "people", "serves"):
            amount = int(serving.amount)
            return amount, amount, "people"

        # Try to parse from description like "Serves 10-12" or "serves 10"
        if serving.description:
            desc = serving.description.lower()

            # Match patterns like "serves 10-12", "serves 10", "10-12 people"
            range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", desc)
            if range_match:
                return int(range_match.group(1)), int(range_match.group(2)), "people"

            single_match = re.search(r"serves?\s*(\d+)", desc)
            if single_match:
                amount = int(single_match.group(1))
                return amount, amount, "people"

        # Fallback: use amount if present
        if serving.amount:
            amount = int(serving.amount)
            return amount, amount, unit

        return None, None, None

    def to_opensearch_doc(self) -> dict:
        """Convert to OpenSearch document format."""
        doc = self.model_dump(exclude={"text", "price_per_person"})
        doc["text"] = self.text
        doc["price_per_person"] = self.price_per_person
        return doc

    def to_pgvector_row(self) -> dict:
        """Convert to pgvector row format (minimal fields + embedding)."""
        return {
            "doc_id": self.doc_id,
            "restaurant_id": self.restaurant_id,
            "city": self.city,
            "base_price": self.base_price,
            "serves_max": self.serves_max,
            "dietary_labels": self.dietary_labels,
        }
