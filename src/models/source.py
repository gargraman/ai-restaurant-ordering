"""Source data models matching the restaurant-schema.json."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MenuCounts(BaseModel):
    """Count of menus, groups, and items."""

    menus: int = 0
    menu_groups: int = Field(default=0, alias="menuGroups")
    menu_items: int = Field(default=0, alias="menuItems")

    class Config:
        populate_by_name = True


class Metadata(BaseModel):
    """Scraping and normalization metadata."""

    schema_version: str = Field(default="0.1.0", alias="schemaVersion")
    source_platform: str | None = Field(default=None, alias="sourcePlatform")
    source_path: str | None = Field(default=None, alias="sourcePath")
    engine: str | None = None
    scraped_at: datetime | None = Field(default=None, alias="scrapedAt")
    normalized_at: datetime | None = Field(default=None, alias="normalizedAt")
    scrape_run_id: str | None = Field(default=None, alias="scrapeRunId")
    content_hash: str | None = Field(default=None, alias="contentHash")
    scraped_menu_counts: MenuCounts | None = Field(default=None, alias="scrapedMenuCounts")
    actual_menu_counts: MenuCounts | None = Field(default=None, alias="actualMenuCounts")

    class Config:
        populate_by_name = True


class Coordinates(BaseModel):
    """Geographic coordinates."""

    latitude: float
    longitude: float


class Location(BaseModel):
    """Restaurant location information."""

    address: str
    city: str
    state: str
    zip_code: str = Field(alias="zipCode")
    country: str | None = None
    coordinates: Coordinates

    class Config:
        populate_by_name = True


class Restaurant(BaseModel):
    """Restaurant information."""

    name: str
    cuisine: list[str] = Field(default_factory=list)
    location: Location


class Price(BaseModel):
    """Pricing information."""

    base_price: float | None = Field(default=None, alias="basePrice")
    display_price: float | None = Field(default=None, alias="displayPrice")
    currency: str = "USD"

    class Config:
        populate_by_name = True


class Image(BaseModel):
    """Image information."""

    url: str
    alt_text: str | None = Field(default=None, alias="altText")

    class Config:
        populate_by_name = True


class ServingSize(BaseModel):
    """Serving size information."""

    amount: float | None = None
    unit: str | None = None
    description: str | None = None


class MinimumOrder(BaseModel):
    """Minimum order requirement."""

    quantity: float | None = None
    unit: str | None = None


class Modifier(BaseModel):
    """Individual modifier option."""

    modifier_id: str | None = Field(default=None, alias="modifierId")
    name: str
    description: str | None = None
    price: Price | None = None
    display_order: int | None = Field(default=None, alias="displayOrder")
    is_default: bool = Field(default=False, alias="isDefault")

    class Config:
        populate_by_name = True


class ModifierGroup(BaseModel):
    """Group of modifiers for customization."""

    modifier_group_id: str | None = Field(default=None, alias="modifierGroupId")
    name: str
    description: str | None = None
    display_order: int | None = Field(default=None, alias="displayOrder")
    selection_type: Literal["single", "multiple"] = Field(
        default="multiple", alias="selectionType"
    )
    is_required: bool = Field(default=False, alias="isRequired")
    min_selections: int = Field(default=0, alias="minSelections")
    max_selections: int | None = Field(default=None, alias="maxSelections")
    default_selections: list[str] = Field(default_factory=list, alias="defaultSelections")
    modifiers: list[Modifier] = Field(default_factory=list)
    applies_to: Literal["menuGroup", "menuItem"] | None = Field(default=None, alias="appliesTo")

    class Config:
        populate_by_name = True


class Portion(BaseModel):
    """Different size options for a menu item."""

    portion_id: str | None = Field(default=None, alias="portionId")
    name: str
    description: str | None = None
    price: Price
    serving_size: ServingSize | None = Field(default=None, alias="servingSize")
    is_default: bool = Field(default=False, alias="isDefault")
    display_order: int | None = Field(default=None, alias="displayOrder")

    class Config:
        populate_by_name = True


DietaryLabel = Literal[
    "vegetarian",
    "vegan",
    "gluten-free",
    "dairy-free",
    "nut-free",
    "halal",
    "kosher",
    "organic",
    "keto",
    "paleo",
    "low-carb",
]


class MenuItem(BaseModel):
    """Individual menu offering."""

    item_id: str | None = Field(default=None, alias="itemId")
    name: str
    description: str | None = None
    price: Price | None = None
    dietary_labels: list[DietaryLabel] = Field(default_factory=list, alias="dietaryLabels")
    images: list[Image] = Field(default_factory=list)
    display_order: int | None = Field(default=None, alias="displayOrder")
    serving_size: ServingSize | None = Field(default=None, alias="servingSize")
    minimum_order: MinimumOrder | None = Field(default=None, alias="minimumOrder")
    modifier_groups: list[ModifierGroup] = Field(default_factory=list, alias="modifierGroups")
    portions: list[Portion] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class MenuGroup(BaseModel):
    """Group of related menu items."""

    group_id: str | None = Field(default=None, alias="groupId")
    name: str
    description: str | None = None
    display_order: int | None = Field(default=None, alias="displayOrder")
    menu_items: list[MenuItem] = Field(default_factory=list, alias="menuItems")

    class Config:
        populate_by_name = True


class Menu(BaseModel):
    """Top-level menu category."""

    menu_id: str | None = Field(default=None, alias="menuId")
    name: str
    description: str | None = None
    display_order: int | None = Field(default=None, alias="displayOrder")
    menu_groups: list[MenuGroup] = Field(default_factory=list, alias="menuGroups")

    class Config:
        populate_by_name = True


class RestaurantData(BaseModel):
    """Complete restaurant data structure (root document)."""

    metadata: Metadata | None = None
    restaurant: Restaurant
    menus: list[Menu] = Field(min_length=1)
