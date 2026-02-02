"""Square catalog sync to local menu_items."""

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.menu_item import MenuItem
from src.pos.models import (
    POSCatalogItem,
    POSCatalogModifier,
    POSCatalogModifierList,
    POSCatalogSyncResult,
)
from src.pos.square.client import SquareClient

logger = structlog.get_logger(__name__)


class SquareCatalogSync:
    """Sync Square catalog to local menu_items table.

    Handles fetching catalog items from Square and upserting them
    into the local database with proper pos_sku_mapping.

    Usage:
        sync = SquareCatalogSync(client)
        items = await sync.fetch_items(location_id)
        result = await sync.sync_to_database(db, restaurant_id, location_id)
    """

    def __init__(self, client: SquareClient) -> None:
        """Initialize catalog sync.

        Args:
            client: Configured Square client
        """
        self.client = client
        self.logger = logger.bind(component="square_catalog_sync")

    async def fetch_items(
        self,
        location_id: str,
    ) -> list[POSCatalogItem]:
        """Fetch all catalog items from Square.

        Args:
            location_id: Square location ID

        Returns:
            List of normalized catalog items
        """
        self.logger.info("Fetching catalog items", location_id=location_id)

        items: list[POSCatalogItem] = []
        modifier_lists: dict[str, POSCatalogModifierList] = {}
        images: dict[str, str] = {}

        cursor: str | None = None

        while True:
            # Fetch catalog objects
            result = await self.client.list_catalog_items(
                types=["ITEM", "MODIFIER_LIST", "IMAGE"],
                cursor=cursor,
            )

            objects = result.get("objects", [])

            for obj in objects:
                obj_type = obj.get("type")

                if obj_type == "IMAGE":
                    # Cache images for later use
                    image_id = obj.get("id")
                    image_data = obj.get("image_data", {})
                    if image_id and image_data.get("url"):
                        images[image_id] = image_data["url"]

                elif obj_type == "MODIFIER_LIST":
                    # Parse modifier lists
                    mod_list = self._parse_modifier_list(obj)
                    modifier_lists[mod_list.pos_modifier_list_id] = mod_list

                elif obj_type == "ITEM":
                    # Parse items (will process after we have all modifier lists)
                    pass

            cursor = result.get("cursor")
            if not cursor:
                break

        # Second pass: process items with modifier lists resolved
        cursor = None
        while True:
            result = await self.client.list_catalog_items(
                types=["ITEM"],
                cursor=cursor,
            )

            objects = result.get("objects", [])

            for obj in objects:
                if obj.get("type") == "ITEM":
                    # Check if item is available at this location
                    if not self._is_available_at_location(obj, location_id):
                        continue

                    parsed_items = self._parse_item(obj, modifier_lists, images, location_id)
                    items.extend(parsed_items)

            cursor = result.get("cursor")
            if not cursor:
                break

        self.logger.info(
            "Catalog fetch complete",
            location_id=location_id,
            items_count=len(items),
        )

        return items

    def _is_available_at_location(
        self,
        item: dict[str, Any],
        location_id: str,
    ) -> bool:
        """Check if item is available at the specified location.

        Args:
            item: Square item object
            location_id: Location to check

        Returns:
            True if available
        """
        # Check present_at_all_locations flag
        if item.get("present_at_all_locations", True):
            absent_at = item.get("absent_at_location_ids", [])
            return location_id not in absent_at

        # Check present_at_location_ids
        present_at = item.get("present_at_location_ids", [])
        return location_id in present_at

    def _parse_modifier_list(
        self,
        obj: dict[str, Any],
    ) -> POSCatalogModifierList:
        """Parse Square modifier list to canonical format.

        Args:
            obj: Square modifier list object

        Returns:
            Parsed modifier list
        """
        mod_list_data = obj.get("modifier_list_data", {})

        modifiers: list[POSCatalogModifier] = []
        for mod in mod_list_data.get("modifiers", []):
            mod_data = mod.get("modifier_data", {})
            price_money = mod_data.get("price_money", {})

            modifiers.append(POSCatalogModifier(
                pos_modifier_id=mod.get("id", ""),
                pos_modifier_list_id=obj.get("id"),
                name=mod_data.get("name", ""),
                price_cents=price_money.get("amount", 0),
                is_default=mod_data.get("on_by_default", False),
            ))

        selection_type = mod_list_data.get("selection_type", "SINGLE")

        return POSCatalogModifierList(
            pos_modifier_list_id=obj.get("id", ""),
            name=mod_list_data.get("name", ""),
            selection_type="multiple" if selection_type == "MULTIPLE" else "single",
            min_selections=mod_list_data.get("min_selected_modifiers", 0),
            max_selections=mod_list_data.get("max_selected_modifiers"),
            modifiers=modifiers,
        )

    def _parse_item(
        self,
        obj: dict[str, Any],
        modifier_lists: dict[str, POSCatalogModifierList],
        images: dict[str, str],
        location_id: str,
    ) -> list[POSCatalogItem]:
        """Parse Square item to canonical format.

        Square items can have multiple variations (sizes, etc.).
        Each variation becomes a separate catalog item.

        Args:
            obj: Square item object
            modifier_lists: Resolved modifier lists
            images: Resolved image URLs
            location_id: Location ID for filtering

        Returns:
            List of parsed catalog items (one per variation)
        """
        item_data = obj.get("item_data", {})
        item_id = obj.get("id", "")
        item_name = item_data.get("name", "")
        description = item_data.get("description", "")
        category_id = item_data.get("category_id")

        # Get image URL
        image_ids = item_data.get("image_ids", [])
        image_url = images.get(image_ids[0]) if image_ids else None

        # Parse modifiers
        modifier_list_info = item_data.get("modifier_list_info", [])
        item_modifiers: list[dict[str, Any]] = []

        for mod_info in modifier_list_info:
            mod_list_id = mod_info.get("modifier_list_id")
            if mod_list_id and mod_list_id in modifier_lists:
                mod_list = modifier_lists[mod_list_id]
                item_modifiers.append({
                    "name": mod_list.name,
                    "pos_modifier_list_id": mod_list.pos_modifier_list_id,
                    "selection_type": mod_list.selection_type,
                    "required": mod_info.get("min_selected_modifiers", 0) > 0,
                    "min_selections": mod_info.get("min_selected_modifiers", mod_list.min_selections),
                    "max_selections": mod_info.get("max_selected_modifiers", mod_list.max_selections),
                    "options": [
                        {
                            "pos_modifier_id": m.pos_modifier_id,
                            "name": m.name,
                            "price_cents": m.price_cents,
                            "is_default": m.is_default,
                        }
                        for m in mod_list.modifiers
                    ],
                })

        # Parse variations
        variations = item_data.get("variations", [])
        items: list[POSCatalogItem] = []

        for var in variations:
            var_data = var.get("item_variation_data", {})
            var_id = var.get("id", "")

            # Skip if not available at location
            if not self._is_available_at_location(var, location_id):
                continue

            # Get price
            price_money = var_data.get("price_money", {})
            price_cents = price_money.get("amount", 0)

            # Variation name - append to item name if not default
            var_name = var_data.get("name", "")
            full_name = item_name
            if var_name and var_name.lower() not in ("regular", "default", "standard"):
                full_name = f"{item_name} - {var_name}"

            # Create content hash for change detection
            content_hash = self._create_content_hash({
                "name": full_name,
                "description": description,
                "price_cents": price_cents,
                "modifiers": item_modifiers,
            })

            items.append(POSCatalogItem(
                pos_item_id=item_id,
                pos_variation_id=var_id,
                name=full_name,
                description=description or None,
                category=category_id,  # Will resolve to name in sync
                price_cents=price_cents,
                is_available=True,
                image_url=image_url,
                modifiers=item_modifiers,
                content_hash=content_hash,
                raw_data={
                    "item_id": item_id,
                    "variation_id": var_id,
                    "item_data": item_data,
                    "variation_data": var_data,
                },
            ))

        return items

    def _create_content_hash(self, data: dict[str, Any]) -> str:
        """Create hash of item content for change detection.

        Args:
            data: Data to hash

        Returns:
            SHA256 hash string
        """
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    async def sync_to_database(
        self,
        db: AsyncSession,
        restaurant_id: str,
        location_id: str,
    ) -> POSCatalogSyncResult:
        """Sync Square catalog to local menu_items table.

        Args:
            db: Database session
            restaurant_id: Restaurant ID for menu items
            location_id: Square location ID

        Returns:
            Sync result with counts
        """
        self.logger.info(
            "Starting catalog sync",
            restaurant_id=restaurant_id,
            location_id=location_id,
        )

        try:
            # Fetch items from Square
            catalog_items = await self.fetch_items(location_id)

            # Get existing menu items for this restaurant
            result = await db.execute(
                select(MenuItem).where(MenuItem.restaurant_id == restaurant_id)
            )
            existing_items = {
                self._get_menu_item_key(item): item
                for item in result.scalars().all()
            }

            # Track seen items for deactivation
            seen_keys: set[str] = set()

            created = 0
            updated = 0

            for catalog_item in catalog_items:
                key = f"square:{catalog_item.pos_item_id}:{catalog_item.pos_variation_id or 'default'}"
                seen_keys.add(key)

                existing = existing_items.get(key)

                if existing:
                    # Update existing item if changed
                    if existing.pos_item_hash != catalog_item.content_hash:
                        self._update_menu_item(existing, catalog_item)
                        updated += 1
                else:
                    # Create new item
                    menu_item = self._create_menu_item(
                        restaurant_id,
                        catalog_item,
                    )
                    db.add(menu_item)
                    created += 1

            # Deactivate items no longer in catalog
            deactivated = 0
            for key, item in existing_items.items():
                if key not in seen_keys and item.is_available:
                    # Only deactivate Square-synced items
                    if item.pos_sku_mapping and "square" in item.pos_sku_mapping:
                        item.is_available = False
                        deactivated += 1

            await db.flush()

            result = POSCatalogSyncResult(
                success=True,
                items_synced=len(catalog_items),
                items_created=created,
                items_updated=updated,
                items_deactivated=deactivated,
                synced_at=datetime.utcnow(),
            )

            self.logger.info(
                "Catalog sync complete",
                restaurant_id=restaurant_id,
                created=created,
                updated=updated,
                deactivated=deactivated,
            )

            return result

        except Exception as e:
            self.logger.error(
                "Catalog sync failed",
                restaurant_id=restaurant_id,
                error=str(e),
            )
            return POSCatalogSyncResult(
                success=False,
                errors=[str(e)],
            )

    def _get_menu_item_key(self, item: MenuItem) -> str:
        """Get unique key for menu item based on POS mapping.

        Args:
            item: Menu item

        Returns:
            Unique key string
        """
        if item.pos_sku_mapping and "square" in item.pos_sku_mapping:
            square = item.pos_sku_mapping["square"]
            item_id = square.get("item_id", "")
            var_id = square.get("variation_id", "default")
            return f"square:{item_id}:{var_id}"
        return f"local:{item.id}"

    def _create_menu_item(
        self,
        restaurant_id: str,
        catalog_item: POSCatalogItem,
    ) -> MenuItem:
        """Create a new MenuItem from catalog item.

        Args:
            restaurant_id: Restaurant ID
            catalog_item: Parsed catalog item

        Returns:
            New MenuItem instance
        """
        return MenuItem(
            restaurant_id=restaurant_id,
            name=catalog_item.name,
            description=catalog_item.description,
            category=catalog_item.category,
            price=Decimal(catalog_item.price_cents) / 100,
            price_per_person=Decimal(catalog_item.price_per_person_cents) / 100 if catalog_item.price_per_person_cents else None,
            is_available=catalog_item.is_available,
            image_url=catalog_item.image_url,
            modifiers=catalog_item.modifiers,
            serves_min=catalog_item.serves_min,
            serves_max=catalog_item.serves_max,
            pos_sku_mapping={
                "square": {
                    "item_id": catalog_item.pos_item_id,
                    "variation_id": catalog_item.pos_variation_id,
                }
            },
            pos_synced_at=datetime.utcnow().isoformat(),
            pos_item_hash=catalog_item.content_hash,
        )

    def _update_menu_item(
        self,
        item: MenuItem,
        catalog_item: POSCatalogItem,
    ) -> None:
        """Update existing MenuItem from catalog item.

        Args:
            item: Existing menu item
            catalog_item: Updated catalog data
        """
        item.name = catalog_item.name
        item.description = catalog_item.description
        item.category = catalog_item.category
        item.price = Decimal(catalog_item.price_cents) / 100
        if catalog_item.price_per_person_cents:
            item.price_per_person = Decimal(catalog_item.price_per_person_cents) / 100
        item.is_available = catalog_item.is_available
        item.image_url = catalog_item.image_url
        item.modifiers = catalog_item.modifiers
        item.serves_min = catalog_item.serves_min
        item.serves_max = catalog_item.serves_max

        # Update POS mapping
        if item.pos_sku_mapping is None:
            item.pos_sku_mapping = {}
        item.pos_sku_mapping["square"] = {
            "item_id": catalog_item.pos_item_id,
            "variation_id": catalog_item.pos_variation_id,
        }
        item.pos_synced_at = datetime.utcnow().isoformat()
        item.pos_item_hash = catalog_item.content_hash
