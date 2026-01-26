"""Transform hierarchical restaurant data to flat index documents."""

import json
from pathlib import Path
from typing import Iterator

import structlog

from src.models.source import RestaurantData
from src.models.index import IndexDocument

logger = structlog.get_logger()


class DocumentTransformer:
    """Transform hierarchical restaurant JSON to flat IndexDocuments."""

    def __init__(self):
        self._stats = {
            "restaurants_processed": 0,
            "menus_processed": 0,
            "groups_processed": 0,
            "items_processed": 0,
            "documents_created": 0,
        }

    @property
    def stats(self) -> dict:
        """Get transformation statistics."""
        return self._stats.copy()

    def transform_file(self, file_path: str | Path) -> list[IndexDocument]:
        """Transform a single JSON file to IndexDocuments."""
        file_path = Path(file_path)

        logger.info("transforming_file", file_path=str(file_path))

        with open(file_path, "r") as f:
            data = json.load(f)

        return self.transform_data(data)

    def transform_data(self, data: dict) -> list[IndexDocument]:
        """Transform restaurant data dict to IndexDocuments."""
        restaurant_data = RestaurantData.model_validate(data)
        return list(self.transform_restaurant(restaurant_data))

    def transform_restaurant(self, data: RestaurantData) -> Iterator[IndexDocument]:
        """Transform a RestaurantData object to IndexDocuments."""
        self._stats["restaurants_processed"] += 1

        restaurant = data.restaurant
        metadata = data.metadata

        logger.info(
            "transforming_restaurant",
            restaurant_name=restaurant.name,
            city=restaurant.location.city,
            menu_count=len(data.menus),
        )

        for menu in data.menus:
            self._stats["menus_processed"] += 1

            for menu_group in menu.menu_groups:
                self._stats["groups_processed"] += 1

                for item in menu_group.menu_items:
                    self._stats["items_processed"] += 1

                    try:
                        doc = IndexDocument.from_menu_item(
                            item=item,
                            restaurant=restaurant,
                            menu=menu,
                            menu_group=menu_group,
                            metadata=metadata,
                        )
                        self._stats["documents_created"] += 1

                        logger.debug(
                            "document_created",
                            doc_id=doc.doc_id,
                            item_name=doc.item_name,
                            restaurant=doc.restaurant_name,
                        )

                        yield doc

                    except Exception as e:
                        logger.error(
                            "transform_error",
                            item_name=item.name,
                            restaurant=restaurant.name,
                            error=str(e),
                        )

    def transform_directory(self, dir_path: str | Path) -> list[IndexDocument]:
        """Transform all JSON files in a directory."""
        dir_path = Path(dir_path)
        documents = []

        json_files = list(dir_path.glob("*.json"))
        logger.info("transforming_directory", path=str(dir_path), file_count=len(json_files))

        for file_path in json_files:
            # Skip schema files
            if "schema" in file_path.name.lower():
                continue

            try:
                docs = self.transform_file(file_path)
                documents.extend(docs)
            except Exception as e:
                logger.error("file_transform_error", file=str(file_path), error=str(e))

        logger.info(
            "transformation_complete",
            total_documents=len(documents),
            stats=self._stats,
        )

        return documents

    def reset_stats(self) -> None:
        """Reset transformation statistics."""
        self._stats = {
            "restaurants_processed": 0,
            "menus_processed": 0,
            "groups_processed": 0,
            "items_processed": 0,
            "documents_created": 0,
        }
