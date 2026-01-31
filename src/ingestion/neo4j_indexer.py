"""Neo4j graph database indexer for relationship data."""

from typing import Any, Sequence

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from src.config import get_settings
from src.models.source import RestaurantData

logger = structlog.get_logger()
settings = get_settings()


# Cypher statements for schema creation
NEO4J_SCHEMA = """
// Constraints for unique IDs
CREATE CONSTRAINT restaurant_id IF NOT EXISTS FOR (r:Restaurant) REQUIRE r.restaurant_id IS UNIQUE;
CREATE CONSTRAINT menu_id IF NOT EXISTS FOR (m:Menu) REQUIRE m.menu_id IS UNIQUE;
CREATE CONSTRAINT menu_group_id IF NOT EXISTS FOR (g:MenuGroup) REQUIRE g.group_id IS UNIQUE;
CREATE CONSTRAINT menu_item_id IF NOT EXISTS FOR (i:MenuItem) REQUIRE i.doc_id IS UNIQUE;
CREATE CONSTRAINT cuisine_name IF NOT EXISTS FOR (c:Cuisine) REQUIRE c.name IS UNIQUE;

// Indexes for common queries
CREATE INDEX restaurant_city IF NOT EXISTS FOR (r:Restaurant) ON (r.city);
CREATE INDEX restaurant_state IF NOT EXISTS FOR (r:Restaurant) ON (r.state);
CREATE INDEX menu_item_price IF NOT EXISTS FOR (i:MenuItem) ON (i.display_price);
CREATE INDEX menu_item_serves IF NOT EXISTS FOR (i:MenuItem) ON (i.serves_max);
"""


class Neo4jIndexer:
    """Index restaurant data to Neo4j for graph-based queries."""

    def __init__(self, driver: AsyncDriver | None = None):
        """Initialize Neo4j indexer.

        Args:
            driver: Optional pre-configured Neo4j driver (for testing)
        """
        self._driver = driver
        self._connected = driver is not None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._connected:
            return

        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=settings.neo4j_max_connection_pool_size,
        )

        # Verify connectivity
        async with self._driver.session() as session:
            await session.run("RETURN 1")

        self._connected = True
        logger.info("neo4j_indexer_connected", uri=settings.neo4j_uri)

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._connected = False
            logger.info("neo4j_indexer_disconnected")

    @property
    def driver(self) -> AsyncDriver:
        """Get the Neo4j driver, raising if not connected."""
        if not self._driver:
            raise RuntimeError("Neo4jIndexer not connected. Call connect() first.")
        return self._driver

    async def create_schema(self) -> None:
        """Create Neo4j constraints and indexes."""
        if not self._connected:
            await self.connect()

        async with self.driver.session() as session:
            # Execute each constraint/index separately (Neo4j requirement)
            for statement in NEO4J_SCHEMA.strip().split(";"):
                statement = statement.strip()
                if statement and not statement.startswith("//"):
                    try:
                        await session.run(statement)
                    except Exception as e:
                        # Constraints may already exist
                        if "already exists" not in str(e).lower():
                            logger.warning("schema_statement_error", statement=statement[:50], error=str(e))

        logger.info("neo4j_schema_created")

    async def clear_all(self) -> None:
        """Delete all nodes and relationships."""
        async with self.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        logger.info("neo4j_cleared")

    async def index_restaurant(self, data: RestaurantData) -> dict[str, Any]:
        """Index a single restaurant with all its menus, groups, and items.

        Args:
            data: RestaurantData from source model

        Returns:
            Stats dict with counts
        """
        stats = {"restaurants": 0, "menus": 0, "groups": 0, "items": 0, "cuisines": 0}

        async with self.driver.session() as session:
            # Create Restaurant node
            restaurant_query = """
            MERGE (r:Restaurant {restaurant_id: $restaurant_id})
            SET r.name = $name,
                r.cuisine = $cuisine,
                r.city = $city,
                r.state = $state,
                r.zip_code = $zip_code,
                r.latitude = $latitude,
                r.longitude = $longitude,
                r.source_platform = $source_platform
            RETURN r
            """

            location = data.restaurant.location
            await session.run(
                restaurant_query,
                {
                    "restaurant_id": data.restaurant.restaurantId,
                    "name": data.restaurant.name,
                    "cuisine": data.restaurant.cuisine or [],
                    "city": location.city if location else None,
                    "state": location.state if location else None,
                    "zip_code": location.zipCode if location else None,
                    "latitude": location.coordinates.latitude if location and location.coordinates else None,
                    "longitude": location.coordinates.longitude if location and location.coordinates else None,
                    "source_platform": data.metadata.sourcePlatform if data.metadata else None,
                },
            )
            stats["restaurants"] = 1

            # Create Cuisine nodes and relationships
            for cuisine_name in (data.restaurant.cuisine or []):
                cuisine_query = """
                MERGE (c:Cuisine {name: $cuisine_name})
                WITH c
                MATCH (r:Restaurant {restaurant_id: $restaurant_id})
                MERGE (r)-[:SERVES_CUISINE]->(c)
                """
                await session.run(
                    cuisine_query,
                    {"cuisine_name": cuisine_name, "restaurant_id": data.restaurant.restaurantId},
                )
                stats["cuisines"] += 1

            # Process menus
            for menu_idx, menu in enumerate(data.restaurant.menus or []):
                menu_id = f"{data.restaurant.restaurantId}_{menu.menuId or menu_idx}"

                menu_query = """
                MERGE (m:Menu {menu_id: $menu_id})
                SET m.name = $name,
                    m.description = $description,
                    m.display_order = $display_order
                WITH m
                MATCH (r:Restaurant {restaurant_id: $restaurant_id})
                MERGE (r)-[:HAS_MENU]->(m)
                """
                await session.run(
                    menu_query,
                    {
                        "menu_id": menu_id,
                        "name": menu.name,
                        "description": menu.description,
                        "display_order": menu_idx,
                        "restaurant_id": data.restaurant.restaurantId,
                    },
                )
                stats["menus"] += 1

                # Process menu groups
                for group_idx, group in enumerate(menu.groups or []):
                    group_id = f"{menu_id}_{group.groupId or group_idx}"

                    group_query = """
                    MERGE (g:MenuGroup {group_id: $group_id})
                    SET g.name = $name,
                        g.description = $description,
                        g.display_order = $display_order
                    WITH g
                    MATCH (m:Menu {menu_id: $menu_id})
                    MERGE (m)-[:HAS_GROUP]->(g)
                    """
                    await session.run(
                        group_query,
                        {
                            "group_id": group_id,
                            "name": group.name,
                            "description": group.description,
                            "display_order": group_idx,
                            "menu_id": menu_id,
                        },
                    )
                    stats["groups"] += 1

                    # Process menu items
                    for item_idx, item in enumerate(group.items or []):
                        doc_id = f"{data.restaurant.restaurantId}_{item.itemId or f'{group_idx}_{item_idx}'}"

                        # Get price info
                        base_price = None
                        display_price = None
                        if item.price:
                            base_price = item.price.basePrice
                            display_price = item.price.displayPrice or item.price.basePrice

                        # Get serving info
                        serves_min = None
                        serves_max = None
                        if item.serving:
                            serves_min = item.serving.min
                            serves_max = item.serving.max

                        item_query = """
                        MERGE (i:MenuItem {doc_id: $doc_id})
                        SET i.name = $name,
                            i.description = $description,
                            i.base_price = $base_price,
                            i.display_price = $display_price,
                            i.serves_min = $serves_min,
                            i.serves_max = $serves_max,
                            i.dietary_labels = $dietary_labels,
                            i.tags = $tags,
                            i.display_order = $display_order
                        WITH i
                        MATCH (g:MenuGroup {group_id: $group_id})
                        MERGE (g)-[:CONTAINS {display_order: $display_order}]->(i)
                        """
                        await session.run(
                            item_query,
                            {
                                "doc_id": doc_id,
                                "name": item.name,
                                "description": item.description,
                                "base_price": base_price,
                                "display_price": display_price,
                                "serves_min": serves_min,
                                "serves_max": serves_max,
                                "dietary_labels": item.dietaryLabels or [],
                                "tags": item.tags or [],
                                "display_order": item_idx,
                                "group_id": group_id,
                            },
                        )
                        stats["items"] += 1

        return stats

    async def index_restaurants(
        self,
        restaurants: Sequence[RestaurantData],
    ) -> dict[str, Any]:
        """Index multiple restaurants.

        Args:
            restaurants: Sequence of RestaurantData objects

        Returns:
            Aggregate stats dict
        """
        if not restaurants:
            logger.warning("no_restaurants_to_index_neo4j")
            return {"restaurants": 0, "menus": 0, "groups": 0, "items": 0, "cuisines": 0}

        if not self._connected:
            await self.connect()

        total_stats = {"restaurants": 0, "menus": 0, "groups": 0, "items": 0, "cuisines": 0}

        for i, data in enumerate(restaurants):
            try:
                stats = await self.index_restaurant(data)
                for key in total_stats:
                    total_stats[key] += stats[key]

                if (i + 1) % 10 == 0:
                    logger.info("neo4j_indexing_progress", processed=i + 1, total=len(restaurants))

            except Exception as e:
                logger.error(
                    "neo4j_index_restaurant_error",
                    restaurant_id=data.restaurant.restaurantId,
                    error=str(e),
                )

        logger.info("neo4j_indexing_complete", **total_stats)
        return total_stats

    async def create_pairing_relationships(self) -> int:
        """Create PAIRS_WITH relationships between items in same menu group.

        Items in the same menu group are assumed to pair well together.

        Returns:
            Number of relationships created
        """
        query = """
        MATCH (g:MenuGroup)-[:CONTAINS]->(i1:MenuItem)
        MATCH (g)-[:CONTAINS]->(i2:MenuItem)
        WHERE i1.doc_id < i2.doc_id
        MERGE (i1)-[p:PAIRS_WITH]->(i2)
        SET p.confidence = 0.7,
            p.source = 'same_group'
        RETURN count(p) as count
        """

        async with self.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            count = record["count"] if record else 0

        logger.info("pairing_relationships_created", count=count)
        return count

    async def create_similar_restaurant_relationships(self, max_distance_km: float = 10.0) -> int:
        """Create SIMILAR_TO relationships between restaurants.

        Restaurants are similar if they:
        - Share at least one cuisine
        - Are within max_distance_km of each other

        Args:
            max_distance_km: Maximum distance in kilometers

        Returns:
            Number of relationships created
        """
        query = """
        MATCH (r1:Restaurant)-[:SERVES_CUISINE]->(c:Cuisine)<-[:SERVES_CUISINE]-(r2:Restaurant)
        WHERE r1.restaurant_id < r2.restaurant_id
          AND r1.latitude IS NOT NULL AND r1.longitude IS NOT NULL
          AND r2.latitude IS NOT NULL AND r2.longitude IS NOT NULL
        WITH r1, r2, collect(DISTINCT c.name) as shared_cuisines,
             point.distance(
                 point({latitude: r1.latitude, longitude: r1.longitude}),
                 point({latitude: r2.latitude, longitude: r2.longitude})
             ) / 1000 as distance_km
        WHERE distance_km <= $max_distance_km
        MERGE (r1)-[s:SIMILAR_TO]->(r2)
        SET s.shared_cuisines = shared_cuisines,
            s.distance_km = distance_km,
            s.similarity_score = size(shared_cuisines) / (1 + distance_km)
        RETURN count(s) as count
        """

        async with self.driver.session() as session:
            result = await session.run(query, {"max_distance_km": max_distance_km})
            record = await result.single()
            count = record["count"] if record else 0

        logger.info("similar_restaurant_relationships_created", count=count, max_distance_km=max_distance_km)
        return count

    async def get_stats(self) -> dict[str, int]:
        """Get counts of all node types."""
        query = """
        MATCH (r:Restaurant) WITH count(r) as restaurants
        MATCH (m:Menu) WITH restaurants, count(m) as menus
        MATCH (g:MenuGroup) WITH restaurants, menus, count(g) as groups
        MATCH (i:MenuItem) WITH restaurants, menus, groups, count(i) as items
        MATCH (c:Cuisine) WITH restaurants, menus, groups, items, count(c) as cuisines
        RETURN restaurants, menus, groups, items, cuisines
        """

        async with self.driver.session() as session:
            result = await session.run(query)
            record = await result.single()

        if record:
            return dict(record)
        return {"restaurants": 0, "menus": 0, "groups": 0, "items": 0, "cuisines": 0}


# Singleton management
_neo4j_indexer: Neo4jIndexer | None = None


async def get_neo4j_indexer() -> Neo4jIndexer:
    """Get or create Neo4jIndexer singleton."""
    global _neo4j_indexer

    if _neo4j_indexer is None:
        _neo4j_indexer = Neo4jIndexer()
        await _neo4j_indexer.connect()

    return _neo4j_indexer
