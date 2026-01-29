"""Neo4j graph search implementation."""

import asyncio
from typing import Any

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from src.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class GraphSearcher:
    """Graph-based search using Neo4j for relationship queries."""

    def __init__(self, driver: AsyncDriver | None = None):
        """Initialize graph searcher.

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
        logger.info("graph_searcher_connected", uri=settings.neo4j_uri)

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._connected = False
            logger.info("graph_searcher_disconnected")

    @property
    def driver(self) -> AsyncDriver:
        """Get the Neo4j driver, raising if not connected."""
        if not self._driver:
            raise RuntimeError("GraphSearcher not connected. Call connect() first.")
        return self._driver

    async def get_restaurant_items(
        self,
        doc_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get other items from the same restaurant as a given item.

        Use case: "Show me more from this restaurant"

        Args:
            doc_id: The doc_id of a previously returned item
            filters: Optional filters (price_max, serves_min, dietary_labels)
            limit: Maximum number of results

        Returns:
            List of menu items from the same restaurant
        """
        filters = filters or {}

        query = """
        MATCH (i:MenuItem {doc_id: $doc_id})
        MATCH (i)<-[:CONTAINS]-(g:MenuGroup)<-[:HAS_GROUP]-(m:Menu)<-[:HAS_MENU]-(r:Restaurant)

        // Find sibling items in same restaurant
        MATCH (r)-[:HAS_MENU]->(m2:Menu)-[:HAS_GROUP]->(g2:MenuGroup)-[:CONTAINS]->(other:MenuItem)
        WHERE other.doc_id <> $doc_id

        // Apply filters
        AND ($price_max IS NULL OR other.display_price <= $price_max)
        AND ($serves_min IS NULL OR other.serves_max >= $serves_min)
        AND ($dietary_labels IS NULL OR any(label IN $dietary_labels WHERE label IN other.dietary_labels))

        RETURN other.doc_id AS doc_id,
               other.name AS item_name,
               other.description AS item_description,
               other.display_price AS display_price,
               other.serves_min AS serves_min,
               other.serves_max AS serves_max,
               other.dietary_labels AS dietary_labels,
               other.tags AS tags,
               r.restaurant_id AS restaurant_id,
               r.name AS restaurant_name,
               r.city AS city,
               r.state AS state,
               g2.name AS menu_group_name,
               m2.name AS menu_name
        ORDER BY m2.display_order, g2.display_order, other.display_price
        LIMIT $limit
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                {
                    "doc_id": doc_id,
                    "price_max": filters.get("price_max"),
                    "serves_min": filters.get("serves_min"),
                    "dietary_labels": filters.get("dietary_labels"),
                    "limit": limit,
                },
            )
            records = await result.data()

        logger.info(
            "graph_restaurant_items_query",
            doc_id=doc_id,
            result_count=len(records),
        )

        return records

    async def get_similar_restaurants(
        self,
        restaurant_id: str,
        city: str | None = None,
        max_distance_km: float = 10.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find restaurants similar to the specified one.

        Use case: "Similar restaurants nearby"

        Similarity based on:
        - Shared cuisines
        - Geographic proximity
        - Same city (if specified)

        Args:
            restaurant_id: The restaurant to find similar ones to
            city: Optional city filter (if None, uses same city as reference)
            max_distance_km: Maximum distance in kilometers
            limit: Maximum number of results

        Returns:
            List of similar restaurants with sample items
        """
        query = """
        MATCH (r:Restaurant {restaurant_id: $restaurant_id})

        // Find restaurants with shared cuisines
        MATCH (r)-[:SERVES_CUISINE]->(c:Cuisine)<-[:SERVES_CUISINE]-(similar:Restaurant)
        WHERE similar.restaurant_id <> $restaurant_id
          AND ($city IS NULL OR similar.city = $city OR similar.city = r.city)

        // Calculate distance
        WITH similar, r, collect(DISTINCT c.name) AS shared_cuisines,
             point.distance(
                 point({latitude: r.latitude, longitude: r.longitude}),
                 point({latitude: similar.latitude, longitude: similar.longitude})
             ) / 1000 AS distance_km

        WHERE distance_km <= $max_distance_km

        // Get sample items from similar restaurant
        OPTIONAL MATCH (similar)-[:HAS_MENU]->(:Menu)-[:HAS_GROUP]->(:MenuGroup)-[:CONTAINS]->(sample:MenuItem)
        WITH similar, shared_cuisines, distance_km, collect(sample)[0..3] AS sample_items

        RETURN similar.restaurant_id AS restaurant_id,
               similar.name AS restaurant_name,
               similar.cuisine AS cuisines,
               similar.city AS city,
               similar.state AS state,
               shared_cuisines,
               distance_km,
               [item IN sample_items | {
                   doc_id: item.doc_id,
                   name: item.name,
                   price: item.display_price
               }] AS sample_items
        ORDER BY size(shared_cuisines) DESC, distance_km ASC
        LIMIT $limit
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                {
                    "restaurant_id": restaurant_id,
                    "city": city,
                    "max_distance_km": max_distance_km,
                    "limit": limit,
                },
            )
            records = await result.data()

        logger.info(
            "graph_similar_restaurants_query",
            restaurant_id=restaurant_id,
            result_count=len(records),
        )

        return records

    async def get_pairings(
        self,
        doc_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get items that pair well with the specified item.

        Use case: "What pairs well with this?"

        First tries PAIRS_WITH relationships, then falls back to
        items in the same menu group.

        Args:
            doc_id: The doc_id of the item to find pairings for
            limit: Maximum number of results

        Returns:
            List of items that pair well with the specified item
        """
        # Query for direct pairings first
        direct_query = """
        MATCH (i:MenuItem {doc_id: $doc_id})
        MATCH (i)-[p:PAIRS_WITH]-(paired:MenuItem)
        MATCH (paired)<-[:CONTAINS]-(g:MenuGroup)<-[:HAS_GROUP]-(:Menu)<-[:HAS_MENU]-(r:Restaurant)

        RETURN paired.doc_id AS doc_id,
               paired.name AS item_name,
               paired.description AS item_description,
               paired.display_price AS display_price,
               paired.serves_max AS serves_max,
               paired.dietary_labels AS dietary_labels,
               r.name AS restaurant_name,
               r.restaurant_id AS restaurant_id,
               r.city AS city,
               r.state AS state,
               g.name AS menu_group_name,
               p.confidence AS pairing_confidence,
               p.source AS pairing_source
        ORDER BY p.confidence DESC
        LIMIT $limit
        """

        async with self.driver.session() as session:
            result = await session.run(direct_query, {"doc_id": doc_id, "limit": limit})
            direct_pairings = await result.data()

        # If we have enough direct pairings, return them
        if len(direct_pairings) >= limit:
            logger.info(
                "graph_pairings_query",
                doc_id=doc_id,
                result_count=len(direct_pairings),
                source="direct",
            )
            return direct_pairings

        # Otherwise, supplement with same-group items
        remaining = limit - len(direct_pairings)
        existing_ids = [p["doc_id"] for p in direct_pairings] + [doc_id]

        fallback_query = """
        MATCH (i:MenuItem {doc_id: $doc_id})
        MATCH (i)<-[:CONTAINS]-(g:MenuGroup)-[:CONTAINS]->(sibling:MenuItem)
        WHERE NOT sibling.doc_id IN $existing_ids

        MATCH (sibling)<-[:CONTAINS]-(g)<-[:HAS_GROUP]-(:Menu)<-[:HAS_MENU]-(r:Restaurant)

        RETURN sibling.doc_id AS doc_id,
               sibling.name AS item_name,
               sibling.description AS item_description,
               sibling.display_price AS display_price,
               sibling.serves_max AS serves_max,
               sibling.dietary_labels AS dietary_labels,
               r.name AS restaurant_name,
               r.restaurant_id AS restaurant_id,
               r.city AS city,
               r.state AS state,
               g.name AS menu_group_name,
               0.5 AS pairing_confidence,
               'same_group' AS pairing_source
        LIMIT $remaining
        """

        async with self.driver.session() as session:
            result = await session.run(
                fallback_query,
                {"doc_id": doc_id, "existing_ids": existing_ids, "remaining": remaining},
            )
            fallback_pairings = await result.data()

        all_pairings = direct_pairings + fallback_pairings

        logger.info(
            "graph_pairings_query",
            doc_id=doc_id,
            result_count=len(all_pairings),
            direct_count=len(direct_pairings),
            fallback_count=len(fallback_pairings),
        )

        return all_pairings

    async def get_multi_cuisine_restaurants(
        self,
        cuisines: list[str],
        city: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find restaurants that serve all specified cuisines.

        Use case: "Restaurants with both Italian AND Mexican"

        Args:
            cuisines: List of required cuisines
            city: Optional city filter
            limit: Maximum number of results

        Returns:
            List of restaurants serving all specified cuisines
        """
        if not cuisines:
            return []

        query = """
        MATCH (r:Restaurant)
        WHERE all(cuisine IN $required_cuisines WHERE cuisine IN r.cuisine)
          AND ($city IS NULL OR r.city = $city)

        // Get sample items
        OPTIONAL MATCH (r)-[:HAS_MENU]->(m:Menu)-[:HAS_GROUP]->(g:MenuGroup)-[:CONTAINS]->(item:MenuItem)
        WITH r, collect(item)[0..5] AS sample_items

        RETURN r.restaurant_id AS restaurant_id,
               r.name AS restaurant_name,
               r.cuisine AS cuisines,
               r.city AS city,
               r.state AS state,
               [item IN sample_items | {
                   doc_id: item.doc_id,
                   name: item.name,
                   price: item.display_price,
                   menu_group: item.group_id
               }] AS sample_items
        ORDER BY size(r.cuisine) DESC
        LIMIT $limit
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                {
                    "required_cuisines": cuisines,
                    "city": city,
                    "limit": limit,
                },
            )
            records = await result.data()

        logger.info(
            "graph_multi_cuisine_query",
            cuisines=cuisines,
            city=city,
            result_count=len(records),
        )

        return records

    async def get_catering_packages(
        self,
        party_size: int,
        budget: float | None = None,
        city: str | None = None,
        cuisines: list[str] | None = None,
        dietary_requirements: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest complete catering packages.

        Use case: "Complete catering package for 50 people"

        Args:
            party_size: Number of people to serve
            budget: Optional maximum total budget
            city: Optional city filter
            cuisines: Optional cuisine preferences
            dietary_requirements: Optional dietary requirements
            limit: Maximum number of packages to return

        Returns:
            List of complete package suggestions
        """
        query = """
        MATCH (r:Restaurant)-[:HAS_MENU]->(m:Menu)
        WHERE ($city IS NULL OR r.city = $city)
          AND ($cuisines IS NULL OR any(c IN $cuisines WHERE c IN r.cuisine))

        // Find appetizers that serve the party
        OPTIONAL MATCH (m)-[:HAS_GROUP]->(appetizers:MenuGroup)
        WHERE toLower(appetizers.name) CONTAINS 'appetizer'
           OR toLower(appetizers.name) CONTAINS 'starter'
        OPTIONAL MATCH (appetizers)-[:CONTAINS]->(app:MenuItem)
        WHERE app.serves_max >= $party_size
          AND ($dietary IS NULL OR all(d IN $dietary WHERE d IN app.dietary_labels))

        // Find entrees
        OPTIONAL MATCH (m)-[:HAS_GROUP]->(entrees:MenuGroup)
        WHERE toLower(entrees.name) CONTAINS 'entree'
           OR toLower(entrees.name) CONTAINS 'main'
        OPTIONAL MATCH (entrees)-[:CONTAINS]->(main:MenuItem)
        WHERE main.serves_max >= $party_size
          AND ($dietary IS NULL OR all(d IN $dietary WHERE d IN main.dietary_labels))

        // Find desserts
        OPTIONAL MATCH (m)-[:HAS_GROUP]->(desserts:MenuGroup)
        WHERE toLower(desserts.name) CONTAINS 'dessert'
           OR toLower(desserts.name) CONTAINS 'sweet'
        OPTIONAL MATCH (desserts)-[:CONTAINS]->(dessert:MenuItem)
        WHERE dessert.serves_max >= $party_size
          AND ($dietary IS NULL OR all(d IN $dietary WHERE d IN dessert.dietary_labels))

        // Calculate package total
        WITH r, m, app, main, dessert,
             coalesce(app.display_price, 0) +
             coalesce(main.display_price, 0) +
             coalesce(dessert.display_price, 0) AS package_total

        WHERE app IS NOT NULL AND main IS NOT NULL
          AND ($budget IS NULL OR package_total <= $budget)

        RETURN r.restaurant_id AS restaurant_id,
               r.name AS restaurant_name,
               r.city AS city,
               r.state AS state,
               r.cuisine AS cuisines,
               {
                   appetizer: CASE WHEN app IS NOT NULL THEN {
                       doc_id: app.doc_id,
                       name: app.name,
                       price: app.display_price,
                       serves: app.serves_max,
                       dietary_labels: app.dietary_labels
                   } ELSE null END,
                   entree: CASE WHEN main IS NOT NULL THEN {
                       doc_id: main.doc_id,
                       name: main.name,
                       price: main.display_price,
                       serves: main.serves_max,
                       dietary_labels: main.dietary_labels
                   } ELSE null END,
                   dessert: CASE WHEN dessert IS NOT NULL THEN {
                       doc_id: dessert.doc_id,
                       name: dessert.name,
                       price: dessert.display_price,
                       serves: dessert.serves_max,
                       dietary_labels: dessert.dietary_labels
                   } ELSE null END
               } AS package,
               package_total AS total_price,
               package_total / $party_size AS price_per_person
        ORDER BY package_total ASC
        LIMIT $limit
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                {
                    "party_size": party_size,
                    "budget": budget,
                    "city": city,
                    "cuisines": cuisines,
                    "dietary": dietary_requirements,
                    "limit": limit,
                },
            )
            records = await result.data()

        logger.info(
            "graph_catering_packages_query",
            party_size=party_size,
            budget=budget,
            result_count=len(records),
        )

        return records

    async def get_menu_structure(
        self,
        restaurant_id: str,
        include_items: bool = True,
    ) -> dict[str, Any] | None:
        """Get complete menu structure for a restaurant.

        Args:
            restaurant_id: The restaurant ID
            include_items: Whether to include individual menu items

        Returns:
            Hierarchical menu structure or None if not found
        """
        query = """
        MATCH (r:Restaurant {restaurant_id: $restaurant_id})

        OPTIONAL MATCH (r)-[hm:HAS_MENU]->(m:Menu)
        OPTIONAL MATCH (m)-[hg:HAS_GROUP]->(g:MenuGroup)
        """

        if include_items:
            query += """
        OPTIONAL MATCH (g)-[c:CONTAINS]->(i:MenuItem)

        WITH r, m, g, collect({
            doc_id: i.doc_id,
            name: i.name,
            description: i.description,
            price: i.display_price,
            serves_max: i.serves_max,
            dietary_labels: i.dietary_labels,
            display_order: c.display_order
        }) AS items
        ORDER BY c.display_order
            """
        else:
            query += """
        WITH r, m, g, [] AS items
            """

        query += """
        WITH r, m, collect({
            group_id: g.group_id,
            name: g.name,
            description: g.description,
            display_order: g.display_order,
            items: items
        }) AS groups
        ORDER BY g.display_order

        WITH r, collect({
            menu_id: m.menu_id,
            name: m.name,
            description: m.description,
            display_order: m.display_order,
            groups: groups
        }) AS menus
        ORDER BY m.display_order

        RETURN r.restaurant_id AS restaurant_id,
               r.name AS restaurant_name,
               r.cuisine AS cuisines,
               r.city AS city,
               r.state AS state,
               menus
        """

        async with self.driver.session() as session:
            result = await session.run(query, {"restaurant_id": restaurant_id})
            record = await result.single()

        if not record:
            return None

        return dict(record)

    async def get_related_items(
        self,
        doc_id: str,
        relation_type: str = "all",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get items related to a specific menu item.

        Args:
            doc_id: The doc_id of the item
            relation_type: Type of relation:
                - same_restaurant: Other items from same restaurant
                - same_group: Items in same menu group
                - pairing: Items that pair well
                - all: Combination of above
            limit: Maximum number of results

        Returns:
            List of related items
        """
        if relation_type == "same_restaurant":
            return await self.get_restaurant_items(doc_id, limit=limit)
        elif relation_type == "pairing":
            return await self.get_pairings(doc_id, limit=limit)
        elif relation_type == "same_group":
            return await self._get_same_group_items(doc_id, limit)
        elif relation_type == "all":
            # Combine different relation types
            tasks = [
                self.get_pairings(doc_id, limit=limit // 2),
                self._get_same_group_items(doc_id, limit=limit // 2),
            ]
            results = await asyncio.gather(*tasks)
            combined = results[0] + results[1]
            # Deduplicate by doc_id
            seen = set()
            unique = []
            for item in combined:
                if item["doc_id"] not in seen:
                    seen.add(item["doc_id"])
                    unique.append(item)
            return unique[:limit]
        else:
            logger.warning("unknown_relation_type", relation_type=relation_type)
            return []

    async def _get_same_group_items(
        self,
        doc_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get items in the same menu group as the specified item."""
        query = """
        MATCH (i:MenuItem {doc_id: $doc_id})
        MATCH (i)<-[:CONTAINS]-(g:MenuGroup)-[:CONTAINS]->(sibling:MenuItem)
        WHERE sibling.doc_id <> $doc_id

        MATCH (g)<-[:HAS_GROUP]-(:Menu)<-[:HAS_MENU]-(r:Restaurant)

        RETURN sibling.doc_id AS doc_id,
               sibling.name AS item_name,
               sibling.description AS item_description,
               sibling.display_price AS display_price,
               sibling.serves_max AS serves_max,
               sibling.dietary_labels AS dietary_labels,
               r.name AS restaurant_name,
               r.restaurant_id AS restaurant_id,
               r.city AS city,
               r.state AS state,
               g.name AS menu_group_name,
               'same_group' AS relation_type
        LIMIT $limit
        """

        async with self.driver.session() as session:
            result = await session.run(query, {"doc_id": doc_id, "limit": limit})
            records = await result.data()

        return records

    async def search_by_doc_ids(
        self,
        doc_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch menu items by their doc_ids.

        Useful for enriching results from other search systems.

        Args:
            doc_ids: List of doc_ids to fetch

        Returns:
            List of menu items with full details
        """
        if not doc_ids:
            return []

        query = """
        MATCH (i:MenuItem)
        WHERE i.doc_id IN $doc_ids

        MATCH (i)<-[:CONTAINS]-(g:MenuGroup)<-[:HAS_GROUP]-(m:Menu)<-[:HAS_MENU]-(r:Restaurant)

        RETURN i.doc_id AS doc_id,
               i.name AS item_name,
               i.description AS item_description,
               i.display_price AS display_price,
               i.base_price AS base_price,
               i.serves_min AS serves_min,
               i.serves_max AS serves_max,
               i.dietary_labels AS dietary_labels,
               i.tags AS tags,
               r.restaurant_id AS restaurant_id,
               r.name AS restaurant_name,
               r.cuisine AS cuisine,
               r.city AS city,
               r.state AS state,
               g.name AS menu_group_name,
               m.name AS menu_name
        """

        async with self.driver.session() as session:
            result = await session.run(query, {"doc_ids": doc_ids})
            records = await result.data()

        return records


# Singleton instance management
_graph_searcher: GraphSearcher | None = None
_graph_searcher_lock = asyncio.Lock()


async def get_graph_searcher() -> GraphSearcher:
    """Get or create GraphSearcher singleton."""
    global _graph_searcher

    if _graph_searcher is not None:
        return _graph_searcher

    async with _graph_searcher_lock:
        if _graph_searcher is None:
            _graph_searcher = GraphSearcher()
            await _graph_searcher.connect()

    return _graph_searcher
