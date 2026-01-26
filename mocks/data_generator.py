"""Generate realistic mock data for testing."""

import random
from datetime import datetime, timedelta
from uuid import uuid4

from faker import Faker

fake = Faker()


class MockDataGenerator:
    """Generate realistic mock catering menu data."""

    CUISINES = ["Italian", "Mexican", "American", "Asian", "Mediterranean", "Indian", "Chinese"]
    DIETARY_LABELS = ["vegetarian", "vegan", "gluten-free", "dairy-free", "nut-free", "halal", "kosher"]
    TAGS = ["popular", "new", "seasonal", "chef-special", "bestseller"]
    MENU_GROUPS = ["Appetizers", "Entrees", "Sides", "Desserts", "Beverages", "Boxed Lunches", "Platters"]

    ITEM_TEMPLATES = {
        "Italian": [
            ("Pasta Tray", "Fresh pasta with {sauce} sauce", 89.99, 10, 12),
            ("Lasagna Tray", "Layers of pasta, meat, and cheese", 109.99, 12, 15),
            ("Caesar Salad Bowl", "Crisp romaine with parmesan and croutons", 49.99, 8, 10),
            ("Chicken Parmesan Tray", "Breaded chicken with marinara and mozzarella", 129.99, 10, 12),
            ("Garlic Bread Basket", "Warm garlic bread slices", 24.99, 10, 12),
        ],
        "Mexican": [
            ("Taco Kit", "Build-your-own taco kit with {protein}", 119.99, 10, 12),
            ("Burrito Bowl Bar", "Rice, beans, proteins, and toppings", 139.99, 12, 15),
            ("Chips & Guacamole", "Fresh tortilla chips with house-made guacamole", 39.99, 8, 10),
            ("Quesadilla Platter", "Assorted quesadillas with salsa", 79.99, 8, 10),
            ("Fajita Bar", "Sizzling fajitas with all the fixings", 149.99, 12, 15),
        ],
        "American": [
            ("Sandwich Platter", "Assorted premium sandwiches", 89.99, 10, 12),
            ("BBQ Sampler", "Pulled pork, ribs, and brisket", 159.99, 12, 15),
            ("Burger Bar", "Gourmet burger setup with toppings", 129.99, 10, 12),
            ("Mac & Cheese Tray", "Creamy baked macaroni and cheese", 59.99, 10, 12),
            ("Garden Salad Bowl", "Mixed greens with assorted toppings", 44.99, 8, 10),
        ],
        "Asian": [
            ("Pad Thai Tray", "Classic Thai noodles with {protein}", 99.99, 10, 12),
            ("Fried Rice Tray", "Wok-fried rice with vegetables", 69.99, 10, 12),
            ("Spring Roll Platter", "Crispy vegetable spring rolls", 49.99, 12, 15),
            ("Teriyaki Bowl Bar", "Rice bowls with teriyaki proteins", 119.99, 10, 12),
            ("Sushi Platter", "Assorted sushi rolls", 149.99, 8, 10),
        ],
    }

    SAUCES = ["marinara", "alfredo", "pesto", "vodka"]
    PROTEINS = ["chicken", "beef", "shrimp", "tofu"]

    def __init__(self, seed: int | None = None):
        if seed:
            random.seed(seed)
            Faker.seed(seed)

    def generate_restaurant(self) -> dict:
        """Generate a mock restaurant."""
        cuisine = random.choice(self.CUISINES)
        city = fake.city()
        state = fake.state_abbr()

        return {
            "metadata": {
                "schemaVersion": "0.1.0",
                "sourcePlatform": "mock",
                "sourcePath": f"/catering/{fake.slug()}",
                "engine": "mock-generator",
                "scrapedAt": datetime.utcnow().isoformat() + "Z",
                "normalizedAt": datetime.utcnow().isoformat() + "Z",
                "scrapeRunId": str(uuid4()),
                "contentHash": fake.sha256(),
            },
            "restaurant": {
                "name": f"{fake.last_name()}'s {cuisine} Kitchen",
                "cuisine": [cuisine] + random.sample(self.CUISINES, k=random.randint(0, 2)),
                "location": {
                    "address": fake.street_address(),
                    "city": city,
                    "state": state,
                    "zipCode": fake.zipcode(),
                    "country": "USA",
                    "coordinates": {
                        "latitude": float(fake.latitude()),
                        "longitude": float(fake.longitude()),
                    },
                },
            },
            "menus": self._generate_menus(cuisine),
        }

    def _generate_menus(self, cuisine: str) -> list:
        """Generate menus for a restaurant."""
        return [
            {
                "menuId": "catering",
                "name": "Catering",
                "menuGroups": self._generate_menu_groups(cuisine),
            }
        ]

    def _generate_menu_groups(self, cuisine: str) -> list:
        """Generate menu groups with items."""
        groups = random.sample(self.MENU_GROUPS, k=random.randint(2, 4))
        return [
            {
                "groupId": group.lower().replace(" ", "-"),
                "name": group,
                "menuItems": self._generate_menu_items(cuisine, random.randint(2, 5)),
            }
            for group in groups
        ]

    def _generate_menu_items(self, cuisine: str, count: int) -> list:
        """Generate menu items for a cuisine."""
        templates = self.ITEM_TEMPLATES.get(cuisine, self.ITEM_TEMPLATES["American"])
        items = []

        for template in random.sample(templates, k=min(count, len(templates))):
            name, desc, price, serves_min, serves_max = template

            # Fill in template variables
            desc = desc.format(
                sauce=random.choice(self.SAUCES),
                protein=random.choice(self.PROTEINS),
            )

            # Add some price variation
            price = round(price * random.uniform(0.9, 1.1), 2)

            item = {
                "itemId": str(uuid4())[:8],
                "name": name,
                "description": desc,
                "price": {
                    "basePrice": price,
                    "displayPrice": price,
                    "currency": "USD",
                },
                "servingSize": {
                    "amount": serves_max,
                    "unit": "serves",
                    "description": f"Serves {serves_min}-{serves_max}",
                },
                "minimumOrder": {
                    "quantity": 1,
                    "unit": "tray",
                },
                "dietaryLabels": random.sample(self.DIETARY_LABELS, k=random.randint(0, 2)),
                "tags": random.sample(self.TAGS, k=random.randint(0, 2)),
            }
            items.append(item)

        return items

    def generate_index_document(self) -> dict:
        """Generate a mock index document (flattened)."""
        restaurant = self.generate_restaurant()
        rest = restaurant["restaurant"]
        loc = rest["location"]
        menu = restaurant["menus"][0]
        group = menu["menuGroups"][0]
        item = group["menuItems"][0]

        serves_min = item["servingSize"]["amount"] - 2
        serves_max = item["servingSize"]["amount"]

        return {
            "doc_id": str(uuid4()),
            "restaurant_id": fake.sha256()[:16],
            "item_id": item["itemId"],
            "restaurant_name": rest["name"],
            "cuisine": rest["cuisine"],
            "city": loc["city"],
            "state": loc["state"],
            "zip_code": loc["zipCode"],
            "coordinates": {
                "lat": loc["coordinates"]["latitude"],
                "lon": loc["coordinates"]["longitude"],
            },
            "menu_name": menu["name"],
            "menu_group_name": group["name"],
            "item_name": item["name"],
            "item_description": item["description"],
            "base_price": item["price"]["basePrice"],
            "display_price": item["price"]["displayPrice"],
            "price_per_person": round(item["price"]["displayPrice"] / serves_max, 2),
            "currency": "USD",
            "serves_min": serves_min,
            "serves_max": serves_max,
            "serving_unit": "people",
            "dietary_labels": item.get("dietaryLabels", []),
            "tags": item.get("tags", []),
            "has_portions": False,
            "has_modifiers": False,
            "source_platform": "mock",
            "indexed_at": datetime.utcnow().isoformat() + "Z",
            "rrf_score": random.uniform(0.01, 0.05),
        }

    def generate_search_results(self, count: int = 10) -> list[dict]:
        """Generate mock search results."""
        return [self.generate_index_document() for _ in range(count)]
