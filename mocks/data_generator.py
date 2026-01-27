"""Generate realistic mock data for testing."""

import random
from datetime import datetime, timezone
from uuid import uuid4

from faker import Faker

fake = Faker()


class MockDataGenerator:
    """Generate realistic mock catering menu data."""

    CUISINES = [
        "Italian", "Mexican", "American", "Asian", "Mediterranean", "Indian", "Chinese",
        "Japanese", "Thai", "Greek", "Korean", "Vietnamese", "BBQ", "Southern",
        "Cajun", "French", "Middle Eastern", "Caribbean", "Hawaiian", "Tex-Mex",
    ]

    DIETARY_LABELS = [
        "vegetarian", "vegan", "gluten-free", "dairy-free", "nut-free",
        "halal", "kosher", "organic", "keto", "paleo", "low-carb",
    ]

    TAGS = [
        "popular", "new", "seasonal", "chef-special", "bestseller",
        "staff-pick", "award-winning", "locally-sourced", "family-style",
        "individually-packaged", "hot", "cold", "make-your-own",
    ]

    MENU_GROUPS = [
        "Appetizers", "Entrees", "Sides", "Desserts", "Beverages",
        "Boxed Lunches", "Platters", "Breakfast", "Brunch",
        "Hot Buffet", "Cold Buffet", "Build Your Own",
        "Family Meals", "Individual Meals", "Party Packages",
        "Salads & Bowls", "Sandwiches & Wraps", "Soups",
    ]

    RESTAURANT_NAME_PATTERNS = [
        "{last_name}'s {cuisine} Kitchen",
        "{last_name}'s {cuisine} Grill",
        "The {cuisine} Table",
        "{city} {cuisine} Catering",
        "{adjective} {cuisine} Cafe",
        "{last_name} & Sons {cuisine}",
        "Little {city} {cuisine}",
        "{cuisine} Express Catering",
        "Urban {cuisine} Co.",
        "Fresh {cuisine} Market",
    ]

    ADJECTIVES = [
        "Golden", "Silver", "Royal", "Grand", "Prime", "Classic",
        "Modern", "Rustic", "Coastal", "Downtown", "Village",
    ]

    ITEM_TEMPLATES = {
        "Italian": [
            ("Pasta Tray", "Fresh pasta with {sauce} sauce", 89.99, 10, 12),
            ("Lasagna Tray", "Layers of pasta, meat, and cheese", 109.99, 12, 15),
            ("Caesar Salad Bowl", "Crisp romaine with parmesan and croutons", 49.99, 8, 10),
            ("Chicken Parmesan Tray", "Breaded chicken with marinara and mozzarella", 129.99, 10, 12),
            ("Garlic Bread Basket", "Warm garlic bread slices", 24.99, 10, 12),
            ("Eggplant Parmesan Tray", "Breaded eggplant with marinara and fresh mozzarella", 99.99, 10, 12),
            ("Italian Meatball Tray", "House-made meatballs in San Marzano tomato sauce", 89.99, 10, 12),
            ("Antipasto Platter", "Cured meats, cheeses, olives, and marinated vegetables", 79.99, 12, 15),
            ("Caprese Salad Tray", "Fresh mozzarella, tomatoes, and basil with balsamic glaze", 59.99, 8, 10),
            ("Tiramisu Tray", "Classic Italian dessert with espresso and mascarpone", 69.99, 12, 15),
        ],
        "Mexican": [
            ("Taco Kit", "Build-your-own taco kit with {protein}", 119.99, 10, 12),
            ("Burrito Bowl Bar", "Rice, beans, proteins, and toppings", 139.99, 12, 15),
            ("Chips & Guacamole", "Fresh tortilla chips with house-made guacamole", 39.99, 8, 10),
            ("Quesadilla Platter", "Assorted quesadillas with salsa and sour cream", 79.99, 8, 10),
            ("Fajita Bar", "Sizzling fajitas with all the fixings", 149.99, 12, 15),
            ("Enchilada Tray", "Rolled tortillas with {protein} and red sauce", 99.99, 10, 12),
            ("Street Corn Tray", "Grilled corn with cotija cheese and lime crema", 44.99, 10, 12),
            ("Carnitas Platter", "Slow-roasted pulled pork with cilantro and onions", 129.99, 12, 15),
            ("Tamale Dozen", "Hand-made tamales with choice of filling", 59.99, 6, 8),
            ("Churros Platter", "Cinnamon sugar churros with chocolate dipping sauce", 34.99, 10, 12),
        ],
        "American": [
            ("Sandwich Platter", "Assorted premium sandwiches", 89.99, 10, 12),
            ("BBQ Sampler", "Pulled pork, ribs, and brisket", 159.99, 12, 15),
            ("Burger Bar", "Gourmet burger setup with toppings", 129.99, 10, 12),
            ("Mac & Cheese Tray", "Creamy baked macaroni and cheese", 59.99, 10, 12),
            ("Garden Salad Bowl", "Mixed greens with assorted toppings", 44.99, 8, 10),
            ("Chicken Tender Platter", "Hand-breaded chicken tenders with dipping sauces", 79.99, 10, 12),
            ("Slider Platter", "Mini burgers with assorted toppings", 89.99, 12, 15),
            ("Hot Dog Bar", "All-beef hot dogs with classic toppings", 69.99, 10, 12),
            ("Loaded Potato Bar", "Baked potatoes with all the fixings", 59.99, 10, 12),
            ("Buffalo Wings Platter", "Crispy wings with ranch and blue cheese", 89.99, 12, 15),
        ],
        "Asian": [
            ("Pad Thai Tray", "Classic Thai noodles with {protein}", 99.99, 10, 12),
            ("Fried Rice Tray", "Wok-fried rice with vegetables", 69.99, 10, 12),
            ("Spring Roll Platter", "Crispy vegetable spring rolls", 49.99, 12, 15),
            ("Teriyaki Bowl Bar", "Rice bowls with teriyaki proteins", 119.99, 10, 12),
            ("Sushi Platter", "Assorted sushi rolls", 149.99, 8, 10),
            ("Orange Chicken Tray", "Crispy chicken in tangy orange glaze", 89.99, 10, 12),
            ("Beef & Broccoli Tray", "Tender beef with fresh broccoli in garlic sauce", 99.99, 10, 12),
            ("Vegetable Lo Mein Tray", "Stir-fried noodles with mixed vegetables", 69.99, 10, 12),
            ("Dumpling Platter", "Steamed or pan-fried dumplings with dipping sauce", 59.99, 10, 12),
            ("General Tso's Chicken Tray", "Crispy chicken in sweet and spicy sauce", 89.99, 10, 12),
        ],
        "Mediterranean": [
            ("Falafel Platter", "Crispy chickpea falafel with tahini sauce", 69.99, 10, 12),
            ("Shawarma Tray", "Seasoned {protein} shawarma with pickled vegetables", 119.99, 10, 12),
            ("Hummus & Pita Spread", "House-made hummus with warm pita bread", 44.99, 10, 12),
            ("Greek Salad Bowl", "Cucumber, tomato, olives, and feta cheese", 54.99, 10, 12),
            ("Lamb Kofta Platter", "Grilled lamb skewers with tzatziki", 139.99, 10, 12),
            ("Grilled Chicken Kebab Tray", "Marinated chicken skewers with vegetables", 109.99, 10, 12),
            ("Baba Ganoush Platter", "Smoky eggplant dip with olive oil", 49.99, 10, 12),
            ("Spanakopita Tray", "Flaky phyllo pastry with spinach and feta", 59.99, 12, 15),
            ("Mediterranean Mezze Platter", "Assorted dips, olives, and grilled vegetables", 89.99, 12, 15),
            ("Baklava Tray", "Sweet phyllo pastry with honey and pistachios", 49.99, 15, 20),
        ],
        "Indian": [
            ("Butter Chicken Tray", "Tender chicken in creamy tomato curry", 109.99, 10, 12),
            ("Vegetable Biryani Tray", "Fragrant basmati rice with mixed vegetables", 79.99, 10, 12),
            ("Samosa Platter", "Crispy pastries filled with spiced potatoes", 49.99, 12, 15),
            ("Chicken Tikka Masala Tray", "Grilled chicken in spiced masala sauce", 119.99, 10, 12),
            ("Naan Bread Basket", "Fresh baked naan with garlic butter", 29.99, 10, 12),
            ("Dal Makhani Tray", "Creamy black lentils simmered overnight", 69.99, 10, 12),
            ("Palak Paneer Tray", "Fresh spinach with house-made cheese cubes", 89.99, 10, 12),
            ("Tandoori Chicken Platter", "Clay oven roasted chicken with spices", 99.99, 10, 12),
            ("Chana Masala Tray", "Chickpeas in aromatic tomato sauce", 69.99, 10, 12),
            ("Mango Lassi Pitcher", "Sweet yogurt drink with mango", 24.99, 8, 10),
        ],
        "Japanese": [
            ("Sushi Party Platter", "Chef's selection of nigiri and maki rolls", 159.99, 10, 12),
            ("Teriyaki Chicken Tray", "Grilled chicken with teriyaki glaze", 99.99, 10, 12),
            ("Tempura Platter", "Lightly battered vegetables and shrimp", 89.99, 10, 12),
            ("Ramen Bowl Bar", "Build your own ramen with toppings", 129.99, 10, 12),
            ("Edamame Tray", "Steamed soybeans with sea salt", 29.99, 10, 12),
            ("Gyoza Platter", "Pan-fried Japanese dumplings", 54.99, 10, 12),
            ("Katsu Curry Tray", "Crispy breaded cutlet with Japanese curry", 109.99, 10, 12),
            ("Poke Bowl Bar", "Fresh fish bowls with rice and toppings", 139.99, 10, 12),
            ("Miso Soup Station", "Traditional miso with tofu and seaweed", 39.99, 12, 15),
            ("Matcha Dessert Platter", "Assorted matcha sweets and mochi", 49.99, 12, 15),
        ],
        "BBQ": [
            ("Brisket Platter", "Slow-smoked beef brisket sliced to order", 169.99, 10, 12),
            ("Pulled Pork Tray", "12-hour smoked pulled pork", 119.99, 12, 15),
            ("Baby Back Ribs Rack", "Fall-off-the-bone tender ribs", 149.99, 8, 10),
            ("BBQ Chicken Tray", "Smoked chicken quarters with house sauce", 99.99, 10, 12),
            ("Smoked Sausage Platter", "House-made smoked sausage links", 89.99, 10, 12),
            ("Burnt Ends Tray", "Caramelized brisket point pieces", 139.99, 8, 10),
            ("Coleslaw Tray", "Creamy house-made coleslaw", 34.99, 12, 15),
            ("Baked Beans Tray", "Slow-cooked beans with smoked meat", 39.99, 12, 15),
            ("Cornbread Basket", "Fresh baked cornbread with honey butter", 29.99, 12, 15),
            ("Mac & Cheese Tray", "Smoked gouda mac and cheese", 54.99, 12, 15),
        ],
        "Southern": [
            ("Fried Chicken Platter", "Buttermilk fried chicken pieces", 109.99, 10, 12),
            ("Shrimp & Grits Tray", "Creamy grits with Cajun shrimp", 129.99, 10, 12),
            ("Collard Greens Tray", "Slow-braised greens with smoked ham hock", 44.99, 12, 15),
            ("Biscuit Basket", "Flaky buttermilk biscuits with honey", 34.99, 12, 15),
            ("Chicken & Waffles Platter", "Crispy chicken with Belgian waffles", 119.99, 10, 12),
            ("Catfish Platter", "Cornmeal-crusted fried catfish", 99.99, 10, 12),
            ("Mashed Potatoes Tray", "Creamy mashed potatoes with gravy", 39.99, 12, 15),
            ("Sweet Potato Casserole", "Brown sugar and pecan topped", 49.99, 12, 15),
            ("Peach Cobbler Tray", "Warm peach cobbler with cinnamon", 54.99, 12, 15),
            ("Banana Pudding Tray", "Layered vanilla wafers and fresh bananas", 44.99, 12, 15),
        ],
        "Greek": [
            ("Gyro Platter", "Seasoned lamb and beef with tzatziki", 119.99, 10, 12),
            ("Souvlaki Skewer Tray", "Grilled {protein} skewers with lemon", 109.99, 10, 12),
            ("Greek Salad Family Size", "Tomatoes, cucumbers, olives, and feta", 54.99, 10, 12),
            ("Moussaka Tray", "Layered eggplant, meat, and bechamel", 99.99, 10, 12),
            ("Dolmades Platter", "Grape leaves stuffed with rice and herbs", 49.99, 12, 15),
            ("Grilled Octopus Platter", "Tender octopus with olive oil and oregano", 139.99, 8, 10),
            ("Pastitsio Tray", "Greek baked pasta with meat sauce", 89.99, 10, 12),
            ("Tzatziki & Pita Spread", "Cucumber yogurt dip with warm pita", 39.99, 10, 12),
            ("Loukoumades Platter", "Greek honey donuts with cinnamon", 44.99, 15, 20),
            ("Saganaki Platter", "Pan-fried cheese with lemon", 54.99, 8, 10),
        ],
        "Thai": [
            ("Pad Thai Tray", "Rice noodles with tamarind sauce and {protein}", 99.99, 10, 12),
            ("Green Curry Tray", "Coconut curry with Thai basil and vegetables", 89.99, 10, 12),
            ("Red Curry Tray", "Spicy coconut curry with bamboo shoots", 89.99, 10, 12),
            ("Thai Basil Chicken Tray", "Stir-fried chicken with holy basil", 99.99, 10, 12),
            ("Tom Yum Soup Station", "Hot and sour soup with shrimp", 59.99, 10, 12),
            ("Satay Skewer Platter", "Grilled {protein} with peanut sauce", 79.99, 12, 15),
            ("Papaya Salad Tray", "Shredded green papaya with lime dressing", 49.99, 10, 12),
            ("Massaman Curry Tray", "Rich curry with potatoes and peanuts", 99.99, 10, 12),
            ("Thai Iced Tea Pitcher", "Sweet creamy Thai tea", 24.99, 10, 12),
            ("Mango Sticky Rice Tray", "Sweet coconut rice with fresh mango", 54.99, 12, 15),
        ],
        "Korean": [
            ("Korean BBQ Kit", "Marinated meats with banchan and lettuce wraps", 149.99, 10, 12),
            ("Bibimbap Bowl Bar", "Rice bowls with vegetables and gochujang", 119.99, 10, 12),
            ("Bulgogi Tray", "Marinated beef with sesame and scallions", 129.99, 10, 12),
            ("Japchae Tray", "Sweet potato glass noodles with vegetables", 79.99, 10, 12),
            ("Kimchi Fried Rice Tray", "Wok-fried rice with kimchi and egg", 69.99, 10, 12),
            ("Korean Fried Chicken Platter", "Crispy double-fried chicken", 99.99, 10, 12),
            ("Tteokbokki Tray", "Spicy rice cakes in gochujang sauce", 54.99, 10, 12),
            ("Kimbap Platter", "Korean rice rolls with vegetables and meat", 69.99, 12, 15),
            ("Banchan Assortment", "Traditional Korean side dishes", 49.99, 10, 12),
            ("Hotteok Platter", "Sweet filled Korean pancakes", 39.99, 12, 15),
        ],
        "Vietnamese": [
            ("Pho Station", "Build your own pho with fresh herbs", 119.99, 10, 12),
            ("Banh Mi Platter", "Vietnamese sandwiches with pickled vegetables", 89.99, 10, 12),
            ("Spring Roll Platter", "Fresh rice paper rolls with peanut sauce", 59.99, 12, 15),
            ("Vermicelli Bowl Bar", "Rice noodles with grilled {protein}", 99.99, 10, 12),
            ("Lemongrass Chicken Tray", "Grilled chicken with lemongrass marinade", 99.99, 10, 12),
            ("Bo Luc Lac Tray", "Shaking beef with watercress", 139.99, 10, 12),
            ("Caramelized Pork Tray", "Clay pot braised pork belly", 109.99, 10, 12),
            ("Goi Cuon Platter", "Summer rolls with shrimp and herbs", 69.99, 12, 15),
            ("Vietnamese Coffee Station", "Drip coffee with condensed milk", 34.99, 10, 12),
            ("Che Dessert Cups", "Assorted Vietnamese sweet desserts", 44.99, 12, 15),
        ],
        "Middle Eastern": [
            ("Mixed Grill Platter", "Assorted grilled meats and kebabs", 149.99, 10, 12),
            ("Shawarma Kit", "Spiced {protein} with pickles and garlic sauce", 129.99, 10, 12),
            ("Falafel Wrap Platter", "Crispy falafel in warm pita", 89.99, 10, 12),
            ("Hummus Trio", "Classic, roasted garlic, and spicy hummus", 54.99, 12, 15),
            ("Fattoush Salad Bowl", "Crispy pita chip salad with sumac", 49.99, 10, 12),
            ("Kibbeh Platter", "Bulgur and lamb croquettes", 79.99, 12, 15),
            ("Tabbouleh Tray", "Fresh parsley and bulgur salad", 44.99, 10, 12),
            ("Manakeesh Platter", "Flatbread with za'atar and cheese", 59.99, 10, 12),
            ("Rice Pilaf Tray", "Fragrant rice with toasted vermicelli", 39.99, 12, 15),
            ("Kunafa Tray", "Sweet cheese pastry with rose syrup", 69.99, 12, 15),
        ],
        "Breakfast": [
            ("Breakfast Burrito Platter", "Scrambled eggs, cheese, and {protein}", 89.99, 10, 12),
            ("Pancake Stack Tray", "Fluffy buttermilk pancakes", 59.99, 10, 12),
            ("Bagel & Lox Spread", "Fresh bagels with cream cheese and smoked salmon", 99.99, 10, 12),
            ("Eggs & Bacon Tray", "Scrambled eggs with crispy bacon", 79.99, 10, 12),
            ("French Toast Platter", "Thick-cut French toast with maple syrup", 69.99, 10, 12),
            ("Breakfast Sandwich Platter", "Egg sandwiches on fresh croissants", 89.99, 10, 12),
            ("Fruit & Yogurt Parfait Cups", "Greek yogurt with granola and berries", 54.99, 10, 12),
            ("Oatmeal Bar", "Steel-cut oats with assorted toppings", 49.99, 10, 12),
            ("Quiche Assortment", "Assorted savory quiches", 79.99, 10, 12),
            ("Pastry Basket", "Assorted muffins, croissants, and danishes", 59.99, 12, 15),
        ],
    }

    SAUCES = ["marinara", "alfredo", "pesto", "vodka", "bolognese", "arrabbiata", "carbonara"]
    PROTEINS = ["chicken", "beef", "shrimp", "tofu", "pork", "lamb", "fish", "turkey"]

    def __init__(self, seed: int | None = None):
        if seed:
            random.seed(seed)
            Faker.seed(seed)

    def _generate_restaurant_name(self, cuisine: str, city: str) -> str:
        """Generate a realistic restaurant name."""
        pattern = random.choice(self.RESTAURANT_NAME_PATTERNS)
        return pattern.format(
            last_name=fake.last_name(),
            cuisine=cuisine,
            city=city,
            adjective=random.choice(self.ADJECTIVES),
        )

    def generate_restaurant(self) -> dict:
        """Generate a mock restaurant."""
        cuisine = random.choice(self.CUISINES)
        city = fake.city()
        state = fake.state_abbr()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        return {
            "metadata": {
                "schemaVersion": "0.1.0",
                "sourcePlatform": "mock",
                "sourcePath": f"/catering/{fake.slug()}",
                "engine": "mock-generator",
                "scrapedAt": now,
                "normalizedAt": now,
                "scrapeRunId": str(uuid4()),
                "contentHash": fake.sha256(),
            },
            "restaurant": {
                "name": self._generate_restaurant_name(cuisine, city),
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
            "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "rrf_score": random.uniform(0.01, 0.05),
        }

    def generate_search_results(self, count: int = 10) -> list[dict]:
        """Generate mock search results."""
        return [self.generate_index_document() for _ in range(count)]
