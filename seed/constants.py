"""Shared constants and realistic data for seed generation.

All seed data is deterministic when using the same seed value,
ensuring reproducible test/dev environments.
"""

from decimal import Decimal

# Default random seed for reproducibility
DEFAULT_SEED = 42

# ── Tenant data ──────────────────────────────────────────────────────────────

TENANTS = [
    {
        "name": "Boston Eats Group",
        "slug": "boston-eats",
        "contact_email": "ops@bostoneats.com",
        "contact_phone": "617-555-0100",
        "billing_email": "billing@bostoneats.com",
        "application_fee_percent": Decimal("0.0250"),
    },
    {
        "name": "Bay Area Catering Co",
        "slug": "bayarea-catering",
        "contact_email": "hello@bayareacatering.com",
        "contact_phone": "415-555-0200",
        "billing_email": "finance@bayareacatering.com",
        "application_fee_percent": Decimal("0.0300"),
    },
    {
        "name": "NYC Food Hall Partners",
        "slug": "nyc-foodhall",
        "contact_email": "team@nycfoodhall.com",
        "contact_phone": "212-555-0300",
        "billing_email": "accounts@nycfoodhall.com",
        "application_fee_percent": Decimal("0.0200"),
    },
]

# ── Restaurant data (keyed by tenant slug) ───────────────────────────────────

RESTAURANTS = {
    "boston-eats": [
        {
            "name": "Mama Rosa's Italian Kitchen",
            "slug": "mama-rosas",
            "description": "Authentic Italian catering with family recipes passed down for generations.",
            "address_line1": "123 Hanover St",
            "city": "Boston",
            "state": "MA",
            "postal_code": "02113",
            "latitude": Decimal("42.3634"),
            "longitude": Decimal("-71.0548"),
            "phone": "617-555-1001",
            "email": "orders@mamarosas.com",
            "cuisine": "Italian",
            "accepts_delivery": True,
            "delivery_radius_miles": Decimal("10.00"),
            "minimum_order_amount": Decimal("75.00"),
            "estimated_prep_time_minutes": 45,
        },
        {
            "name": "Dragon Palace",
            "slug": "dragon-palace",
            "description": "Premium Chinese and dim sum catering for events of all sizes.",
            "address_line1": "88 Beach St",
            "city": "Boston",
            "state": "MA",
            "postal_code": "02111",
            "latitude": Decimal("42.3513"),
            "longitude": Decimal("-71.0603"),
            "phone": "617-555-1002",
            "email": "catering@dragonpalace.com",
            "cuisine": "Asian",
            "accepts_delivery": True,
            "delivery_radius_miles": Decimal("15.00"),
            "minimum_order_amount": Decimal("100.00"),
            "estimated_prep_time_minutes": 60,
        },
        {
            "name": "Cambridge BBQ Pit",
            "slug": "cambridge-bbq",
            "description": "Low and slow smoked meats, perfect for large corporate events.",
            "address_line1": "456 Mass Ave",
            "city": "Cambridge",
            "state": "MA",
            "postal_code": "02139",
            "latitude": Decimal("42.3629"),
            "longitude": Decimal("-71.1003"),
            "phone": "617-555-1003",
            "email": "events@cambridgebbq.com",
            "cuisine": "BBQ",
            "accepts_delivery": True,
            "delivery_radius_miles": Decimal("20.00"),
            "minimum_order_amount": Decimal("150.00"),
            "estimated_prep_time_minutes": 90,
        },
    ],
    "bayarea-catering": [
        {
            "name": "Taqueria El Sol",
            "slug": "taqueria-el-sol",
            "description": "Vibrant Mexican catering with build-your-own taco bars and fresh salsas.",
            "address_line1": "789 Mission St",
            "city": "San Francisco",
            "state": "CA",
            "postal_code": "94103",
            "latitude": Decimal("37.7849"),
            "longitude": Decimal("-122.4094"),
            "phone": "415-555-2001",
            "email": "fiestas@taqueriaelsol.com",
            "cuisine": "Mexican",
            "accepts_delivery": True,
            "delivery_radius_miles": Decimal("12.00"),
            "minimum_order_amount": Decimal("80.00"),
            "estimated_prep_time_minutes": 40,
        },
        {
            "name": "Sakura Sushi & Bento",
            "slug": "sakura-sushi",
            "description": "Fresh Japanese catering with sushi platters and bento boxes.",
            "address_line1": "321 Post St",
            "city": "San Francisco",
            "state": "CA",
            "postal_code": "94108",
            "latitude": Decimal("37.7879"),
            "longitude": Decimal("-122.4074"),
            "phone": "415-555-2002",
            "email": "catering@sakurasf.com",
            "cuisine": "Japanese",
            "accepts_delivery": False,
            "delivery_radius_miles": None,
            "minimum_order_amount": Decimal("120.00"),
            "estimated_prep_time_minutes": 50,
        },
    ],
    "nyc-foodhall": [
        {
            "name": "Bombay Spice House",
            "slug": "bombay-spice",
            "description": "Aromatic Indian catering with tandoori specialties and biryanis.",
            "address_line1": "100 Lexington Ave",
            "city": "New York",
            "state": "NY",
            "postal_code": "10016",
            "latitude": Decimal("40.7459"),
            "longitude": Decimal("-73.9813"),
            "phone": "212-555-3001",
            "email": "events@bombayspice.com",
            "cuisine": "Indian",
            "accepts_delivery": True,
            "delivery_radius_miles": Decimal("8.00"),
            "minimum_order_amount": Decimal("100.00"),
            "estimated_prep_time_minutes": 55,
        },
        {
            "name": "Athenian Grill",
            "slug": "athenian-grill",
            "description": "Mediterranean and Greek catering with mezze platters and grilled meats.",
            "address_line1": "250 W 30th St",
            "city": "New York",
            "state": "NY",
            "postal_code": "10001",
            "latitude": Decimal("40.7488"),
            "longitude": Decimal("-73.9936"),
            "phone": "212-555-3002",
            "email": "catering@atheniangrill.com",
            "cuisine": "Greek",
            "accepts_delivery": True,
            "delivery_radius_miles": Decimal("10.00"),
            "minimum_order_amount": Decimal("90.00"),
            "estimated_prep_time_minutes": 45,
        },
        {
            "name": "Seoul Kitchen",
            "slug": "seoul-kitchen",
            "description": "Korean BBQ and bibimbap catering with banchan sides.",
            "address_line1": "32 W 32nd St",
            "city": "New York",
            "state": "NY",
            "postal_code": "10001",
            "latitude": Decimal("40.7479"),
            "longitude": Decimal("-73.9878"),
            "phone": "212-555-3003",
            "email": "orders@seoulkitchen.com",
            "cuisine": "Korean",
            "accepts_delivery": True,
            "delivery_radius_miles": Decimal("7.00"),
            "minimum_order_amount": Decimal("80.00"),
            "estimated_prep_time_minutes": 50,
        },
    ],
}

# ── Menu items by cuisine ────────────────────────────────────────────────────

MENU_ITEMS = {
    "Italian": [
        ("Pasta Tray - Marinara", "Penne with house marinara sauce and fresh basil", "Entrees", Decimal("89.99"), Decimal("7.50"), 10, 12, ["vegetarian"], ["popular"]),
        ("Lasagna Tray", "Layers of pasta, beef ragu, ricotta, and mozzarella", "Entrees", Decimal("109.99"), Decimal("7.33"), 12, 15, [], ["bestseller"]),
        ("Chicken Parmesan Tray", "Breaded chicken cutlets with marinara and melted mozzarella", "Entrees", Decimal("129.99"), Decimal("10.83"), 10, 12, [], ["chef-special"]),
        ("Caesar Salad Bowl", "Crisp romaine, parmesan, croutons, and house Caesar dressing", "Salads & Bowls", Decimal("49.99"), Decimal("5.00"), 8, 10, ["vegetarian"], []),
        ("Antipasto Platter", "Cured meats, imported cheeses, olives, and marinated vegetables", "Appetizers", Decimal("79.99"), Decimal("5.33"), 12, 15, [], ["popular"]),
        ("Garlic Bread Basket", "Warm garlic bread with herb butter", "Sides", Decimal("24.99"), Decimal("2.08"), 10, 12, ["vegetarian"], []),
        ("Tiramisu Tray", "Classic espresso-soaked ladyfingers with mascarpone cream", "Desserts", Decimal("69.99"), Decimal("4.67"), 12, 15, ["vegetarian"], ["chef-special"]),
        ("Eggplant Parmesan Tray", "Breaded eggplant with marinara and fresh mozzarella", "Entrees", Decimal("99.99"), Decimal("8.33"), 10, 12, ["vegetarian"], []),
    ],
    "Asian": [
        ("Kung Pao Chicken Tray", "Wok-tossed chicken with peanuts and chili peppers", "Entrees", Decimal("99.99"), Decimal("8.33"), 10, 12, ["gluten-free"], ["popular"]),
        ("Vegetable Fried Rice Tray", "Wok-fried jasmine rice with seasonal vegetables", "Entrees", Decimal("69.99"), Decimal("5.83"), 10, 12, ["vegan"], []),
        ("Spring Roll Platter", "Crispy vegetable spring rolls with sweet chili sauce", "Appetizers", Decimal("49.99"), Decimal("3.33"), 12, 15, ["vegan"], []),
        ("Dim Sum Assortment", "Chef's selection of steamed and fried dumplings", "Appetizers", Decimal("89.99"), Decimal("7.50"), 10, 12, [], ["chef-special"]),
        ("Orange Chicken Tray", "Crispy chicken in tangy orange glaze", "Entrees", Decimal("89.99"), Decimal("7.50"), 10, 12, [], ["bestseller"]),
        ("Beef & Broccoli Tray", "Tender beef with fresh broccoli in garlic sauce", "Entrees", Decimal("99.99"), Decimal("8.33"), 10, 12, ["gluten-free"], []),
        ("Hot & Sour Soup Pot", "Traditional soup with tofu and mushrooms", "Soups", Decimal("44.99"), Decimal("3.75"), 10, 12, ["vegetarian"], []),
        ("Mango Pudding Cups", "Creamy mango pudding with coconut cream", "Desserts", Decimal("39.99"), Decimal("3.33"), 10, 12, ["vegetarian", "gluten-free"], []),
    ],
    "BBQ": [
        ("Brisket Platter", "14-hour smoked beef brisket sliced to order", "Entrees", Decimal("169.99"), Decimal("14.17"), 10, 12, ["gluten-free"], ["bestseller"]),
        ("Pulled Pork Tray", "Slow-smoked pulled pork with Carolina vinegar sauce", "Entrees", Decimal("119.99"), Decimal("8.00"), 12, 15, ["gluten-free"], ["popular"]),
        ("Baby Back Ribs Rack", "Fall-off-the-bone tender ribs with house dry rub", "Entrees", Decimal("149.99"), Decimal("15.00"), 8, 10, ["gluten-free"], ["chef-special"]),
        ("Smoked Sausage Platter", "House-made smoked sausage links with mustard", "Entrees", Decimal("89.99"), Decimal("7.50"), 10, 12, ["gluten-free"], []),
        ("Coleslaw Tray", "Creamy house-made coleslaw", "Sides", Decimal("34.99"), Decimal("2.33"), 12, 15, ["vegetarian", "gluten-free"], []),
        ("Baked Beans Tray", "Slow-cooked beans with smoked pork belly", "Sides", Decimal("39.99"), Decimal("2.67"), 12, 15, [], []),
        ("Cornbread Basket", "Fresh-baked cornbread with honey butter", "Sides", Decimal("29.99"), Decimal("2.00"), 12, 15, ["vegetarian"], []),
        ("Mac & Cheese Tray", "Smoked gouda and cheddar mac and cheese", "Sides", Decimal("54.99"), Decimal("3.67"), 12, 15, ["vegetarian"], ["popular"]),
    ],
    "Mexican": [
        ("Taco Bar Kit", "Build-your-own taco kit with grilled chicken, steak, and carnitas", "Entrees", Decimal("139.99"), Decimal("11.67"), 10, 12, [], ["bestseller"]),
        ("Burrito Bowl Bar", "Rice, beans, choice of proteins, and all toppings", "Entrees", Decimal("129.99"), Decimal("8.67"), 12, 15, [], ["popular"]),
        ("Chips & Guacamole", "Fresh tortilla chips with house-made guacamole", "Appetizers", Decimal("39.99"), Decimal("4.00"), 8, 10, ["vegan", "gluten-free"], []),
        ("Enchilada Tray", "Rolled tortillas with chicken and red chile sauce", "Entrees", Decimal("99.99"), Decimal("8.33"), 10, 12, [], []),
        ("Street Corn Tray", "Grilled corn with cotija cheese and lime crema", "Sides", Decimal("44.99"), Decimal("3.75"), 10, 12, ["vegetarian", "gluten-free"], ["chef-special"]),
        ("Carnitas Platter", "Slow-roasted pulled pork with cilantro and onions", "Entrees", Decimal("129.99"), Decimal("8.67"), 12, 15, ["gluten-free"], []),
        ("Churros Platter", "Cinnamon sugar churros with chocolate dipping sauce", "Desserts", Decimal("34.99"), Decimal("2.92"), 10, 12, ["vegetarian"], []),
        ("Horchata Pitcher", "Sweet cinnamon rice drink", "Beverages", Decimal("19.99"), Decimal("2.00"), 8, 10, ["vegan", "gluten-free"], []),
    ],
    "Japanese": [
        ("Sushi Party Platter", "Chef's selection of 60 pieces - nigiri and specialty rolls", "Entrees", Decimal("159.99"), Decimal("13.33"), 10, 12, [], ["bestseller"]),
        ("Teriyaki Chicken Tray", "Grilled chicken thighs with teriyaki glaze", "Entrees", Decimal("99.99"), Decimal("8.33"), 10, 12, [], ["popular"]),
        ("Tempura Platter", "Lightly battered shrimp and seasonal vegetables", "Appetizers", Decimal("89.99"), Decimal("7.50"), 10, 12, [], []),
        ("Edamame Tray", "Steamed soybeans with sea salt", "Appetizers", Decimal("29.99"), Decimal("2.50"), 10, 12, ["vegan", "gluten-free"], []),
        ("Bento Box Set", "Individual bento boxes with teriyaki, rice, and salad", "Individual Meals", Decimal("119.99"), Decimal("12.00"), 10, 10, [], ["chef-special"]),
        ("Gyoza Platter", "Pan-fried pork dumplings with dipping sauce", "Appetizers", Decimal("54.99"), Decimal("4.58"), 10, 12, [], []),
        ("Miso Soup Station", "Traditional miso with tofu, seaweed, and scallions", "Soups", Decimal("39.99"), Decimal("3.33"), 10, 12, ["vegetarian"], []),
        ("Matcha Mochi Tray", "Assorted matcha and fruit mochi ice cream", "Desserts", Decimal("49.99"), Decimal("3.33"), 12, 15, ["vegetarian"], []),
    ],
    "Indian": [
        ("Butter Chicken Tray", "Tender chicken in rich tomato-cream curry", "Entrees", Decimal("109.99"), Decimal("9.17"), 10, 12, ["gluten-free"], ["bestseller"]),
        ("Vegetable Biryani Tray", "Fragrant basmati rice with mixed vegetables and saffron", "Entrees", Decimal("79.99"), Decimal("6.67"), 10, 12, ["vegan", "gluten-free"], ["popular"]),
        ("Samosa Platter", "Crispy pastries filled with spiced potatoes and peas", "Appetizers", Decimal("49.99"), Decimal("3.33"), 12, 15, ["vegetarian"], []),
        ("Chicken Tikka Masala Tray", "Grilled chicken in spiced masala sauce", "Entrees", Decimal("119.99"), Decimal("10.00"), 10, 12, ["gluten-free"], ["chef-special"]),
        ("Naan Bread Basket", "Assorted fresh-baked naan: plain, garlic, and butter", "Sides", Decimal("29.99"), Decimal("2.50"), 10, 12, ["vegetarian"], []),
        ("Dal Makhani Tray", "Creamy black lentils simmered overnight", "Entrees", Decimal("69.99"), Decimal("5.83"), 10, 12, ["vegan", "gluten-free"], []),
        ("Tandoori Chicken Platter", "Clay oven roasted chicken with yogurt marinade", "Entrees", Decimal("99.99"), Decimal("8.33"), 10, 12, ["gluten-free"], []),
        ("Mango Lassi Pitcher", "Sweet yogurt drink with Alphonso mango", "Beverages", Decimal("24.99"), Decimal("2.50"), 8, 10, ["vegetarian", "gluten-free"], []),
    ],
    "Greek": [
        ("Gyro Platter", "Seasoned lamb and beef with tzatziki and warm pita", "Entrees", Decimal("119.99"), Decimal("10.00"), 10, 12, [], ["bestseller"]),
        ("Greek Salad Family Size", "Tomatoes, cucumbers, kalamata olives, and feta", "Salads & Bowls", Decimal("54.99"), Decimal("4.58"), 10, 12, ["vegetarian", "gluten-free"], ["popular"]),
        ("Souvlaki Skewer Tray", "Grilled chicken skewers with lemon and oregano", "Entrees", Decimal("109.99"), Decimal("9.17"), 10, 12, ["gluten-free"], []),
        ("Spanakopita Tray", "Flaky phyllo pastry with spinach and feta filling", "Appetizers", Decimal("59.99"), Decimal("4.00"), 12, 15, ["vegetarian"], []),
        ("Hummus & Pita Spread", "House-made hummus with warm pita triangles", "Appetizers", Decimal("44.99"), Decimal("3.75"), 10, 12, ["vegan"], []),
        ("Moussaka Tray", "Layered eggplant, seasoned lamb, and bechamel", "Entrees", Decimal("99.99"), Decimal("8.33"), 10, 12, [], ["chef-special"]),
        ("Baklava Tray", "Sweet phyllo pastry with honey and crushed pistachios", "Desserts", Decimal("49.99"), Decimal("2.50"), 15, 20, ["vegetarian"], []),
        ("Tzatziki & Pita Spread", "Cucumber yogurt dip with herbs and warm pita", "Appetizers", Decimal("39.99"), Decimal("3.33"), 10, 12, ["vegetarian"], []),
    ],
    "Korean": [
        ("Korean BBQ Kit", "Marinated bulgogi, pork belly, and chicken with banchan", "Entrees", Decimal("149.99"), Decimal("12.50"), 10, 12, [], ["bestseller"]),
        ("Bibimbap Bowl Bar", "Rice bowls with vegetables, egg, and gochujang", "Entrees", Decimal("119.99"), Decimal("10.00"), 10, 12, [], ["popular"]),
        ("Japchae Tray", "Sweet potato glass noodles with vegetables and sesame", "Entrees", Decimal("79.99"), Decimal("6.67"), 10, 12, ["vegan"], []),
        ("Korean Fried Chicken Platter", "Crispy double-fried chicken with yangnyeom sauce", "Entrees", Decimal("99.99"), Decimal("8.33"), 10, 12, [], ["chef-special"]),
        ("Kimchi Fried Rice Tray", "Wok-fried rice with aged kimchi and sesame", "Entrees", Decimal("69.99"), Decimal("5.83"), 10, 12, [], []),
        ("Banchan Assortment", "Traditional Korean side dishes - 8 varieties", "Sides", Decimal("49.99"), Decimal("4.17"), 10, 12, ["vegetarian"], []),
        ("Tteokbokki Tray", "Spicy rice cakes in gochujang sauce", "Sides", Decimal("54.99"), Decimal("4.58"), 10, 12, ["vegan"], []),
        ("Hotteok Platter", "Sweet filled Korean pancakes with brown sugar", "Desserts", Decimal("39.99"), Decimal("2.67"), 12, 15, ["vegetarian"], []),
    ],
}

# ── Operating hours template ─────────────────────────────────────────────────

DEFAULT_OPERATING_HOURS = {
    "monday": {"open": "09:00", "close": "21:00"},
    "tuesday": {"open": "09:00", "close": "21:00"},
    "wednesday": {"open": "09:00", "close": "21:00"},
    "thursday": {"open": "09:00", "close": "21:00"},
    "friday": {"open": "09:00", "close": "22:00"},
    "saturday": {"open": "10:00", "close": "22:00"},
    "sunday": {"open": "10:00", "close": "20:00"},
}

# ── User templates ───────────────────────────────────────────────────────────

# password for all seed users: "SeedPass123!"
DEFAULT_PASSWORD = "SeedPass123!"

PLATFORM_ADMIN = {
    "email": "admin@platform.com",
    "first_name": "Platform",
    "last_name": "Admin",
    "phone": "800-555-0001",
}

# Per-tenant admin + customer users (keyed by tenant slug)
TENANT_USERS = {
    "boston-eats": {
        "admin": {
            "email": "admin@bostoneats.com",
            "first_name": "Sarah",
            "last_name": "Johnson",
            "phone": "617-555-0101",
        },
        "customers": [
            {"email": "mike.chen@example.com", "first_name": "Mike", "last_name": "Chen", "phone": "617-555-0110"},
            {"email": "lisa.rodriguez@example.com", "first_name": "Lisa", "last_name": "Rodriguez", "phone": "617-555-0111"},
        ],
    },
    "bayarea-catering": {
        "admin": {
            "email": "admin@bayareacatering.com",
            "first_name": "David",
            "last_name": "Kim",
            "phone": "415-555-0201",
        },
        "customers": [
            {"email": "emily.watson@example.com", "first_name": "Emily", "last_name": "Watson", "phone": "415-555-0210"},
            {"email": "alex.patel@example.com", "first_name": "Alex", "last_name": "Patel", "phone": "415-555-0211"},
        ],
    },
    "nyc-foodhall": {
        "admin": {
            "email": "admin@nycfoodhall.com",
            "first_name": "Jordan",
            "last_name": "Williams",
            "phone": "212-555-0301",
        },
        "customers": [
            {"email": "priya.sharma@example.com", "first_name": "Priya", "last_name": "Sharma", "phone": "212-555-0310"},
            {"email": "tom.baker@example.com", "first_name": "Tom", "last_name": "Baker", "phone": "212-555-0311"},
        ],
    },
}

# Restaurant admin/staff per restaurant slug
RESTAURANT_STAFF = {
    "mama-rosas": {"email": "rosa@mamarosas.com", "first_name": "Rosa", "last_name": "Bianchi", "phone": "617-555-1010"},
    "dragon-palace": {"email": "wei@dragonpalace.com", "first_name": "Wei", "last_name": "Zhang", "phone": "617-555-1020"},
    "cambridge-bbq": {"email": "james@cambridgebbq.com", "first_name": "James", "last_name": "Walker", "phone": "617-555-1030"},
    "taqueria-el-sol": {"email": "carlos@taqueriaelsol.com", "first_name": "Carlos", "last_name": "Ramirez", "phone": "415-555-2010"},
    "sakura-sushi": {"email": "yuki@sakurasf.com", "first_name": "Yuki", "last_name": "Tanaka", "phone": "415-555-2020"},
    "bombay-spice": {"email": "raj@bombayspice.com", "first_name": "Raj", "last_name": "Gupta", "phone": "212-555-3010"},
    "athenian-grill": {"email": "nikos@atheniangrill.com", "first_name": "Nikos", "last_name": "Papadopoulos", "phone": "212-555-3020"},
    "seoul-kitchen": {"email": "jin@seoulkitchen.com", "first_name": "Jin", "last_name": "Park", "phone": "212-555-3030"},
}

# ── POS connection templates ─────────────────────────────────────────────────

POS_CONFIGS = {
    "mama-rosas": {"provider": "square", "merchant_id": "MLR-BOSTON-001", "location_id": "LOC-MR-01"},
    "dragon-palace": {"provider": "toast", "merchant_id": "DP-BOSTON-001", "location_id": "LOC-DP-01"},
    "cambridge-bbq": {"provider": "square", "merchant_id": "CBQ-CAMB-001", "location_id": "LOC-CBQ-01"},
    "taqueria-el-sol": {"provider": "square", "merchant_id": "TES-SF-001", "location_id": "LOC-TES-01"},
    "sakura-sushi": {"provider": "toast", "merchant_id": "SS-SF-001", "location_id": "LOC-SS-01"},
    "bombay-spice": {"provider": "square", "merchant_id": "BS-NYC-001", "location_id": "LOC-BS-01"},
    "athenian-grill": {"provider": "olo", "merchant_id": "AG-NYC-001", "location_id": "LOC-AG-01"},
    "seoul-kitchen": {"provider": "square", "merchant_id": "SK-NYC-001", "location_id": "LOC-SK-01"},
}
