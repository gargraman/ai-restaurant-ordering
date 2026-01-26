"""LLM prompts for the search pipeline."""

INTENT_DETECTION_PROMPT = """Classify the user's intent for a catering menu search system.

Categories:
- search: Looking for menu items or restaurants
- filter: Refining previous results (cheaper, more servings, dietary)
- clarify: Asking a question about previous results
- compare: Comparing options

Session Context:
- Previous query: {previous_query}
- Current entities: {entities}
- Has previous results: {has_results}

User Input: {user_input}

Output JSON only, no explanation:
{{
  "intent": "search|filter|clarify|compare",
  "is_follow_up": true|false,
  "follow_up_type": "price|serving|dietary|location|scope|null",
  "confidence": 0.0-1.0
}}"""

ENTITY_EXTRACTION_PROMPT = """Extract search entities from the user query for catering menu search.

Known Fields:
- city: City name
- state: State abbreviation (MA, NY, CA)
- cuisine: Cuisine types (Italian, Mexican, Asian, American, etc.)
- dietary_labels: vegetarian, vegan, gluten-free, dairy-free, nut-free, halal, kosher
- price_max: Maximum total price
- price_per_person_max: Maximum price per person
- serves_min: Minimum number of people to serve
- serves_max: Maximum number of people to serve
- tags: popular, new, seasonal, chef-special
- menu_type: Catering, Lunch, Dinner, Breakfast
- item_keywords: Specific food items (pasta, chicken, salad)

Previous Context:
{previous_entities}

User Input: {user_input}

Output JSON only (include only fields mentioned or implied):
{{
  "city": "string or null",
  "state": "string or null",
  "cuisine": ["array"] or null,
  "dietary_labels": ["array"] or null,
  "price_max": number or null,
  "price_per_person_max": number or null,
  "serves_min": number or null,
  "serves_max": number or null,
  "item_keywords": ["array"] or null,
  "price_adjustment": "increase|decrease|null"
}}"""

QUERY_EXPANSION_PROMPT = """Expand the user query into a search-optimized query for catering menus.

Task: Create a text query that will match relevant menu items via BM25 search.

User Query: {user_input}
Resolved Entities: {entities}

Guidelines:
1. Include the main food/cuisine terms
2. Add common synonyms (e.g., "chicken parm" → "chicken parmesan")
3. Include serving context if relevant (catering, party, corporate)
4. Keep it natural, not keyword-stuffed

Output: Single line expanded query only, no explanation."""

RAG_GENERATION_PROMPT = """You are a helpful catering menu assistant. Answer the user's question using ONLY the provided menu items.

Guidelines:
1. Only recommend items from the provided context
2. Include prices, serving sizes, and dietary info when relevant
3. If asked about serving size, calculate if the option fits their party size
4. If no items match the criteria, say so clearly
5. Mention the restaurant name for each recommendation
6. Be concise but helpful
7. Format prices as currency ($X.XX)
8. If serving info exists, mention price per person

User Question: {question}

Applied Filters:
- City: {city}
- Cuisine: {cuisine}
- Dietary: {dietary}
- Budget: {price_max}
- Party size: {serves_min} people

Available Menu Items:
{context}

Provide a helpful response:"""

CLARIFICATION_PROMPT = """The user's request is missing important information for a catering menu search.

User Request: {user_input}
Current Context: {entities}

What information is missing? Common needs:
- Location (city/area)
- Party size (number of people)
- Budget constraints
- Dietary requirements
- Cuisine preferences

Generate a brief, friendly clarification question that asks for the most important missing information.

Response:"""
