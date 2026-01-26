"""LangGraph state and session models."""

from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class SearchFilters(TypedDict, total=False):
    """Search filters extracted from user query."""

    city: str
    state: str
    zip_code: str
    cuisine: list[str]
    dietary_labels: list[str]
    price_max: float
    price_per_person_max: float
    serves_min: int
    serves_max: int
    tags: list[str]
    restaurant_name: str
    restaurant_id: str
    menu_type: str  # Catering, Lunch, Dinner


IntentType = Literal["search", "filter", "clarify", "compare"]
FollowUpType = Literal["price", "serving", "dietary", "location", "scope"]


class GraphState(TypedDict):
    """LangGraph pipeline state."""

    # Session
    session_id: str
    user_input: str
    timestamp: str

    # Intent
    intent: IntentType
    is_follow_up: bool
    follow_up_type: FollowUpType | None
    confidence: float

    # Resolved query
    resolved_query: str
    filters: SearchFilters
    expanded_query: str

    # Retrieval
    candidate_doc_ids: list[str]
    bm25_results: list[dict[str, Any]]
    vector_results: list[dict[str, Any]]

    # Fusion
    merged_results: list[dict[str, Any]]
    final_context: list[dict[str, Any]]

    # Output
    answer: str
    sources: list[str]

    # Error handling
    error: str | None


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    result_ids: list[str] = Field(default_factory=list)


class SessionEntities(BaseModel):
    """Tracked entities across conversation turns."""

    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    cuisine: list[str] = Field(default_factory=list)
    dietary_labels: list[str] = Field(default_factory=list)
    price_max: float | None = None
    price_per_person_max: float | None = None
    serves_min: int | None = None
    serves_max: int | None = None
    restaurant_name: str | None = None
    restaurant_id: str | None = None
    tags: list[str] = Field(default_factory=list)

    def to_filters(self) -> SearchFilters:
        """Convert to SearchFilters dict."""
        filters: SearchFilters = {}
        if self.city:
            filters["city"] = self.city
        if self.state:
            filters["state"] = self.state
        if self.zip_code:
            filters["zip_code"] = self.zip_code
        if self.cuisine:
            filters["cuisine"] = self.cuisine
        if self.dietary_labels:
            filters["dietary_labels"] = self.dietary_labels
        if self.price_max is not None:
            filters["price_max"] = self.price_max
        if self.price_per_person_max is not None:
            filters["price_per_person_max"] = self.price_per_person_max
        if self.serves_min is not None:
            filters["serves_min"] = self.serves_min
        if self.serves_max is not None:
            filters["serves_max"] = self.serves_max
        if self.restaurant_name:
            filters["restaurant_name"] = self.restaurant_name
        if self.restaurant_id:
            filters["restaurant_id"] = self.restaurant_id
        if self.tags:
            filters["tags"] = self.tags
        return filters

    def update_from_filters(self, filters: SearchFilters) -> None:
        """Update entities from filters dict."""
        if "city" in filters:
            self.city = filters["city"]
        if "state" in filters:
            self.state = filters["state"]
        if "zip_code" in filters:
            self.zip_code = filters["zip_code"]
        if "cuisine" in filters:
            self.cuisine = filters["cuisine"]
        if "dietary_labels" in filters:
            self.dietary_labels = filters["dietary_labels"]
        if "price_max" in filters:
            self.price_max = filters["price_max"]
        if "price_per_person_max" in filters:
            self.price_per_person_max = filters["price_per_person_max"]
        if "serves_min" in filters:
            self.serves_min = filters["serves_min"]
        if "serves_max" in filters:
            self.serves_max = filters["serves_max"]
        if "restaurant_name" in filters:
            self.restaurant_name = filters["restaurant_name"]
        if "restaurant_id" in filters:
            self.restaurant_id = filters["restaurant_id"]
        if "tags" in filters:
            self.tags = filters["tags"]


class SessionPreferences(BaseModel):
    """User preferences inferred from conversation."""

    price_sensitivity: Literal["low", "medium", "high"] = "medium"
    preferred_cuisines: list[str] = Field(default_factory=list)


class SessionState(BaseModel):
    """Complete session state stored in Redis."""

    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: int = 86400  # 24 hours

    entities: SessionEntities = Field(default_factory=SessionEntities)
    conversation: list[ConversationTurn] = Field(default_factory=list)
    previous_results: list[str] = Field(default_factory=list)
    previous_query: str | None = None
    preferences: SessionPreferences = Field(default_factory=SessionPreferences)

    def add_user_turn(self, content: str) -> None:
        """Add a user message to conversation."""
        self.conversation.append(
            ConversationTurn(role="user", content=content)
        )
        self.last_activity = datetime.utcnow()

    def add_assistant_turn(self, content: str, result_ids: list[str] | None = None) -> None:
        """Add an assistant response to conversation."""
        self.conversation.append(
            ConversationTurn(
                role="assistant",
                content=content,
                result_ids=result_ids or [],
            )
        )
        if result_ids:
            self.previous_results = result_ids
        self.last_activity = datetime.utcnow()

    def get_recent_conversation(self, max_turns: int = 5) -> list[ConversationTurn]:
        """Get recent conversation turns."""
        return self.conversation[-max_turns * 2:] if self.conversation else []
