"""API request and response models."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search API request."""

    session_id: str = Field(..., min_length=8, max_length=64)
    user_input: str = Field(..., max_length=500)
    max_results: int = Field(default=10, ge=1, le=50)


class MenuItemResult(BaseModel):
    """Menu item in search results."""

    doc_id: str
    restaurant_name: str
    city: str
    state: str
    item_name: str
    item_description: str | None = None
    display_price: float | None = None
    price_per_person: float | None = None
    serves_min: int | None = None
    serves_max: int | None = None
    dietary_labels: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    rrf_score: float = 0.0


class SearchResponse(BaseModel):
    """Search API response."""

    session_id: str
    resolved_query: str
    intent: str
    is_follow_up: bool
    filters: dict
    results: list[MenuItemResult]
    answer: str
    confidence: float
    processing_time_ms: float


class SessionResponse(BaseModel):
    """Session state response."""

    session_id: str
    created_at: str
    last_activity: str
    entities: dict
    conversation_length: int
    previous_results_count: int


class FeedbackRequest(BaseModel):
    """User feedback on search results."""

    doc_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = None
