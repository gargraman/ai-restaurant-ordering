"""Mock scenarios for different testing conditions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ScenarioType(str, Enum):
    """Available mock scenarios."""

    HAPPY_PATH = "happy_path"
    EMPTY_RESULTS = "empty_results"
    ERROR_SCENARIO = "error_scenario"
    SLOW_RESPONSE = "slow_response"
    RATE_LIMITED = "rate_limited"
    PARTIAL_FAILURE = "partial_failure"


@dataclass
class Scenario:
    """Definition of a mock scenario."""

    name: str
    description: str
    response_delay_ms: int = 0
    error_rate: float = 0.0
    error_code: int = 500
    error_message: str = "Internal server error"
    empty_results: bool = False
    result_count: int = 10
    custom_data: dict = field(default_factory=dict)


class ScenarioManager:
    """Manage mock scenarios for testing."""

    def __init__(self):
        self.current_scenario = ScenarioType.HAPPY_PATH
        self._scenarios = self._initialize_scenarios()
        self._request_counts: dict[str, int] = {}

    def _initialize_scenarios(self) -> dict[ScenarioType, Scenario]:
        """Initialize predefined scenarios."""
        return {
            ScenarioType.HAPPY_PATH: Scenario(
                name="happy_path",
                description="All operations succeed with realistic data",
                response_delay_ms=50,
                result_count=10,
            ),
            ScenarioType.EMPTY_RESULTS: Scenario(
                name="empty_results",
                description="Search returns no results",
                response_delay_ms=50,
                empty_results=True,
                result_count=0,
            ),
            ScenarioType.ERROR_SCENARIO: Scenario(
                name="error_scenario",
                description="Various error conditions",
                error_rate=0.5,
                error_code=500,
                error_message="Simulated server error",
            ),
            ScenarioType.SLOW_RESPONSE: Scenario(
                name="slow_response",
                description="Slow responses simulating network latency",
                response_delay_ms=3000,
                result_count=10,
            ),
            ScenarioType.RATE_LIMITED: Scenario(
                name="rate_limited",
                description="Rate limiting after N requests",
                custom_data={"rate_limit_after": 5},
            ),
            ScenarioType.PARTIAL_FAILURE: Scenario(
                name="partial_failure",
                description="Some services fail while others succeed",
                error_rate=0.3,
                result_count=5,
            ),
        }

    def set_scenario(self, scenario: ScenarioType | str) -> None:
        """Set the current active scenario."""
        if isinstance(scenario, str):
            scenario = ScenarioType(scenario)
        self.current_scenario = scenario
        self._request_counts = {}

    def get_scenario(self) -> Scenario:
        """Get the current scenario configuration."""
        return self._scenarios[self.current_scenario]

    def should_return_error(self, endpoint: str) -> bool:
        """Check if this request should return an error based on scenario."""
        scenario = self.get_scenario()

        # Rate limiting check
        if self.current_scenario == ScenarioType.RATE_LIMITED:
            self._request_counts[endpoint] = self._request_counts.get(endpoint, 0) + 1
            limit = scenario.custom_data.get("rate_limit_after", 10)
            return self._request_counts[endpoint] > limit

        # Random error based on error_rate
        if scenario.error_rate > 0:
            import random
            return random.random() < scenario.error_rate

        return False

    def get_response_delay(self) -> int:
        """Get the response delay in milliseconds."""
        return self.get_scenario().response_delay_ms

    def should_return_empty(self) -> bool:
        """Check if results should be empty."""
        return self.get_scenario().empty_results

    def get_result_count(self) -> int:
        """Get the number of results to return."""
        return self.get_scenario().result_count

    def reset(self) -> None:
        """Reset scenario state."""
        self._request_counts = {}

    def list_scenarios(self) -> list[dict]:
        """List all available scenarios."""
        return [
            {
                "name": s.value,
                "description": self._scenarios[s].description,
            }
            for s in ScenarioType
        ]


# Pre-built scenario configurations for specific test cases
SEARCH_SCENARIOS = {
    "italian_boston": {
        "filters": {"city": "Boston", "cuisine": ["Italian"]},
        "expected_count": 5,
    },
    "vegetarian_options": {
        "filters": {"dietary_labels": ["vegetarian"]},
        "expected_count": 3,
    },
    "large_group": {
        "filters": {"serves_min": 50},
        "expected_count": 2,
    },
    "budget_friendly": {
        "filters": {"price_max": 50.0},
        "expected_count": 4,
    },
}

CONVERSATION_SCENARIOS = {
    "multi_turn_refinement": {
        "turns": [
            {"user": "Find catering in Boston", "expected_intent": "search"},
            {"user": "vegetarian options", "expected_intent": "filter", "is_follow_up": True},
            {"user": "cheaper ones", "expected_intent": "filter", "is_follow_up": True},
        ]
    },
    "clarification_needed": {
        "turns": [
            {"user": "food for a party", "expected_intent": "clarify"},
        ]
    },
    "serving_size_refinement": {
        "turns": [
            {"user": "Italian for 25 people in Cambridge", "expected_intent": "search"},
            {"user": "that's too many, more like 15", "expected_intent": "filter", "is_follow_up": True},
        ]
    },
}
