"""Tests for LLM provider selection (OpenAI vs DeepSeek)."""

import pytest
from unittest.mock import MagicMock, patch

from src.config.settings import Settings


class TestActiveLlmModel:
    """Tests for Settings.active_llm_model property."""

    def test_openai_returns_openai_model(self):
        settings = Settings(llm_provider="openai", openai_model="gpt-4-turbo-preview")
        assert settings.active_llm_model == "gpt-4-turbo-preview"

    def test_deepseek_returns_deepseek_model(self):
        settings = Settings(llm_provider="deepseek", deepseek_model="deepseek-chat")
        assert settings.active_llm_model == "deepseek-chat"

    def test_deepseek_custom_model(self):
        settings = Settings(llm_provider="deepseek", deepseek_model="deepseek-coder")
        assert settings.active_llm_model == "deepseek-coder"

    def test_default_provider_returns_openai_model(self):
        settings = Settings()
        assert settings.llm_provider == "openai"
        assert settings.active_llm_model == settings.openai_model

    def test_invalid_provider_raises_validation_error(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(llm_provider="anthropic")


class TestValidateRequiredConfigs:
    """Tests for validate_required_configs with DeepSeek."""

    def test_deepseek_without_api_key_raises_error(self):
        settings = Settings(llm_provider="deepseek", deepseek_api_key="")
        errors = settings.validate_required_configs()
        assert any("deepseek_api_key" in e for e in errors)

    def test_deepseek_with_api_key_no_error(self):
        settings = Settings(
            llm_provider="deepseek",
            deepseek_api_key="sk-deepseek-key",
            sendgrid_api_key="sg-key",  # satisfy notification check
        )
        errors = settings.validate_required_configs()
        assert not any("deepseek_api_key" in e for e in errors)

    def test_openai_provider_no_deepseek_error(self):
        settings = Settings(llm_provider="openai", deepseek_api_key="")
        errors = settings.validate_required_configs()
        assert not any("deepseek_api_key" in e for e in errors)

    def test_deepseek_http_base_url_raises_error(self):
        settings = Settings(
            llm_provider="deepseek",
            deepseek_api_key="sk-deepseek-key",
            deepseek_base_url="http://api.deepseek.com",  # http not https
        )
        errors = settings.validate_required_configs()
        assert any("deepseek_base_url" in e for e in errors)

    def test_deepseek_https_base_url_no_error(self):
        settings = Settings(
            llm_provider="deepseek",
            deepseek_api_key="sk-deepseek-key",
            deepseek_base_url="https://api.deepseek.com",
            sendgrid_api_key="sg-key",
        )
        errors = settings.validate_required_configs()
        assert not any("deepseek_base_url" in e for e in errors)

    def test_openai_missing_api_key_is_not_validated(self):
        """openai_api_key is not validated — it can come from the environment."""
        settings = Settings(
            llm_provider="openai",
            openai_api_key="",
            sendgrid_api_key="sg-key",
        )
        errors = settings.validate_required_configs()
        assert not any("openai_api_key" in e for e in errors)

    def test_deepseek_and_payments_both_missing_keys(self):
        settings = Settings(
            llm_provider="deepseek",
            deepseek_api_key="",
            enable_payments=True,
            stripe_secret_key="",
            sendgrid_api_key="sg-key",
        )
        errors = settings.validate_required_configs()
        assert any("deepseek_api_key" in e for e in errors)
        assert any("stripe_secret_key" in e for e in errors)


class TestGetLlm:
    """Tests for _get_llm() provider selection."""

    def setup_method(self):
        """Reset the cached LLM instance before each test."""
        import src.langgraph.nodes as nodes_module
        nodes_module._llm_instance = None

    def teardown_method(self):
        """Reset the cached LLM instance after each test."""
        import src.langgraph.nodes as nodes_module
        nodes_module._llm_instance = None

    def test_openai_provider_uses_openai_settings(self):
        mock_settings = MagicMock()
        mock_settings.llm_provider = "openai"
        mock_settings.openai_model = "gpt-4-turbo-preview"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.active_llm_model = "gpt-4-turbo-preview"

        with (
            patch("src.langgraph.nodes.settings", mock_settings),
            patch("src.langgraph.nodes.ChatOpenAI") as mock_chat,
        ):
            from src.langgraph.nodes import _get_llm
            _get_llm()

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "gpt-4-turbo-preview"
            assert call_kwargs["api_key"] == "sk-test"
            assert call_kwargs["temperature"] == 0
            assert "base_url" not in call_kwargs

    def test_deepseek_provider_uses_deepseek_settings(self):
        mock_settings = MagicMock()
        mock_settings.llm_provider = "deepseek"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings.deepseek_api_key = "sk-deepseek"
        mock_settings.deepseek_base_url = "https://api.deepseek.com"
        mock_settings.active_llm_model = "deepseek-chat"

        with (
            patch("src.langgraph.nodes.settings", mock_settings),
            patch("src.langgraph.nodes.ChatOpenAI") as mock_chat,
        ):
            from src.langgraph.nodes import _get_llm
            _get_llm()

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "deepseek-chat"
            assert call_kwargs["api_key"] == "sk-deepseek"
            assert call_kwargs["base_url"] == "https://api.deepseek.com"
            assert call_kwargs["temperature"] == 0

    def test_openai_without_api_key_omits_api_key(self):
        mock_settings = MagicMock()
        mock_settings.llm_provider = "openai"
        mock_settings.openai_model = "gpt-4-turbo-preview"
        mock_settings.openai_api_key = ""
        mock_settings.active_llm_model = "gpt-4-turbo-preview"

        with (
            patch("src.langgraph.nodes.settings", mock_settings),
            patch("src.langgraph.nodes.ChatOpenAI") as mock_chat,
        ):
            from src.langgraph.nodes import _get_llm
            _get_llm()

            call_kwargs = mock_chat.call_args[1]
            assert "api_key" not in call_kwargs

    def test_deepseek_without_api_key_omits_api_key(self):
        """DeepSeek also omits api_key when empty, matching OpenAI behavior."""
        mock_settings = MagicMock()
        mock_settings.llm_provider = "deepseek"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings.deepseek_api_key = ""
        mock_settings.deepseek_base_url = "https://api.deepseek.com"
        mock_settings.active_llm_model = "deepseek-chat"

        with (
            patch("src.langgraph.nodes.settings", mock_settings),
            patch("src.langgraph.nodes.ChatOpenAI") as mock_chat,
        ):
            from src.langgraph.nodes import _get_llm
            _get_llm()

            call_kwargs = mock_chat.call_args[1]
            assert "api_key" not in call_kwargs

    def test_llm_instance_cached_after_first_call(self):
        """Second call returns same instance without re-constructing ChatOpenAI."""
        mock_settings = MagicMock()
        mock_settings.llm_provider = "openai"
        mock_settings.openai_model = "gpt-4-turbo-preview"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.active_llm_model = "gpt-4-turbo-preview"

        with (
            patch("src.langgraph.nodes.settings", mock_settings),
            patch("src.langgraph.nodes.ChatOpenAI") as mock_chat,
        ):
            from src.langgraph.nodes import _get_llm
            instance1 = _get_llm()
            instance2 = _get_llm()

            assert mock_chat.call_count == 1  # constructed only once
            assert instance1 is instance2


class TestLlmPricing:
    """Tests for per-model pricing in record_llm_call."""

    def test_deepseek_chat_pricing(self):
        from unittest.mock import patch as _patch
        from src.metrics import record_llm_call, LLM_COST_USD

        with _patch.object(LLM_COST_USD.labels(model="deepseek-chat"), "inc") as mock_inc:
            # 1M input tokens = $0.27, 1M output tokens = $1.10 → total $1.37
            record_llm_call(
                "deepseek-chat", "rag_generation", 0.5,
                input_tokens=1_000_000, output_tokens=1_000_000,
            )
            total = mock_inc.call_args[0][0]
            assert abs(total - 1.37) < 0.001

    def test_gpt4_turbo_pricing(self):
        from unittest.mock import patch as _patch
        from src.metrics import record_llm_call, LLM_COST_USD

        with _patch.object(LLM_COST_USD.labels(model="gpt-4-turbo-preview"), "inc") as mock_inc:
            # 1M input = $10, 1M output = $30 → total $40
            record_llm_call(
                "gpt-4-turbo-preview", "rag_generation", 0.5,
                input_tokens=1_000_000, output_tokens=1_000_000,
            )
            total = mock_inc.call_args[0][0]
            assert abs(total - 40.0) < 0.001

    def test_unknown_model_logs_warning_and_uses_fallback(self):
        from unittest.mock import patch as _patch
        from src.metrics import record_llm_call, LLM_COST_USD, _metrics_logger

        with (
            _patch.object(LLM_COST_USD.labels(model="gpt-99"), "inc") as mock_inc,
            _patch.object(_metrics_logger, "warning") as mock_warn,
        ):
            record_llm_call(
                "gpt-99", "intent_detection", 0.1,
                input_tokens=1_000_000, output_tokens=0,
            )
            mock_warn.assert_called_once()
            total = mock_inc.call_args[0][0]
            # Falls back to gpt-4-turbo-preview: $10/M input
            assert abs(total - 10.0) < 0.001

    def test_zero_tokens_zero_cost(self):
        from unittest.mock import patch as _patch
        from src.metrics import record_llm_call, LLM_COST_USD

        with _patch.object(LLM_COST_USD.labels(model="deepseek-chat"), "inc") as mock_inc:
            record_llm_call("deepseek-chat", "intent_detection", 0.05, input_tokens=0, output_tokens=0)
            assert mock_inc.call_args[0][0] == 0.0

    def test_mock_object_token_coercion_does_not_raise(self):
        """The defensive int() conversion in record_llm_call handles mock objects."""
        from src.metrics import record_llm_call
        record_llm_call(
            "deepseek-chat", "intent_detection", 0.5,
            input_tokens=MagicMock(), output_tokens=MagicMock(),
        )
        # Should not raise
