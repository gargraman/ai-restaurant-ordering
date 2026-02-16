"""Unit tests for src/pos module."""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.pos.exceptions import (
    POSAuthError,
    POSConnectionError,
    POSError,
    POSNotFoundError,
    POSRateLimitError,
    POSValidationError,
)
from src.pos.models import (
    POSCatalogItem,
    POSCatalogModifier,
    POSCatalogModifierList,
    POSCatalogSyncResult,
    POSCredentials,
    POSOrderLineItem,
    POSOrderRequest,
    POSOrderResult,
    POSOrderStatus,
    POSWebhookEvent,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TestPOSExceptions:

    def test_pos_error_str_with_provider(self):
        err = POSError("something broke", provider="square")
        assert str(err) == "[square] something broke"

    def test_pos_error_str_without_provider(self):
        err = POSError("something broke")
        assert str(err) == "something broke"

    def test_pos_error_details_default(self):
        err = POSError("msg")
        assert err.details == {}

    def test_pos_error_details_provided(self):
        err = POSError("msg", details={"key": "val"})
        assert err.details == {"key": "val"}

    def test_pos_auth_error_defaults(self):
        err = POSAuthError()
        assert err.message == "Authentication failed"
        assert err.needs_reauth is True

    def test_pos_auth_error_custom(self):
        err = POSAuthError("bad token", provider="square", needs_reauth=False)
        assert err.needs_reauth is False
        assert "[square]" in str(err)

    def test_pos_rate_limit_error(self):
        err = POSRateLimitError(retry_after_seconds=30)
        assert err.retry_after_seconds == 30
        assert err.message == "Rate limit exceeded"

    def test_pos_validation_error(self):
        err = POSValidationError(field="email")
        assert err.field == "email"

    def test_pos_not_found_error(self):
        err = POSNotFoundError(resource_type="order", resource_id="ord-123")
        assert err.resource_type == "order"
        assert err.resource_id == "ord-123"

    def test_pos_connection_error_retryable(self):
        err = POSConnectionError(is_retryable=True)
        assert err.is_retryable is True

    def test_pos_connection_error_not_retryable(self):
        err = POSConnectionError(is_retryable=False)
        assert err.is_retryable is False

    def test_all_exceptions_inherit_from_pos_error(self):
        for cls in [POSAuthError, POSRateLimitError, POSValidationError,
                     POSNotFoundError, POSConnectionError]:
            assert issubclass(cls, POSError)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestPOSOrderLineItem:

    def test_total_cents_no_modifiers(self):
        item = POSOrderLineItem(
            menu_item_id="m1", pos_item_id="p1",
            name="Burger", quantity=2, unit_price_cents=1000,
        )
        assert item.total_cents == 2000

    def test_total_cents_with_modifiers(self):
        item = POSOrderLineItem(
            menu_item_id="m1", pos_item_id="p1",
            name="Burger", quantity=1, unit_price_cents=1000,
            modifiers=[{"name": "Cheese", "price_cents": 150}],
        )
        assert item.total_cents == 1150

    def test_total_cents_multiple_modifiers(self):
        item = POSOrderLineItem(
            menu_item_id="m1", pos_item_id="p1",
            name="Burger", quantity=3, unit_price_cents=1000,
            modifiers=[
                {"name": "Cheese", "price_cents": 100},
                {"name": "Bacon", "price_cents": 200},
            ],
        )
        assert item.total_cents == 3900  # (1000+100+200)*3


class TestPOSOrderResult:

    def test_is_retryable_rate_limited(self):
        result = POSOrderResult(success=False, error_code="RATE_LIMITED")
        assert result.is_retryable is True

    def test_is_retryable_timeout(self):
        result = POSOrderResult(success=False, error_code="TIMEOUT")
        assert result.is_retryable is True

    def test_is_retryable_service_unavailable(self):
        result = POSOrderResult(success=False, error_code="SERVICE_UNAVAILABLE")
        assert result.is_retryable is True

    def test_is_not_retryable(self):
        result = POSOrderResult(success=False, error_code="INVALID_REQUEST")
        assert result.is_retryable is False

    def test_is_retryable_no_error_code(self):
        result = POSOrderResult(success=True)
        assert result.is_retryable is False


class TestPOSCredentials:

    def test_not_expired_when_no_expiry(self):
        creds = POSCredentials(access_token="tok")
        assert creds.is_expired() is False

    def test_expired_token(self):
        creds = POSCredentials(
            access_token="tok",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert creds.is_expired() is True

    def test_not_expired_far_future(self):
        creds = POSCredentials(
            access_token="tok",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        assert creds.is_expired() is False

    def test_expired_within_buffer(self):
        # 3 minutes from now is within the 5-minute buffer
        creds = POSCredentials(
            access_token="tok",
            expires_at=datetime.utcnow() + timedelta(minutes=3),
        )
        assert creds.is_expired() is True

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            POSCredentials(access_token="tok", unknown_field="x")


class TestPOSOrderStatus:

    def test_enum_values(self):
        assert POSOrderStatus.PENDING == "pending"
        assert POSOrderStatus.COMPLETED == "completed"
        assert POSOrderStatus.CANCELED == "canceled"
        assert POSOrderStatus.UNKNOWN == "unknown"


class TestPOSCatalogModels:

    def test_catalog_item_defaults(self):
        item = POSCatalogItem(pos_item_id="p1", name="Pizza", price_cents=1500)
        assert item.is_available is True
        assert item.modifiers == []
        assert item.description is None

    def test_catalog_modifier(self):
        mod = POSCatalogModifier(pos_modifier_id="mod1", name="Extra Cheese", price_cents=200)
        assert mod.is_default is False

    def test_catalog_modifier_list(self):
        ml = POSCatalogModifierList(
            pos_modifier_list_id="ml1", name="Toppings",
            selection_type="multiple", max_selections=3,
        )
        assert ml.min_selections == 0
        assert ml.modifiers == []

    def test_catalog_sync_result(self):
        result = POSCatalogSyncResult(success=True, items_synced=10, items_created=5)
        assert result.items_updated == 0
        assert result.errors == []


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class TestPOSAdapterBase:

    def test_map_to_order_status(self):
        from src.pos.base import POSAdapter
        from src.db.models.order import OrderStatus

        creds = POSCredentials(access_token="tok")

        # Create a concrete subclass for testing the concrete method
        class DummyAdapter(POSAdapter):
            provider_name = "dummy"
            async def validate_connection(self): ...
            async def sync_catalog(self, *a, **kw): ...
            async def fetch_catalog_items(self, *a, **kw): ...
            async def inject_order(self, *a, **kw): ...
            async def get_order_status(self, *a, **kw): ...
            async def cancel_order(self, *a, **kw): ...
            def normalize_status(self, s): ...
            def parse_webhook(self, *a, **kw): ...

        adapter = DummyAdapter(creds)

        assert adapter.map_to_order_status(POSOrderStatus.PENDING) == OrderStatus.SENT_TO_POS
        assert adapter.map_to_order_status(POSOrderStatus.ACCEPTED) == OrderStatus.ACCEPTED
        assert adapter.map_to_order_status(POSOrderStatus.PREPARING) == OrderStatus.PREPARING
        assert adapter.map_to_order_status(POSOrderStatus.READY) == OrderStatus.READY
        assert adapter.map_to_order_status(POSOrderStatus.COMPLETED) == OrderStatus.COMPLETED
        assert adapter.map_to_order_status(POSOrderStatus.CANCELED) == OrderStatus.CANCELED
        assert adapter.map_to_order_status(POSOrderStatus.REJECTED) == OrderStatus.FAILED

    @pytest.mark.asyncio
    async def test_refresh_credentials_default(self):
        from src.pos.base import POSAdapter

        creds = POSCredentials(access_token="tok")

        class DummyAdapter(POSAdapter):
            provider_name = "dummy"
            async def validate_connection(self): ...
            async def sync_catalog(self, *a, **kw): ...
            async def fetch_catalog_items(self, *a, **kw): ...
            async def inject_order(self, *a, **kw): ...
            async def get_order_status(self, *a, **kw): ...
            async def cancel_order(self, *a, **kw): ...
            def normalize_status(self, s): ...
            def parse_webhook(self, *a, **kw): ...

        adapter = DummyAdapter(creds)
        result = await adapter.refresh_credentials()
        assert result is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestGetAdapter:

    def test_square_returns_adapter(self):
        from src.pos.registry import get_adapter, POSProvider

        creds = POSCredentials(access_token="tok", merchant_id="m1")
        adapter = get_adapter(POSProvider.SQUARE, creds)
        assert adapter.provider_name == "square"

    def test_toast_raises(self):
        from src.pos.registry import get_adapter, POSProvider

        creds = POSCredentials(access_token="tok")
        with pytest.raises(POSError, match="Toast"):
            get_adapter(POSProvider.TOAST, creds)

    def test_olo_raises(self):
        from src.pos.registry import get_adapter, POSProvider

        creds = POSCredentials(access_token="tok")
        with pytest.raises(POSError, match="Olo"):
            get_adapter(POSProvider.OLO, creds)

    def test_omnivore_raises(self):
        from src.pos.registry import get_adapter, POSProvider

        creds = POSCredentials(access_token="tok")
        with pytest.raises(POSError, match="Omnivore"):
            get_adapter(POSProvider.OMNIVORE, creds)


class TestPOSRegistry:

    @pytest.mark.asyncio
    async def test_get_adapter_no_credentials_raises(self):
        from src.pos.registry import POSRegistry, POSProvider

        with pytest.raises(POSAuthError, match="No credentials"):
            await POSRegistry.get_adapter(POSProvider.SQUARE, credentials_encrypted=None)

    @pytest.mark.asyncio
    async def test_get_adapter_with_credentials(self):
        from src.pos.registry import POSRegistry, POSProvider

        creds_data = {"access_token": "tok123", "merchant_id": "m1"}

        with patch("src.pos.registry.decrypt_credentials", return_value=creds_data):
            adapter = await POSRegistry.get_adapter(
                POSProvider.SQUARE,
                credentials_encrypted="encrypted_blob",
                merchant_id="m1",
            )
            assert adapter.provider_name == "square"

    @pytest.mark.asyncio
    async def test_get_adapter_bad_json_raises(self):
        from src.pos.registry import POSRegistry, POSProvider

        with patch("src.pos.registry.decrypt_credentials", side_effect=json.JSONDecodeError("x", "", 0)):
            with pytest.raises(POSAuthError, match="Invalid credentials"):
                await POSRegistry.get_adapter(
                    POSProvider.SQUARE,
                    credentials_encrypted="bad",
                )


# ---------------------------------------------------------------------------
# Square order adapter
# ---------------------------------------------------------------------------

class TestSquareOrderAdapter:

    def _make_order_request(self, **overrides):
        defaults = dict(
            order_id="ord-1",
            order_number="ORD-001",
            idempotency_key="idem-1",
            location_id="loc-1",
            line_items=[
                POSOrderLineItem(
                    menu_item_id="m1", pos_item_id="p1",
                    name="Burger", quantity=2, unit_price_cents=1000,
                ),
            ],
            fulfillment_type="pickup",
            subtotal_cents=2000,
            total_cents=2200,
        )
        defaults.update(overrides)
        return POSOrderRequest(**defaults)

    def test_to_square_order_basic(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        req = self._make_order_request()
        result = adapter.to_square_order(req)

        assert result["idempotency_key"] == "idem-1"
        order = result["order"]
        assert order["location_id"] == "loc-1"
        assert order["state"] == "OPEN"
        assert len(order["line_items"]) == 1
        assert order["line_items"][0]["quantity"] == "2"

    def test_to_square_order_with_taxes(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        req = self._make_order_request(tax_cents=200)
        result = adapter.to_square_order(req)

        assert "taxes" in result["order"]
        assert result["order"]["taxes"][0]["applied_money"]["amount"] == 200

    def test_to_square_order_with_tip_and_delivery_fee(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        req = self._make_order_request(tip_cents=500, delivery_fee_cents=300)
        result = adapter.to_square_order(req)

        charges = result["order"]["service_charges"]
        assert len(charges) == 2
        names = {c["name"] for c in charges}
        assert "Tip" in names
        assert "Delivery Fee" in names

    def test_to_square_order_with_discount(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        req = self._make_order_request(discount_cents=100)
        result = adapter.to_square_order(req)

        assert "discounts" in result["order"]

    def test_to_square_order_delivery_fulfillment(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        req = self._make_order_request(
            fulfillment_type="delivery",
            customer_name="John",
            delivery_address={"street1": "123 Main", "city": "Boston", "state": "MA", "zip": "02101"},
        )
        result = adapter.to_square_order(req)
        fulfillment = result["order"]["fulfillments"][0]
        assert fulfillment["type"] == "DELIVERY"

    def test_from_square_response_success(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        response = {
            "order": {
                "id": "sq-ord-1",
                "state": "OPEN",
            }
        }
        result = adapter.from_square_response(response)
        assert result.success is True
        assert result.pos_order_id == "sq-ord-1"
        assert result.pos_status == POSOrderStatus.PENDING

    def test_from_square_response_no_order(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        result = adapter.from_square_response({})
        assert result.success is False
        assert result.error_code == "NO_ORDER"

    def test_from_square_response_with_fulfillment(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        response = {
            "order": {
                "id": "sq-ord-1",
                "state": "OPEN",
                "fulfillments": [{"state": "PREPARED"}],
            }
        }
        result = adapter.from_square_response(response)
        assert result.pos_status == POSOrderStatus.PREPARING

    def test_normalize_status_order_states(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        assert adapter.normalize_status("OPEN") == POSOrderStatus.PENDING
        assert adapter.normalize_status("CANCELED") == POSOrderStatus.CANCELED
        assert adapter.normalize_status("DRAFT") == POSOrderStatus.PENDING

    def test_normalize_status_fulfillment_states(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        assert adapter.normalize_status("PROPOSED") == POSOrderStatus.PENDING
        assert adapter.normalize_status("RESERVED") == POSOrderStatus.ACCEPTED
        assert adapter.normalize_status("PREPARED") == POSOrderStatus.PREPARING

    def test_normalize_status_unknown(self):
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        assert adapter.normalize_status("SOMETHING_ELSE") == POSOrderStatus.UNKNOWN

    def test_line_item_ad_hoc(self):
        """Test line item without pos_item_id or pos_variation_id."""
        from src.pos.square.order_adapter import SquareOrderAdapter

        adapter = SquareOrderAdapter()
        # No pos_variation_id and empty pos_item_id
        req = self._make_order_request(
            line_items=[
                POSOrderLineItem(
                    menu_item_id="m1", pos_item_id="",
                    name="Custom Item", quantity=1, unit_price_cents=500,
                ),
            ],
        )
        result = adapter.to_square_order(req)
        li = result["order"]["line_items"][0]
        assert li["name"] == "Custom Item"
        assert li["base_price_money"]["amount"] == 500


# ---------------------------------------------------------------------------
# Square webhook handler
# ---------------------------------------------------------------------------

class TestSquareWebhookHandler:

    def _make_payload(self, event_type="order.updated", order_state="OPEN", **extra):
        data = {
            "type": event_type,
            "event_id": "evt-1",
            "merchant_id": "merch-1",
            "created_at": "2026-01-15T12:00:00Z",
            "data": {
                "object": {
                    "order": {
                        "id": "sq-ord-1",
                        "location_id": "loc-1",
                        "state": order_state,
                        **extra,
                    }
                }
            },
        }
        return json.dumps(data).encode()

    def test_parse_webhook_basic(self):
        from src.pos.square.webhooks import SquareWebhookHandler

        handler = SquareWebhookHandler()
        payload = self._make_payload()
        event = handler.parse_webhook(payload)

        assert event.event_id == "evt-1"
        assert event.merchant_id == "merch-1"
        assert event.pos_order_id == "sq-ord-1"
        assert event.event_type == "order.updated"

    def test_parse_webhook_order_created(self):
        from src.pos.square.webhooks import SquareWebhookHandler

        handler = SquareWebhookHandler()
        payload = self._make_payload(event_type="order.created")
        event = handler.parse_webhook(payload)

        assert event.new_status == POSOrderStatus.PENDING

    def test_parse_webhook_invalid_json(self):
        from src.pos.square.webhooks import SquareWebhookHandler

        handler = SquareWebhookHandler()
        with pytest.raises(POSValidationError, match="Invalid JSON"):
            handler.parse_webhook(b"not json")

    def test_is_supported_event(self):
        from src.pos.square.webhooks import SquareWebhookHandler

        handler = SquareWebhookHandler()
        assert handler.is_supported_event("order.created") is True
        assert handler.is_supported_event("order.updated") is True
        assert handler.is_supported_event("order.fulfillment.updated") is True
        assert handler.is_supported_event("payment.created") is False

    def test_verify_signature_no_key_skips(self):
        from src.pos.square.webhooks import SquareWebhookHandler

        handler = SquareWebhookHandler(signature_key=None)
        assert handler.verify_signature(b"body", "sig", "http://x") is True

    def test_verify_signature_valid(self):
        from src.pos.square.webhooks import SquareWebhookHandler

        key = "my-secret-key"
        handler = SquareWebhookHandler(signature_key=key)

        url = "https://example.com/webhook"
        body = b'{"test": true}'
        string_to_sign = url.encode() + body
        expected = base64.b64encode(
            hmac.new(key.encode(), string_to_sign, hashlib.sha256).digest()
        ).decode()

        assert handler.verify_signature(body, expected, url) is True

    def test_verify_signature_invalid(self):
        from src.pos.square.webhooks import SquareWebhookHandler

        handler = SquareWebhookHandler(signature_key="my-secret-key")
        with pytest.raises(POSValidationError, match="Invalid webhook signature"):
            handler.verify_signature(b"body", "bad-sig", "http://x")

    def test_extract_order_update(self):
        from src.pos.square.webhooks import SquareWebhookHandler

        handler = SquareWebhookHandler()
        payload = self._make_payload(
            event_type="order.fulfillment.updated",
            fulfillments=[{"state": "COMPLETED", "type": "PICKUP",
                           "pickup_details": {"pickup_at": "2026-01-15T13:00:00Z"}}],
        )
        event = handler.parse_webhook(payload)
        update = handler.extract_order_update(event)

        assert update["pos_order_id"] == "sq-ord-1"
        assert update["fulfillment_state"] == "COMPLETED"
        assert update["fulfillment_type"] == "PICKUP"


# ---------------------------------------------------------------------------
# Square client - _handle_error
# ---------------------------------------------------------------------------

class TestSquareClientHandleError:

    def _make_client(self):
        from src.pos.square.client import SquareClient

        creds = POSCredentials(access_token="tok", merchant_id="m1")
        return SquareClient(creds)

    def _mock_response(self, status_code, json_data=None, headers=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.headers = headers or {}
        resp.text = json.dumps(json_data or {})
        if json_data is not None:
            resp.json.return_value = json_data
        else:
            resp.json.side_effect = Exception("no json")
        return resp

    def test_handle_error_401(self):
        client = self._make_client()
        resp = self._mock_response(401, {"errors": [{"code": "UNAUTHORIZED", "detail": "bad token"}]})
        with pytest.raises(POSAuthError) as exc_info:
            client._handle_error(resp)
        assert exc_info.value.needs_reauth is True

    def test_handle_error_403_expired(self):
        client = self._make_client()
        resp = self._mock_response(403, {"errors": [{"code": "ACCESS_TOKEN_EXPIRED", "detail": "expired"}]})
        with pytest.raises(POSAuthError) as exc_info:
            client._handle_error(resp)
        assert exc_info.value.needs_reauth is True

    def test_handle_error_403_other(self):
        client = self._make_client()
        resp = self._mock_response(403, {"errors": [{"code": "FORBIDDEN", "detail": "nope"}]})
        with pytest.raises(POSAuthError) as exc_info:
            client._handle_error(resp)
        assert exc_info.value.needs_reauth is False

    def test_handle_error_404(self):
        client = self._make_client()
        resp = self._mock_response(404, {"errors": [{"code": "NOT_FOUND", "detail": "gone"}]})
        with pytest.raises(POSNotFoundError):
            client._handle_error(resp)

    def test_handle_error_429_with_retry_after(self):
        client = self._make_client()
        resp = self._mock_response(
            429,
            {"errors": [{"code": "RATE_LIMITED", "detail": "slow down"}]},
            headers={"Retry-After": "60"},
        )
        with pytest.raises(POSRateLimitError) as exc_info:
            client._handle_error(resp)
        assert exc_info.value.retry_after_seconds == 60

    def test_handle_error_400(self):
        client = self._make_client()
        resp = self._mock_response(400, {"errors": [{"code": "BAD_REQUEST", "detail": "invalid"}]})
        with pytest.raises(POSValidationError):
            client._handle_error(resp)

    def test_handle_error_500(self):
        client = self._make_client()
        resp = self._mock_response(500, {"errors": [{"code": "INTERNAL", "detail": "boom"}]})
        with pytest.raises(POSConnectionError) as exc_info:
            client._handle_error(resp)
        assert exc_info.value.is_retryable is True

    def test_handle_error_unknown_status(self):
        client = self._make_client()
        resp = self._mock_response(418, {"errors": [{"code": "TEAPOT", "detail": "short and stout"}]})
        with pytest.raises(POSError):
            client._handle_error(resp)

    def test_get_headers(self):
        client = self._make_client()
        headers = client._get_headers()
        assert "Bearer tok" in headers["Authorization"]
        assert "Square-Version" in headers
        assert headers["Content-Type"] == "application/json"
