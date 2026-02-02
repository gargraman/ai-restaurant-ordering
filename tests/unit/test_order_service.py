"""Unit tests for OrderService.

Tests cover order creation, payment processing, retrieval, and state management.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.menu_item import MenuItem
from src.db.models.order import Order, OrderStatus, FulfillmentType
from src.db.models.order_item import OrderItem
from src.db.models.payment import PaymentIntent as DBPaymentIntent, PaymentStatus
from src.db.models.restaurant import Restaurant
from src.db.models.stripe_account import StripeAccount
from src.db.models.tenant import Tenant
from src.db.models.user import User, UserRole as DBUserRole
from src.orders.cart import Cart, CartItem, CartManager
from src.orders.service import (
    OrderError,
    OrderNotFoundError,
    OrderService,
    OrderValidationError,
    PaymentError,
)
from src.payments.models import PaymentIntentResponse
from src.payments.stripe_client import StripeClient


@pytest.fixture
def mock_db():
    """Mock AsyncSession."""
    db = AsyncMock(spec=AsyncSession)
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


@pytest.fixture
def mock_cart_manager():
    """Mock CartManager."""
    return AsyncMock(spec=CartManager)


@pytest.fixture
def mock_stripe_client():
    """Mock StripeClient."""
    return AsyncMock(spec=StripeClient)


@pytest.fixture
def order_service(mock_db, mock_cart_manager, mock_stripe_client):
    """Create OrderService with mocks."""
    return OrderService(
        db=mock_db,
        cart_manager=mock_cart_manager,
        stripe_client=mock_stripe_client,
    )


@pytest.fixture
def sample_cart():
    """Create sample cart for testing."""
    cart_item = CartItem(
        menu_item_id="item-123",
        name="Test Item",
        unit_price_cents=1000,
        quantity=2,
        modifiers=[{"name": "Extra Cheese", "price_cents": 200}],
        special_instructions="No onions",
    )

    cart = Cart(
        tenant_id="tenant-123",
        session_id="session-456",
        restaurant_id="rest-789",
        restaurant_name="Test Restaurant",
        items=[cart_item],
        subtotal_cents=2400,  # (1000 + 200) * 2
    )

    return cart


@pytest.fixture
def sample_restaurant():
    """Create sample restaurant."""
    from src.db.models.stripe_account import StripeAccount, StripeAccountStatus

    stripe_account = StripeAccount(
        id="stripe-123",
        restaurant_id="rest-789",
        stripe_account_id="acct_test123",
        status=StripeAccountStatus.ENABLED,
        charges_enabled=True,
        payouts_enabled=True,
        details_submitted=True,
    )

    restaurant = Restaurant(
        id="rest-789",
        tenant_id="tenant-123",
        name="Test Restaurant",
        is_active=True,
        is_accepting_orders=True,
        estimated_prep_time_minutes=15,
        stripe_account=stripe_account,
    )

    return restaurant


@pytest.fixture
def sample_tenant():
    """Create sample tenant."""
    tenant = Tenant(
        id="tenant-123",
        name="Test Tenant",
        application_fee_percent=Decimal("0.025"),  # 2.5%
    )

    return tenant


@pytest.fixture
def mock_db_result():
    """Mock SQLAlchemy Result object."""
    result = MagicMock()
    result.scalar_one_or_none = MagicMock()
    result.scalar = MagicMock()
    result.scalars = MagicMock()
    result.scalars.return_value.all = MagicMock()
    return result


class TestOrderServiceCreateOrder:
    """Test order creation from cart."""

    def setup_db_mocks(self, mock_db, restaurant=None, tenant=None, menu_item=None, order_count=0):
        """Helper to setup consistent database mocks."""
        call_count = 0
        
        def mock_execute_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result_mock = MagicMock()
            
            if call_count == 1 and restaurant is not None:  # Restaurant query
                result_mock.scalar_one_or_none.return_value = restaurant
            elif call_count == 2 and tenant is not None:  # Tenant query
                result_mock.scalar_one_or_none.return_value = tenant
            elif call_count == 3 and menu_item is not None:  # Menu item query
                result_mock.scalar_one_or_none.return_value = menu_item
            elif call_count == 4:  # Count query in _generate_order_number
                result_mock.scalar.return_value = order_count
            else:
                result_mock.scalar_one_or_none.return_value = None
                
            return result_mock
        
        mock_db.execute.side_effect = mock_execute_side_effect

    @pytest.mark.asyncio
    async def test_create_order_from_cart_success(
        self, order_service, mock_db, mock_cart_manager, sample_cart, sample_restaurant, sample_tenant
    ):
        """Test successful order creation from cart."""
        # Setup mocks
        mock_cart_manager.get_cart.return_value = sample_cart
        self.setup_db_mocks(mock_db, restaurant=sample_restaurant, tenant=sample_tenant, menu_item=MenuItem(id="item-123", name="Test Item", pos_sku_mapping="SKU123"))

        # Execute
        order = await order_service.create_order_from_cart(
            tenant_id="tenant-123",
            session_id="session-456",
            customer_name="John Doe",
            customer_email="john@example.com",
            fulfillment_type="pickup",
        )

        # Assertions
        assert isinstance(order, Order)
        assert order.tenant_id == "tenant-123"
        assert order.restaurant_id == "rest-789"
        assert order.status == OrderStatus.CREATED
        assert order.fulfillment_type == FulfillmentType.PICKUP
        assert order.customer_name == "John Doe"
        assert order.customer_email == "john@example.com"
        assert order.subtotal_cents == 2400
        assert order.total_cents == 2400 + 192 + 0 + 0  # subtotal + tax + tip + delivery
        assert order.application_fee_cents == 60  # 2400 * 0.025

        # Verify DB operations
        assert mock_db.add.call_count == 2  # Order + OrderItem
        assert mock_db.flush.call_count == 2

    @pytest.mark.asyncio
    async def test_create_order_from_empty_cart_raises_error(
        self, order_service, mock_cart_manager
    ):
        """Test order creation fails with empty cart."""
        # Setup empty cart
        empty_cart = Cart(
            tenant_id="tenant-123",
            session_id="session-456",
            restaurant_id=None,
            restaurant_name=None,
            items=[],
            subtotal_cents=0,
        )
        mock_cart_manager.get_cart.return_value = empty_cart

        # Execute and assert
        with pytest.raises(OrderValidationError, match="Cart is empty"):
            await order_service.create_order_from_cart(
                tenant_id="tenant-123",
                session_id="session-456",
            )

    @pytest.mark.asyncio
    async def test_create_order_delivery_without_address_raises_error(
        self, order_service, mock_cart_manager, sample_cart
    ):
        """Test delivery order requires address."""
        mock_cart_manager.get_cart.return_value = sample_cart

        with pytest.raises(OrderValidationError, match="Delivery address required"):
            await order_service.create_order_from_cart(
                tenant_id="tenant-123",
                session_id="session-456",
                fulfillment_type="delivery",
                delivery_address=None,
            )

    @pytest.mark.asyncio
    async def test_create_order_restaurant_not_found_raises_error(
        self, order_service, mock_db, mock_cart_manager, sample_cart
    ):
        """Test order creation fails when restaurant not found."""
        mock_cart_manager.get_cart.return_value = sample_cart

        # Mock restaurant not found
        self.setup_db_mocks(mock_db, restaurant=None)

        with pytest.raises(OrderValidationError, match="Restaurant not found"):
            await order_service.create_order_from_cart(
                tenant_id="tenant-123",
                session_id="session-456",
            )

    @pytest.mark.asyncio
    async def test_create_order_inactive_restaurant_raises_error(
        self, order_service, mock_db, mock_cart_manager, sample_cart
    ):
        """Test order creation fails for inactive restaurant."""
        mock_cart_manager.get_cart.return_value = sample_cart

        # Mock inactive restaurant
        inactive_restaurant = Restaurant(
            id="rest-789",
            tenant_id="tenant-123",
            name="Test Restaurant",
            is_active=False,
            is_accepting_orders=True,
        )

        self.setup_db_mocks(mock_db, restaurant=inactive_restaurant)

        with pytest.raises(OrderValidationError, match="Restaurant is not active"):
            await order_service.create_order_from_cart(
                tenant_id="tenant-123",
                session_id="session-456",
            )

    @pytest.mark.asyncio
    async def test_create_order_restaurant_not_accepting_orders_raises_error(
        self, order_service, mock_db, mock_cart_manager, sample_cart, sample_restaurant
    ):
        """Test order creation fails when restaurant not accepting orders."""
        mock_cart_manager.get_cart.return_value = sample_cart

        # Mock restaurant not accepting orders
        sample_restaurant.is_accepting_orders = False

        self.setup_db_mocks(mock_db, restaurant=sample_restaurant)

        with pytest.raises(OrderValidationError, match="not accepting orders"):
            await order_service.create_order_from_cart(
                tenant_id="tenant-123",
                session_id="session-456",
            )

    @pytest.mark.asyncio
    async def test_create_order_missing_payment_setup_raises_error(
        self, order_service, mock_db, mock_cart_manager, sample_cart
    ):
        """Test order creation fails without payment setup."""
        mock_cart_manager.get_cart.return_value = sample_cart

        # Mock restaurant without stripe account
        restaurant_no_payment = Restaurant(
            id="rest-789",
            tenant_id="tenant-123",
            name="Test Restaurant",
            is_active=True,
            is_accepting_orders=True,
            stripe_account=None,
        )

        self.setup_db_mocks(mock_db, restaurant=restaurant_no_payment)

        with pytest.raises(OrderValidationError, match="payment setup incomplete"):
            await order_service.create_order_from_cart(
                tenant_id="tenant-123",
                session_id="session-456",
            )

    @pytest.mark.asyncio
    async def test_create_order_tenant_not_found_raises_error(
        self, order_service, mock_db, mock_cart_manager, sample_cart, sample_restaurant
    ):
        """Test order creation fails when tenant not found."""
        mock_cart_manager.get_cart.return_value = sample_cart

        # Mock restaurant found, tenant not found
        self.setup_db_mocks(mock_db, restaurant=sample_restaurant, tenant=None)

        with pytest.raises(OrderValidationError, match="Tenant not found"):
            await order_service.create_order_from_cart(
                tenant_id="tenant-123",
                session_id="session-456",
            )

    @pytest.mark.asyncio
    async def test_create_order_with_delivery_fee(
        self, order_service, mock_db, mock_cart_manager, sample_cart, sample_restaurant, sample_tenant
    ):
        """Test order creation with delivery fee calculation."""
        mock_cart_manager.get_cart.return_value = sample_cart

        # Setup mocks
        self.setup_db_mocks(mock_db, restaurant=sample_restaurant, tenant=sample_tenant, menu_item=MenuItem(id="item-123", name="Test Item"))

        # Execute with delivery
        order = await order_service.create_order_from_cart(
            tenant_id="tenant-123",
            session_id="session-456",
            fulfillment_type="delivery",
            delivery_address={"street": "123 Main St", "city": "Test City"},
        )

        # Assertions
        assert order.fulfillment_type == FulfillmentType.DELIVERY
        assert order.delivery_address == {"street": "123 Main St", "city": "Test City"}
        assert order.delivery_fee_cents == 499  # Default delivery fee

    @pytest.mark.asyncio
    async def test_create_order_with_tip(
        self, order_service, mock_db, mock_cart_manager, sample_cart, sample_restaurant, sample_tenant
    ):
        """Test order creation with tip calculation."""
        mock_cart_manager.get_cart.return_value = sample_cart

        # Setup mocks
        self.setup_db_mocks(mock_db, restaurant=sample_restaurant, tenant=sample_tenant, menu_item=MenuItem(id="item-123", name="Test Item"))

        # Execute with tip
        order = await order_service.create_order_from_cart(
            tenant_id="tenant-123",
            session_id="session-456",
            tip_cents=300,  # $3.00 tip
        )

        # Assertions
        assert order.tip_cents == 300
        assert order.total_cents == 2400 + 192 + 300 + 0  # subtotal + tax + tip + delivery


class TestOrderServicePaymentIntent:
    """Test payment intent creation."""

    @pytest.mark.asyncio
    async def test_create_payment_intent_success(
        self, order_service, mock_db, mock_stripe_client, sample_restaurant
    ):
        """Test successful payment intent creation."""
        # Create test order
        order = Order(
            id=uuid4(),
            tenant_id="tenant-123",
            restaurant_id="rest-789",
            order_number="TEST-001",
            idempotency_key="test-key-123",
            total_cents=2500,
            application_fee_cents=62,
        )

        # Mock restaurant query
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_restaurant
        mock_db.execute.return_value = mock_result

        # Mock Stripe payment intent creation
        from src.payments.models import PaymentStatus
        from datetime import datetime
        
        pi_response = PaymentIntentResponse(
            payment_intent_id="pi_test123",
            client_secret="pi_test123_secret_456",
            status=PaymentStatus.PENDING,
            amount_cents=2500,
            application_fee_cents=62,
            connected_account_id="acct_test123",
            metadata={"order_id": str(order.id)},
            created_at=datetime.now(),
        )

        with patch("src.orders.service.PaymentIntentManager") as mock_pi_manager_class:
            mock_pi_manager = AsyncMock()
            mock_pi_manager.create_payment_intent.return_value = pi_response
            mock_pi_manager_class.return_value = mock_pi_manager

            # Execute
            payment_intent = await order_service.create_payment_intent(order)

            # Assertions
            assert isinstance(payment_intent, DBPaymentIntent)
            assert payment_intent.stripe_payment_intent_id == "pi_test123"
            assert payment_intent.amount_cents == 2500
            assert payment_intent.application_fee_cents == 62
            assert payment_intent.client_secret == "pi_test123_secret_456"

    @pytest.mark.asyncio
    async def test_create_payment_intent_no_stripe_client_raises_error(
        self, mock_db, mock_cart_manager
    ):
        """Test payment intent creation fails without Stripe client."""
        # Create service without stripe client
        service = OrderService(db=mock_db, cart_manager=mock_cart_manager, stripe_client=None)

        order = Order(id=uuid4(), total_cents=1000, application_fee_cents=25)

        with pytest.raises(PaymentError, match="Stripe client not configured"):
            await service.create_payment_intent(order)

    @pytest.mark.asyncio
    async def test_create_payment_intent_restaurant_no_stripe_account_raises_error(
        self, order_service, mock_db
    ):
        """Test payment intent creation fails without restaurant Stripe account."""
        order = Order(
            id=uuid4(),
            restaurant_id="rest-789",
            total_cents=1000,
            application_fee_cents=25,
        )

        # Mock restaurant without stripe account
        restaurant_no_stripe = Restaurant(
            id="rest-789",
            tenant_id="tenant-123",
            name="Test Restaurant",
            stripe_account=None,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = restaurant_no_stripe
        mock_db.execute.return_value = mock_result

        with pytest.raises(PaymentError, match="Restaurant Stripe account not found"):
            await order_service.create_payment_intent(order)


class TestOrderServiceRetrieval:
    """Test order retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_order_success(self, order_service, mock_db):
        """Test successful order retrieval."""
        order_id = str(uuid4())
        expected_order = Order(id=order_id, tenant_id="tenant-123")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_order
        mock_db.execute.return_value = mock_result

        order = await order_service.get_order(order_id, "tenant-123")

        assert order == expected_order
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, order_service, mock_db):
        """Test order retrieval returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        order = await order_service.get_order("nonexistent", "tenant-123")

        assert order is None

    @pytest.mark.asyncio
    async def test_get_order_by_number_success(self, order_service, mock_db):
        """Test successful order retrieval by number."""
        expected_order = Order(
            id=uuid4(),
            tenant_id="tenant-123",
            order_number="TEST-001"
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_order
        mock_db.execute.return_value = mock_result

        order = await order_service.get_order_by_number("TEST-001", "tenant-123")

        assert order == expected_order

    @pytest.mark.asyncio
    async def test_list_orders_with_filters(self, order_service, mock_db):
        """Test order listing with filters."""
        orders = [
            Order(id=uuid4(), tenant_id="tenant-123", restaurant_id="rest-1", status=OrderStatus.CREATED),
            Order(id=uuid4(), tenant_id="tenant-123", restaurant_id="rest-1", status=OrderStatus.PAID),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = orders
        mock_db.execute.return_value = mock_result

        result = await order_service.list_orders(
            tenant_id="tenant-123",
            restaurant_id="rest-1",
            status=OrderStatus.CREATED,
            limit=10,
            offset=0,
        )

        assert result == orders


class TestOrderServiceCancellation:
    """Test order cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, order_service, mock_db):
        """Test successful order cancellation."""
        order_id = str(uuid4())
        order = Order(
            id=order_id,
            tenant_id="tenant-123",
            status=OrderStatus.CREATED,
            order_number="TEST-001",
            customer_email="test@example.com",
        )
        order.restaurant = Restaurant(name="Test Restaurant")

        # Mock get_order
        with patch.object(order_service, 'get_order', return_value=order) as mock_get:
            result = await order_service.cancel_order(order_id, "tenant-123", "Customer request")

            assert result == order
            assert result.status == OrderStatus.CANCELED
            assert result.canceled_reason == "Customer request"
            mock_get.assert_called_once_with(order_id, "tenant-123")

    @pytest.mark.asyncio
    async def test_cancel_order_not_found_raises_error(self, order_service):
        """Test cancellation fails when order not found."""
        with patch.object(order_service, 'get_order', return_value=None):
            with pytest.raises(OrderNotFoundError, match="not found"):
                await order_service.cancel_order("nonexistent", "tenant-123")

    @pytest.mark.asyncio
    async def test_cancel_order_invalid_state_raises_error(self, order_service, mock_db):
        """Test cancellation fails for invalid state transitions."""
        order = Order(
            id=uuid4(),
            tenant_id="tenant-123",
            status=OrderStatus.COMPLETED,  # Cannot cancel completed orders
        )

        with patch.object(order_service, 'get_order', return_value=order):
            with pytest.raises(OrderError, match="Cannot cancel order"):
                await order_service.cancel_order(str(order.id), "tenant-123")


class TestOrderServiceHelpers:
    """Test helper methods."""

    def test_calculate_tax(self, order_service):
        """Test tax calculation."""
        restaurant = Restaurant(id="rest-123", name="Test")

        tax = order_service._calculate_tax(1000, restaurant)  # $10.00

        # 8% tax = $0.80 = 80 cents
        assert tax == 80

    def test_calculate_delivery_fee_pickup(self, order_service):
        """Test delivery fee for pickup orders."""
        restaurant = Restaurant(id="rest-123", name="Test")

        fee = order_service._calculate_delivery_fee(FulfillmentType.PICKUP, restaurant)

        assert fee == 0

    def test_calculate_delivery_fee_delivery(self, order_service):
        """Test delivery fee for delivery orders."""
        restaurant = Restaurant(id="rest-123", name="Test")

        fee = order_service._calculate_delivery_fee(FulfillmentType.DELIVERY, restaurant)

        assert fee == 499  # $4.99

    @pytest.mark.asyncio
    async def test_generate_order_number(self, order_service, mock_db):
        """Test order number generation."""
        # Mock count query
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_db.execute.return_value = mock_result

        order_number = await order_service._generate_order_number("tenant-123")

        # Should be 2 letters + dash + 6-digit number
        assert len(order_number) == 9  # XX-XXXXXX
        assert "-" in order_number
        parts = order_number.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 2
        assert len(parts[1]) == 6
        assert parts[1] == "000043"  # 42 + 1

    def test_generate_idempotency_key(self, order_service):
        """Test idempotency key generation."""
        cart = Cart(
            tenant_id="tenant-123",
            session_id="session-456",
            restaurant_id="rest-789",
            restaurant_name="Test Restaurant",
            items=[
                CartItem(
                    menu_item_id="item-1",
                    name="Item 1",
                    unit_price_cents=1000,
                    quantity=1,
                    modifiers=[],
                    special_instructions="",
                )
            ],
            subtotal_cents=1000,
        )

        key1 = order_service._generate_idempotency_key("tenant-123", "session-456", cart)
        key2 = order_service._generate_idempotency_key("tenant-123", "session-456", cart)

        # Same inputs should generate same key
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex length

    @pytest.mark.asyncio
    async def test_clear_cart_after_order(self, order_service, mock_cart_manager):
        """Test cart clearing after order."""
        await order_service.clear_cart_after_order("tenant-123", "session-456")

        mock_cart_manager.clear_cart.assert_called_once_with("tenant-123", "session-456")

    @pytest.mark.asyncio
    async def test_get_or_create_guest_user_existing_email(self, order_service, mock_db):
        """Test getting existing guest user by email."""
        existing_user = User(
            id=uuid4(),
            email="john@example.com",
            role=DBUserRole.CUSTOMER,
            tenant_id="tenant-123",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_user
        mock_db.execute.return_value = mock_result

        user_id = await order_service.get_or_create_guest_user(
            tenant_id="tenant-123",
            guest_identifier="john@example.com",
            name="John Doe",
            email="john@example.com",
        )

        assert user_id == str(existing_user.id)
        # Should not create new user
        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_guest_user_create_new(self, order_service, mock_db):
        """Test creating new guest user."""
        # Mock no existing user found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        user_id = await order_service.get_or_create_guest_user(
            tenant_id="tenant-123",
            guest_identifier="session-123",
            name="Jane Doe",
            email="jane@example.com",
        )

        # Should create new user
        assert mock_db.add.called
        call_args = mock_db.add.call_args[0][0]
        assert isinstance(call_args, User)
        assert call_args.email == "jane@example.com"
        assert call_args.first_name == "Jane"
        assert call_args.last_name == "Doe"
        assert call_args.role == DBUserRole.CUSTOMER