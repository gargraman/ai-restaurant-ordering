"""Order service for creating and managing orders."""

import secrets
import string
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.models.menu_item import MenuItem
from src.db.models.order import Order, OrderStatus, FulfillmentType
from src.db.models.order_item import OrderItem
from src.db.models.payment import PaymentIntent, PaymentStatus
from src.db.models.restaurant import Restaurant
from src.db.models.tenant import Tenant
from src.db.models.user import User, UserRole as DBUserRole
from src.orders.cart import Cart, CartManager
from src.orders.state_machine import OrderStateMachine
from src.payments.charges import PaymentIntentManager
from src.payments.stripe_client import StripeClient
from src.notifications.service import (
    send_order_confirmation_notification,
    send_order_status_notification,
)

logger = structlog.get_logger(__name__)


class OrderError(Exception):
    """Base exception for order errors."""

    pass


class OrderValidationError(OrderError):
    """Raised when order validation fails."""

    pass


class OrderNotFoundError(OrderError):
    """Raised when order is not found."""

    pass


class PaymentError(OrderError):
    """Raised when payment processing fails."""

    pass


class OrderService:
    """Service for managing orders.

    Handles order creation, payment processing, and status updates.
    """

    def __init__(
        self,
        db: AsyncSession,
        cart_manager: CartManager,
        stripe_client: StripeClient | None = None,
    ) -> None:
        """Initialize order service.

        Args:
            db: Database session
            cart_manager: Cart manager for retrieving carts
            stripe_client: Stripe client for payments
        """
        self.db = db
        self.cart_manager = cart_manager
        self.stripe_client = stripe_client
        self.logger = logger.bind(component="order_service")

    async def create_order_from_cart(
        self,
        tenant_id: str,
        session_id: str,
        customer_id: str | None = None,
        customer_name: str | None = None,
        customer_email: str | None = None,
        customer_phone: str | None = None,
        fulfillment_type: str = "pickup",
        delivery_address: dict | None = None,
        notes: str | None = None,
        tip_cents: int = 0,
    ) -> Order:
        """Create an order from the cart.

        Args:
            tenant_id: Tenant ID
            session_id: Session ID with cart
            customer_id: Optional customer user ID
            customer_name: Customer name
            customer_email: Customer email
            customer_phone: Customer phone
            fulfillment_type: pickup or delivery
            delivery_address: Delivery address (required for delivery)
            notes: Order notes
            tip_cents: Tip amount in cents

        Returns:
            Created order

        Raises:
            OrderValidationError: If validation fails
        """
        # Get cart
        cart = await self.cart_manager.get_cart(tenant_id, session_id)

        if cart.is_empty:
            raise OrderValidationError("Cart is empty")

        if not cart.restaurant_id:
            raise OrderValidationError("Cart has no restaurant")

        # Validate fulfillment
        ft = FulfillmentType(fulfillment_type.lower())
        if ft == FulfillmentType.DELIVERY and not delivery_address:
            raise OrderValidationError("Delivery address required for delivery orders")

        # Get restaurant and validate it can accept orders
        result = await self.db.execute(
            select(Restaurant)
            .options(selectinload(Restaurant.stripe_account))
            .where(Restaurant.id == cart.restaurant_id)
        )
        restaurant = result.scalar_one_or_none()

        if not restaurant:
            raise OrderValidationError("Restaurant not found")

        if not restaurant.is_accepting_orders:
            raise OrderValidationError("Restaurant is not accepting orders")

        if not restaurant.stripe_account or not restaurant.stripe_account.is_ready:
            raise OrderValidationError("Restaurant payment setup incomplete")

        # Get tenant for fee calculation
        result = await self.db.execute(select(Tenant).where(Tenant.id == tenant_id))
        tenant = result.scalar_one_or_none()

        if not tenant:
            raise OrderValidationError("Tenant not found")

        # Calculate totals
        subtotal_cents = cart.subtotal_cents
        tax_cents = self._calculate_tax(subtotal_cents, restaurant)
        delivery_fee_cents = self._calculate_delivery_fee(ft, restaurant)
        total_cents = subtotal_cents + tax_cents + tip_cents + delivery_fee_cents

        # Calculate platform fee
        fee_percent = tenant.get_application_fee_percent()
        application_fee_cents = int(subtotal_cents * float(fee_percent))

        # Generate order number
        order_number = await self._generate_order_number(tenant_id)

        # Create order
        order = Order(
            tenant_id=tenant_id,
            restaurant_id=cart.restaurant_id,
            customer_id=customer_id,
            order_number=order_number,
            idempotency_key=str(uuid4()),
            status=OrderStatus.CREATED,
            fulfillment_type=ft,
            delivery_address=delivery_address,
            customer_name=customer_name,
            customer_email=customer_email,
            customer_phone=customer_phone,
            notes=notes,
            subtotal_cents=subtotal_cents,
            tax_cents=tax_cents,
            tip_cents=tip_cents,
            delivery_fee_cents=delivery_fee_cents,
            discount_cents=0,
            total_cents=total_cents,
            application_fee_cents=application_fee_cents,
        )

        self.db.add(order)
        await self.db.flush()

        # Create order items
        for cart_item in cart.items:
            # Get menu item for snapshot
            result = await self.db.execute(
                select(MenuItem).where(MenuItem.id == cart_item.menu_item_id)
            )
            menu_item = result.scalar_one_or_none()

            # Calculate modifiers price
            modifiers_price_cents = sum(m.get("price_cents", 0) for m in cart_item.modifiers)
            total_cents = (cart_item.unit_price_cents + modifiers_price_cents) * cart_item.quantity

            order_item = OrderItem(
                order_id=str(order.id),
                menu_item_id=cart_item.menu_item_id,
                item_name=cart_item.name,
                quantity=cart_item.quantity,
                unit_price_cents=cart_item.unit_price_cents,
                modifiers_price_cents=modifiers_price_cents,
                total_cents=total_cents,
                modifiers=cart_item.modifiers,
                notes=cart_item.special_instructions,
                pos_sku=menu_item.pos_sku_mapping if menu_item else None,
            )
            self.db.add(order_item)

        await self.db.flush()

        self.logger.info(
            "Order created",
            order_id=str(order.id),
            order_number=order_number,
            total_cents=total_cents,
        )

        # Send confirmation notification (best-effort)
        try:
            estimated_time = f"{restaurant.estimated_prep_time_minutes} minutes"
            await send_order_confirmation_notification(
                customer_email=customer_email,
                customer_phone=customer_phone,
                order_number=order_number,
                restaurant_name=restaurant.name,
                estimated_time=estimated_time,
            )
        except Exception as e:
            self.logger.warning(
                "order_confirmation_notification_failed",
                order_id=str(order.id),
                error=str(e),
            )

        return order

    async def create_payment_intent(self, order: Order) -> PaymentIntent:
        """Create Stripe payment intent for order.

        Args:
            order: Order to pay for

        Returns:
            PaymentIntent record

        Raises:
            PaymentError: If payment creation fails
        """
        if not self.stripe_client:
            raise PaymentError("Stripe client not configured")

        # Get restaurant's Stripe account
        result = await self.db.execute(
            select(Restaurant)
            .options(selectinload(Restaurant.stripe_account))
            .where(Restaurant.id == order.restaurant_id)
        )
        restaurant = result.scalar_one_or_none()

        if not restaurant or not restaurant.stripe_account:
            raise PaymentError("Restaurant Stripe account not found")

        stripe_account_id = restaurant.stripe_account.stripe_account_id

        # Create payment intent manager
        payment_manager = PaymentIntentManager(self.stripe_client)

        try:
            # Create Stripe payment intent
            from src.payments.models import PaymentIntentRequest

            request = PaymentIntentRequest(
                amount_cents=order.total_cents,
                currency="usd",
                connected_account_id=stripe_account_id,
                application_fee_cents=order.application_fee_cents,
                metadata={
                    "order_id": str(order.id),
                    "order_number": order.order_number,
                    "tenant_id": str(order.tenant_id),
                    "restaurant_id": str(order.restaurant_id),
                },
                idempotency_key=order.idempotency_key,
            )

            pi_response = await payment_manager.create_payment_intent(request)

            # Create payment intent record
            payment_intent = PaymentIntent(
                order_id=str(order.id),
                stripe_payment_intent_id=pi_response.payment_intent_id,
                stripe_account_id=stripe_account_id,
                status=PaymentStatus.REQUIRES_PAYMENT_METHOD,
                amount_cents=order.total_cents,
                application_fee_cents=order.application_fee_cents,
                client_secret=pi_response.client_secret,
            )

            self.db.add(payment_intent)
            await self.db.flush()

            self.logger.info(
                "Payment intent created",
                order_id=str(order.id),
                payment_intent_id=pi_response.payment_intent_id,
            )

            return payment_intent

        except Exception as e:
            self.logger.error(
                "Failed to create payment intent",
                order_id=str(order.id),
                error=str(e),
            )
            raise PaymentError(f"Payment creation failed: {str(e)}")

    async def get_order(
        self,
        order_id: str,
        tenant_id: str,
    ) -> Order | None:
        """Get order by ID.

        Args:
            order_id: Order ID
            tenant_id: Tenant ID for authorization

        Returns:
            Order or None if not found
        """
        result = await self.db.execute(
            select(Order)
            .options(
                selectinload(Order.items),
                selectinload(Order.payment_intent),
                selectinload(Order.restaurant),
            )
            .where(Order.id == order_id, Order.tenant_id == tenant_id)
        )
        return result.scalar_one_or_none()

    async def get_order_by_number(
        self,
        order_number: str,
        tenant_id: str,
    ) -> Order | None:
        """Get order by order number.

        Args:
            order_number: Human-readable order number
            tenant_id: Tenant ID for authorization

        Returns:
            Order or None if not found
        """
        result = await self.db.execute(
            select(Order)
            .options(
                selectinload(Order.items),
                selectinload(Order.payment_intent),
            )
            .where(Order.order_number == order_number, Order.tenant_id == tenant_id)
        )
        return result.scalar_one_or_none()

    async def list_orders(
        self,
        tenant_id: str,
        restaurant_id: str | None = None,
        customer_id: str | None = None,
        status: OrderStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Order]:
        """List orders with filters.

        Args:
            tenant_id: Tenant ID
            restaurant_id: Filter by restaurant
            customer_id: Filter by customer
            status: Filter by status
            limit: Max results
            offset: Pagination offset

        Returns:
            List of orders
        """
        query = select(Order).where(Order.tenant_id == tenant_id)

        if restaurant_id:
            query = query.where(Order.restaurant_id == restaurant_id)
        if customer_id:
            query = query.where(Order.customer_id == customer_id)
        if status:
            query = query.where(Order.status == status)

        query = query.order_by(Order.created_at.desc()).limit(limit).offset(offset)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def cancel_order(
        self,
        order_id: str,
        tenant_id: str,
        reason: str | None = None,
    ) -> Order:
        """Cancel an order.

        Args:
            order_id: Order ID
            tenant_id: Tenant ID
            reason: Cancellation reason

        Returns:
            Updated order

        Raises:
            OrderNotFoundError: If order not found
            OrderError: If order cannot be canceled
        """
        order = await self.get_order(order_id, tenant_id)

        if not order:
            raise OrderNotFoundError(f"Order {order_id} not found")

        state_machine = OrderStateMachine(order)

        if not state_machine.can_transition_to(OrderStatus.CANCELED):
            raise OrderError(
                f"Cannot cancel order in {order.status.value} status"
            )

        state_machine.cancel(reason=reason)

        # Send status update notification (best-effort)
        try:
            await send_order_status_notification(
                customer_email=order.customer_email,
                customer_phone=order.customer_phone,
                order_number=order.order_number,
                status=OrderStatus.CANCELED.value,
                restaurant_name=order.restaurant.name if order.restaurant else "",
            )
        except Exception as e:
            self.logger.warning(
                "order_status_notification_failed",
                order_id=str(order.id),
                error=str(e),
            )

        self.logger.info(
            "Order canceled",
            order_id=str(order.id),
            reason=reason,
        )

        return order

    def _calculate_tax(self, subtotal_cents: int, restaurant: Restaurant) -> int:
        """Calculate tax amount.

        Args:
            subtotal_cents: Subtotal in cents
            restaurant: Restaurant for tax rate

        Returns:
            Tax in cents
        """
        # TODO: Integrate with tax calculation service
        # Using a simple 8% rate for now
        tax_rate = Decimal("0.08")
        return int(Decimal(subtotal_cents) * tax_rate)

    def _calculate_delivery_fee(
        self,
        fulfillment_type: FulfillmentType,
        restaurant: Restaurant,
    ) -> int:
        """Calculate delivery fee.

        Args:
            fulfillment_type: Fulfillment type
            restaurant: Restaurant

        Returns:
            Delivery fee in cents
        """
        if fulfillment_type != FulfillmentType.DELIVERY:
            return 0

        # TODO: Dynamic delivery fee calculation
        return 499  # $4.99 default

    async def _generate_order_number(self, tenant_id: str) -> str:
        """Generate unique order number.

        Format: {2-letter prefix}-{6-digit number}
        Example: AB-123456

        Args:
            tenant_id: Tenant ID

        Returns:
            Unique order number
        """
        # Generate random prefix
        prefix = "".join(secrets.choice(string.ascii_uppercase) for _ in range(2))

        # Get next number for tenant (simplified - should use sequence)
        result = await self.db.execute(
            select(func.count(Order.id)).where(Order.tenant_id == tenant_id)
        )
        count = result.scalar() or 0
        number = str(count + 1).zfill(6)

        return f"{prefix}-{number}"

    async def clear_cart_after_order(
        self,
        tenant_id: str,
        session_id: str,
    ) -> None:
        """Clear cart after successful order creation.

        Args:
            tenant_id: Tenant ID
            session_id: Session ID
        """
        await self.cart_manager.clear_cart(tenant_id, session_id)
        self.logger.debug(
            "Cart cleared after order",
            tenant_id=tenant_id,
            session_id=session_id,
        )

    async def get_or_create_guest_user(
        self,
        tenant_id: str,
        guest_identifier: str,
        name: str,
        email: str | None = None,
        phone: str | None = None,
    ) -> str:
        """Get or create a guest user record.

        Args:
            tenant_id: Tenant ID
            guest_identifier: Guest identifier (email, phone, or session-based)
            name: Guest name
            email: Guest email
            phone: Guest phone

        Returns:
            User ID of the guest user
        """
        # Try to find existing guest user by identifier
        # First, try to find by email if provided
        if email:
            result = await self.db.execute(
                select(User).where(
                    User.tenant_id == tenant_id,
                    User.email == email,
                    User.role == DBUserRole.CUSTOMER
                )
            )
            existing_user = result.scalar_one_or_none()

            if existing_user:
                return str(existing_user.id)

        # If not found by email, try to find by guest_identifier
        # This could be a session ID or other identifier
        result = await self.db.execute(
            select(User).where(
                User.tenant_id == tenant_id,
                User.email.like(f"%{guest_identifier}%"),  # Match guest identifier in email
                User.role == DBUserRole.CUSTOMER
            )
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            return str(existing_user.id)

        # Create new guest user
        # Generate a unique email for the guest user
        import uuid
        guest_email = email or f"guest_{uuid.uuid4()}@example.com"

        user = User(
            email=guest_email,
            password_hash="",  # No password for guest users
            first_name=name.split()[0] if name else "",
            last_name=" ".join(name.split()[1:]) if len(name.split()) > 1 else "",
            phone=phone,
            role=DBUserRole.CUSTOMER,
            tenant_id=tenant_id,
            is_active=True,
            is_verified=False,  # Guest users are not verified
        )

        self.db.add(user)
        await self.db.flush()  # Get the user ID

        self.logger.info(
            "Created guest user",
            user_id=str(user.id),
            tenant_id=tenant_id,
            guest_identifier=guest_identifier
        )

        return str(user.id)
