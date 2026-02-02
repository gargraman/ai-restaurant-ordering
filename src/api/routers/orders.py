"""Order management API endpoints."""

from datetime import datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth import CurrentUserDep, TenantIdDep, get_current_user_or_guest
from src.config.settings import get_settings
from src.db import get_db, set_tenant_context
from src.db.models.order import OrderStatus, FulfillmentType
from src.orders.cart import CartManager, Cart
from src.orders.service import (
    OrderService,
    OrderValidationError,
    OrderNotFoundError,
    OrderError,
    PaymentError,
)
from src.payments.stripe_client import StripeClient

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/orders", tags=["Orders"])


# Redis client singleton
_redis_client: Redis | None = None


async def get_redis() -> Redis:
    """Get Redis client."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,
        )
    return _redis_client


async def get_cart_manager() -> CartManager:
    """Get cart manager."""
    redis = await get_redis()
    return CartManager(redis)


async def get_stripe_client() -> StripeClient:
    """Get Stripe client."""
    settings = get_settings()
    return StripeClient(api_key=settings.stripe_secret_key)


# Request/Response Models
class AddToCartRequest(BaseModel):
    """Request to add item to cart."""

    menu_item_id: str
    name: str
    unit_price_cents: int
    restaurant_id: str
    restaurant_name: str
    quantity: int = Field(default=1, ge=1, le=99)
    modifiers: list[dict] = Field(default_factory=list)
    special_instructions: str | None = None


class UpdateCartItemRequest(BaseModel):
    """Request to update cart item quantity."""

    quantity: int = Field(ge=0, le=99)


class CartItemResponse(BaseModel):
    """Cart item in response."""

    id: str
    menu_item_id: str
    name: str
    quantity: int
    unit_price_cents: int
    total_cents: int
    modifiers: list[dict]
    special_instructions: str | None


class CartResponse(BaseModel):
    """Cart response."""

    session_id: str
    restaurant_id: str | None
    restaurant_name: str | None
    items: list[CartItemResponse]
    item_count: int
    subtotal_cents: int


class CreateOrderRequest(BaseModel):
    """Request to create order from cart."""

    customer_name: str = Field(..., min_length=1, max_length=255)
    customer_email: str | None = None
    customer_phone: str | None = None
    fulfillment_type: str = Field(default="pickup", pattern="^(pickup|delivery)$")
    delivery_address: dict | None = None
    notes: str | None = None
    tip_cents: int = Field(default=0, ge=0)
    # Guest checkout identifier - can be email, phone, or session-based
    guest_identifier: str | None = Field(
        None,
        description="Identifier for guest checkout (email, phone, or session ID)"
    )


class OrderItemResponse(BaseModel):
    """Order item in response."""

    id: str
    menu_item_id: str | None
    name: str
    quantity: int
    unit_price_cents: int
    modifiers_price_cents: int
    total_cents: int
    modifiers: list | None
    notes: str | None


class PaymentIntentResponse(BaseModel):
    """Payment intent details."""

    id: str
    stripe_payment_intent_id: str
    status: str
    amount_cents: int
    client_secret: str | None


class OrderResponse(BaseModel):
    """Order response."""

    id: str
    order_number: str
    status: str
    restaurant_id: str
    customer_name: str | None
    customer_email: str | None
    fulfillment_type: str
    subtotal_cents: int
    tax_cents: int
    tip_cents: int
    delivery_fee_cents: int
    total_cents: int
    items: list[OrderItemResponse]
    payment: PaymentIntentResponse | None
    created_at: datetime
    updated_at: datetime


class OrderListResponse(BaseModel):
    """Paginated order list."""

    items: list[OrderResponse]
    total: int
    limit: int
    offset: int


# Cart endpoints
@router.get("/cart", response_model=CartResponse)
async def get_cart(
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    session_id: str = Query(..., description="Session ID"),
    cart_manager: CartManager = Depends(get_cart_manager),
    db: AsyncSession = Depends(get_db),
) -> CartResponse:
    """Get current cart contents."""
    await set_tenant_context(db, tenant_id)
    cart = await cart_manager.get_cart(tenant_id, session_id)

    return CartResponse(
        session_id=cart.session_id,
        restaurant_id=cart.restaurant_id,
        restaurant_name=cart.restaurant_name,
        items=[
            CartItemResponse(
                id=item.id,
                menu_item_id=item.menu_item_id,
                name=item.name,
                quantity=item.quantity,
                unit_price_cents=item.unit_price_cents,
                total_cents=item.total_cents,
                modifiers=item.modifiers,
                special_instructions=item.special_instructions,
            )
            for item in cart.items
        ],
        item_count=cart.item_count,
        subtotal_cents=cart.subtotal_cents,
    )


@router.post("/cart/items", response_model=CartResponse, status_code=status.HTTP_201_CREATED)
async def add_to_cart(
    request: AddToCartRequest,
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    session_id: str = Query(..., description="Session ID"),
    cart_manager: CartManager = Depends(get_cart_manager),
    db: AsyncSession = Depends(get_db),
) -> CartResponse:
    """Add item to cart."""
    await set_tenant_context(db, tenant_id)
    try:
        cart = await cart_manager.add_item(
            tenant_id=tenant_id,
            session_id=session_id,
            menu_item_id=request.menu_item_id,
            name=request.name,
            unit_price_cents=request.unit_price_cents,
            restaurant_id=request.restaurant_id,
            restaurant_name=request.restaurant_name,
            quantity=request.quantity,
            modifiers=request.modifiers,
            special_instructions=request.special_instructions,
        )

        return CartResponse(
            session_id=cart.session_id,
            restaurant_id=cart.restaurant_id,
            restaurant_name=cart.restaurant_name,
            items=[
                CartItemResponse(
                    id=item.id,
                    menu_item_id=item.menu_item_id,
                    name=item.name,
                    quantity=item.quantity,
                    unit_price_cents=item.unit_price_cents,
                    total_cents=item.total_cents,
                    modifiers=item.modifiers,
                    special_instructions=item.special_instructions,
                )
                for item in cart.items
            ],
            item_count=cart.item_count,
            subtotal_cents=cart.subtotal_cents,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.patch("/cart/items/{item_id}", response_model=CartResponse)
async def update_cart_item(
    item_id: str,
    request: UpdateCartItemRequest,
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    session_id: str = Query(..., description="Session ID"),
    cart_manager: CartManager = Depends(get_cart_manager),
    db: AsyncSession = Depends(get_db),
) -> CartResponse:
    """Update cart item quantity (0 to remove)."""
    await set_tenant_context(db, tenant_id)
    try:
        cart = await cart_manager.update_item_quantity(
            tenant_id=tenant_id,
            session_id=session_id,
            item_id=item_id,
            quantity=request.quantity,
        )

        return CartResponse(
            session_id=cart.session_id,
            restaurant_id=cart.restaurant_id,
            restaurant_name=cart.restaurant_name,
            items=[
                CartItemResponse(
                    id=item.id,
                    menu_item_id=item.menu_item_id,
                    name=item.name,
                    quantity=item.quantity,
                    unit_price_cents=item.unit_price_cents,
                    total_cents=item.total_cents,
                    modifiers=item.modifiers,
                    special_instructions=item.special_instructions,
                )
                for item in cart.items
            ],
            item_count=cart.item_count,
            subtotal_cents=cart.subtotal_cents,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/cart/items/{item_id}", response_model=CartResponse)
async def remove_cart_item(
    item_id: str,
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    session_id: str = Query(..., description="Session ID"),
    cart_manager: CartManager = Depends(get_cart_manager),
    db: AsyncSession = Depends(get_db),
) -> CartResponse:
    """Remove item from cart."""
    await set_tenant_context(db, tenant_id)
    try:
        cart = await cart_manager.remove_item(
            tenant_id=tenant_id,
            session_id=session_id,
            item_id=item_id,
        )

        return CartResponse(
            session_id=cart.session_id,
            restaurant_id=cart.restaurant_id,
            restaurant_name=cart.restaurant_name,
            items=[
                CartItemResponse(
                    id=item.id,
                    menu_item_id=item.menu_item_id,
                    name=item.name,
                    quantity=item.quantity,
                    unit_price_cents=item.unit_price_cents,
                    total_cents=item.total_cents,
                    modifiers=item.modifiers,
                    special_instructions=item.special_instructions,
                )
                for item in cart.items
            ],
            item_count=cart.item_count,
            subtotal_cents=cart.subtotal_cents,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/cart", status_code=status.HTTP_204_NO_CONTENT)
async def clear_cart(
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    session_id: str = Query(..., description="Session ID"),
    cart_manager: CartManager = Depends(get_cart_manager),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Clear all items from cart."""
    await set_tenant_context(db, tenant_id)
    await cart_manager.clear_cart(tenant_id, session_id)


# Order endpoints
def _build_order_response(order) -> OrderResponse:
    """Build order response from order object."""
    return OrderResponse(
        id=str(order.id),
        order_number=order.order_number,
        status=order.status.value,
        restaurant_id=str(order.restaurant_id),
        customer_name=order.customer_name,
        customer_email=order.customer_email,
        fulfillment_type=order.fulfillment_type.value,
        subtotal_cents=order.subtotal_cents,
        tax_cents=order.tax_cents,
        tip_cents=order.tip_cents,
        delivery_fee_cents=order.delivery_fee_cents,
        total_cents=order.total_cents,
        items=[
            OrderItemResponse(
                id=str(item.id),
                menu_item_id=str(item.menu_item_id) if item.menu_item_id else None,
                name=item.item_name,
                quantity=item.quantity,
                unit_price_cents=item.unit_price_cents,
                modifiers_price_cents=item.modifiers_price_cents,
                total_cents=item.total_cents,
                modifiers=item.modifiers,
                notes=item.notes,
            )
            for item in order.items
        ],
        payment=PaymentIntentResponse(
            id=str(order.payment_intent.id),
            stripe_payment_intent_id=order.payment_intent.stripe_payment_intent_id,
            status=order.payment_intent.status.value,
            amount_cents=order.payment_intent.amount_cents,
            client_secret=order.payment_intent.client_secret,
        )
        if order.payment_intent
        else None,
        created_at=order.created_at,
        updated_at=order.updated_at,
    )


@router.post("", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(
    request: CreateOrderRequest,
    user: Annotated[CurrentUser, Depends(get_current_user_or_guest)],
    tenant_id: TenantIdDep,
    session_id: str = Query(..., description="Session ID"),
    db: AsyncSession = Depends(get_db),
    cart_manager: CartManager = Depends(get_cart_manager),
    stripe_client: StripeClient = Depends(get_stripe_client),
) -> OrderResponse:
    """Create order from cart and initiate payment."""
    await set_tenant_context(db, tenant_id)

    order_service = OrderService(db, cart_manager, stripe_client)

    try:
        # Determine customer ID based on authentication status
        customer_id = user.user_id if user.user_id != "anonymous" and not user.user_id.startswith("guest_") else None

        # If this is a guest checkout, use the guest identifier
        if not customer_id and request.guest_identifier:
            # Create or retrieve a guest user record
            customer_id = await order_service.get_or_create_guest_user(
                tenant_id=tenant_id,
                guest_identifier=request.guest_identifier,
                name=request.customer_name,
                email=request.customer_email,
                phone=request.customer_phone
            )
        elif not customer_id and user.user_id.startswith("guest_"):
            # If user is already a guest, use their ID
            customer_id = user.user_id

        # Create order
        order = await order_service.create_order_from_cart(
            tenant_id=tenant_id,
            session_id=session_id,
            customer_id=customer_id,
            customer_name=request.customer_name,
            customer_email=request.customer_email,
            customer_phone=request.customer_phone,
            fulfillment_type=request.fulfillment_type,
            delivery_address=request.delivery_address,
            notes=request.notes,
            tip_cents=request.tip_cents,
        )

        # Create payment intent
        payment = await order_service.create_payment_intent(order)

        # Clear cart
        await order_service.clear_cart_after_order(tenant_id, session_id)

        # Reload order with relationships
        order = await order_service.get_order(str(order.id), tenant_id)

        return _build_order_response(order)

    except OrderValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PaymentError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Payment failed: {str(e)}",
        )
    except Exception as e:
        logger.error("Order creation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Order creation failed",
        )


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
    cart_manager: CartManager = Depends(get_cart_manager),
) -> OrderResponse:
    """Get order by ID."""
    await set_tenant_context(db, tenant_id)

    order_service = OrderService(db, cart_manager)
    order = await order_service.get_order(order_id, tenant_id)

    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")

    return _build_order_response(order)


@router.get("", response_model=list[OrderResponse])
async def list_orders(
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
    cart_manager: CartManager = Depends(get_cart_manager),
    restaurant_id: str | None = Query(None),
    order_status: str | None = Query(None, alias="status"),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[OrderResponse]:
    """List orders for the tenant."""
    await set_tenant_context(db, tenant_id)

    # Parse status filter
    status_filter = None
    if order_status:
        try:
            status_filter = OrderStatus(order_status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Valid values: {[s.value for s in OrderStatus]}",
            )

    order_service = OrderService(db, cart_manager)
    orders = await order_service.list_orders(
        tenant_id=tenant_id,
        restaurant_id=restaurant_id,
        customer_id=user.user_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )

    return [_build_order_response(order) for order in orders]


@router.post("/{order_id}/cancel", response_model=OrderResponse)
async def cancel_order(
    order_id: str,
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    reason: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    cart_manager: CartManager = Depends(get_cart_manager),
) -> OrderResponse:
    """Cancel an order."""
    await set_tenant_context(db, tenant_id)

    order_service = OrderService(db, cart_manager)

    try:
        order = await order_service.cancel_order(order_id, tenant_id, reason=reason)
        return _build_order_response(order)
    except OrderNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    except OrderError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
