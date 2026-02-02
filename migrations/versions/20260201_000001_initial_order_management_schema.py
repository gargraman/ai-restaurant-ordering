"""Initial order management schema with RLS policies.

Revision ID: 20260201_000001
Revises:
Create Date: 2026-02-01

This migration creates:
- All Phase 0 core tables for order management
- Row-Level Security (RLS) policies for tenant isolation
- Required indexes for performance
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260201_000001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE user_role AS ENUM ('platform_admin', 'tenant_admin', 'restaurant_admin', 'restaurant_staff', 'customer')")
    op.execute("CREATE TYPE stripe_account_status AS ENUM ('pending', 'enabled', 'restricted', 'disabled')")
    op.execute("CREATE TYPE pos_provider AS ENUM ('square', 'toast', 'olo', 'omnivore')")
    op.execute("CREATE TYPE pos_connection_status AS ENUM ('pending', 'connected', 'disconnected', 'error', 'expired')")
    op.execute("CREATE TYPE order_status AS ENUM ('created', 'paid', 'sent_to_pos', 'accepted', 'preparing', 'ready', 'completed', 'failed', 'canceled', 'refunded')")
    op.execute("CREATE TYPE fulfillment_type AS ENUM ('pickup', 'delivery')")
    op.execute("CREATE TYPE payment_status AS ENUM ('requires_payment_method', 'requires_confirmation', 'requires_action', 'processing', 'succeeded', 'canceled', 'requires_capture')")
    op.execute("CREATE TYPE refund_status AS ENUM ('pending', 'succeeded', 'failed', 'canceled')")
    op.execute("CREATE TYPE refund_reason AS ENUM ('customer_request', 'restaurant_canceled', 'item_unavailable', 'quality_issue', 'late_delivery', 'wrong_order', 'duplicate', 'fraudulent', 'other')")
    op.execute("CREATE TYPE webhook_provider AS ENUM ('stripe', 'square', 'toast', 'olo', 'omnivore')")
    op.execute("CREATE TYPE webhook_event_status AS ENUM ('received', 'processing', 'processed', 'failed', 'ignored')")

    # =========================================================================
    # TENANTS TABLE
    # =========================================================================
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("contact_email", sa.String(255), nullable=False),
        sa.Column("contact_phone", sa.String(20), nullable=True),
        sa.Column("billing_email", sa.String(255), nullable=True),
        sa.Column("application_fee_percent", sa.Numeric(5, 4), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_tenants"),
        sa.UniqueConstraint("slug", name="uq_tenants_slug"),
    )

    # =========================================================================
    # USERS TABLE
    # =========================================================================
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("first_name", sa.String(100), nullable=True),
        sa.Column("last_name", sa.String(100), nullable=True),
        sa.Column("phone", sa.String(20), nullable=True),
        sa.Column("role", postgresql.ENUM("platform_admin", "tenant_admin", "restaurant_admin", "restaurant_staff", "customer", name="user_role", create_type=False), nullable=False, server_default="customer"),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("restaurant_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("reset_token", sa.String(255), nullable=True),
        sa.Column("reset_token_expires", sa.DateTime(), nullable=True),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_users_tenant_id_tenants", ondelete="CASCADE"),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    op.create_index("ix_users_email_tenant", "users", ["email", "tenant_id"])

    # =========================================================================
    # RESTAURANTS TABLE
    # =========================================================================
    op.create_table(
        "restaurants",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("address_line1", sa.String(255), nullable=False),
        sa.Column("address_line2", sa.String(255), nullable=True),
        sa.Column("city", sa.String(100), nullable=False),
        sa.Column("state", sa.String(50), nullable=False),
        sa.Column("postal_code", sa.String(20), nullable=False),
        sa.Column("country", sa.String(2), nullable=False, server_default="US"),
        sa.Column("latitude", sa.Numeric(10, 7), nullable=True),
        sa.Column("longitude", sa.Numeric(10, 7), nullable=True),
        sa.Column("timezone", sa.String(50), nullable=False, server_default="America/New_York"),
        sa.Column("phone", sa.String(20), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("operating_hours", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("accepts_pickup", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("accepts_delivery", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("delivery_radius_miles", sa.Numeric(5, 2), nullable=True),
        sa.Column("minimum_order_amount", sa.Numeric(10, 2), nullable=True),
        sa.Column("estimated_prep_time_minutes", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("is_accepting_orders", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_restaurants"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_restaurants_tenant_id_tenants", ondelete="CASCADE"),
    )
    op.create_index("ix_restaurants_tenant_id", "restaurants", ["tenant_id"])
    op.create_index("ix_restaurants_tenant_slug", "restaurants", ["tenant_id", "slug"], unique=True)
    op.create_index("ix_restaurants_location", "restaurants", ["city", "state", "postal_code"])

    # Add restaurant_id FK to users after restaurants table exists
    op.create_foreign_key(
        "fk_users_restaurant_id_restaurants",
        "users",
        "restaurants",
        ["restaurant_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # =========================================================================
    # STRIPE ACCOUNTS TABLE
    # =========================================================================
    op.create_table(
        "stripe_accounts",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("restaurant_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("stripe_account_id", sa.String(255), nullable=False),
        sa.Column("status", postgresql.ENUM("pending", "enabled", "restricted", "disabled", name="stripe_account_status", create_type=False), nullable=False, server_default="pending"),
        sa.Column("charges_enabled", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("payouts_enabled", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("details_submitted", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("business_name", sa.String(255), nullable=True),
        sa.Column("business_type", sa.String(50), nullable=True),
        sa.Column("onboarding_url", sa.String(2048), nullable=True),
        sa.Column("onboarding_expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_stripe_accounts"),
        sa.UniqueConstraint("restaurant_id", name="uq_stripe_accounts_restaurant_id"),
        sa.UniqueConstraint("stripe_account_id", name="uq_stripe_accounts_stripe_account_id"),
        sa.ForeignKeyConstraint(["restaurant_id"], ["restaurants.id"], name="fk_stripe_accounts_restaurant_id_restaurants", ondelete="CASCADE"),
    )
    op.create_index("ix_stripe_accounts_restaurant_id", "stripe_accounts", ["restaurant_id"])
    op.create_index("ix_stripe_accounts_stripe_id", "stripe_accounts", ["stripe_account_id"])

    # =========================================================================
    # POS CONNECTIONS TABLE
    # =========================================================================
    op.create_table(
        "pos_connections",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("restaurant_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("provider", postgresql.ENUM("square", "toast", "olo", "omnivore", name="pos_provider", create_type=False), nullable=False),
        sa.Column("status", postgresql.ENUM("pending", "connected", "disconnected", "error", "expired", name="pos_connection_status", create_type=False), nullable=False, server_default="pending"),
        sa.Column("merchant_id", sa.String(255), nullable=True),
        sa.Column("location_id", sa.String(255), nullable=True),
        sa.Column("credentials_encrypted", sa.Text(), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(), nullable=True),
        sa.Column("token_refreshed_at", sa.DateTime(), nullable=True),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("last_catalog_sync_at", sa.DateTime(), nullable=True),
        sa.Column("catalog_sync_status", sa.String(50), nullable=True),
        sa.Column("last_error_at", sa.DateTime(), nullable=True),
        sa.Column("last_error_message", sa.Text(), nullable=True),
        sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_pos_connections"),
        sa.UniqueConstraint("restaurant_id", name="uq_pos_connections_restaurant_id"),
        sa.ForeignKeyConstraint(["restaurant_id"], ["restaurants.id"], name="fk_pos_connections_restaurant_id_restaurants", ondelete="CASCADE"),
    )
    op.create_index("ix_pos_connections_restaurant_id", "pos_connections", ["restaurant_id"])
    op.create_index("ix_pos_connections_provider", "pos_connections", ["provider"])
    op.create_index("ix_pos_connections_status", "pos_connections", ["status"])

    # =========================================================================
    # MENU ITEMS TABLE
    # =========================================================================
    op.create_table(
        "menu_items",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("restaurant_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("price", sa.Numeric(10, 2), nullable=False),
        sa.Column("price_per_person", sa.Numeric(10, 2), nullable=True),
        sa.Column("serves_min", sa.Integer(), nullable=True),
        sa.Column("serves_max", sa.Integer(), nullable=True),
        sa.Column("is_available", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("is_featured", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("dietary_labels", postgresql.ARRAY(sa.String(50)), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String(50)), nullable=True),
        sa.Column("image_url", sa.String(2048), nullable=True),
        sa.Column("pos_sku_mapping", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("modifiers", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pos_synced_at", sa.String(50), nullable=True),
        sa.Column("pos_item_hash", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_menu_items"),
        sa.ForeignKeyConstraint(["restaurant_id"], ["restaurants.id"], name="fk_menu_items_restaurant_id_restaurants", ondelete="CASCADE"),
    )
    op.create_index("ix_menu_items_restaurant_id", "menu_items", ["restaurant_id"])
    op.create_index("ix_menu_items_restaurant_category", "menu_items", ["restaurant_id", "category"])
    op.create_index("ix_menu_items_available", "menu_items", ["restaurant_id", "is_available"])

    # =========================================================================
    # ORDERS TABLE
    # =========================================================================
    op.create_table(
        "orders",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("restaurant_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("customer_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("order_number", sa.String(50), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("status", postgresql.ENUM("created", "paid", "sent_to_pos", "accepted", "preparing", "ready", "completed", "failed", "canceled", "refunded", name="order_status", create_type=False), nullable=False, server_default="created"),
        sa.Column("fulfillment_type", postgresql.ENUM("pickup", "delivery", name="fulfillment_type", create_type=False), nullable=False, server_default="pickup"),
        sa.Column("scheduled_for", sa.DateTime(), nullable=True),
        sa.Column("delivery_address", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("customer_name", sa.String(255), nullable=True),
        sa.Column("customer_email", sa.String(255), nullable=True),
        sa.Column("customer_phone", sa.String(20), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("subtotal_cents", sa.Integer(), nullable=False),
        sa.Column("tax_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tip_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("delivery_fee_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("discount_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_cents", sa.Integer(), nullable=False),
        sa.Column("application_fee_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pos_order_id", sa.String(255), nullable=True),
        sa.Column("pos_sent_at", sa.DateTime(), nullable=True),
        sa.Column("pos_accepted_at", sa.DateTime(), nullable=True),
        sa.Column("pos_error_message", sa.Text(), nullable=True),
        sa.Column("pos_retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("paid_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("canceled_at", sa.DateTime(), nullable=True),
        sa.Column("canceled_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_orders"),
        sa.UniqueConstraint("idempotency_key", name="uq_orders_idempotency_key"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_orders_tenant_id_tenants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["restaurant_id"], ["restaurants.id"], name="fk_orders_restaurant_id_restaurants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["customer_id"], ["users.id"], name="fk_orders_customer_id_users", ondelete="SET NULL"),
    )
    op.create_index("ix_orders_tenant_id", "orders", ["tenant_id"])
    op.create_index("ix_orders_restaurant_id", "orders", ["restaurant_id"])
    op.create_index("ix_orders_customer_id", "orders", ["customer_id"])
    op.create_index("ix_orders_status", "orders", ["status"])
    op.create_index("ix_orders_tenant_number", "orders", ["tenant_id", "order_number"])
    op.create_index("ix_orders_created_at", "orders", ["created_at"])

    # =========================================================================
    # ORDER ITEMS TABLE
    # =========================================================================
    op.create_table(
        "order_items",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("order_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("menu_item_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("item_name", sa.String(255), nullable=False),
        sa.Column("item_description", sa.Text(), nullable=True),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("unit_price_cents", sa.Integer(), nullable=False),
        sa.Column("modifiers_price_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_cents", sa.Integer(), nullable=False),
        sa.Column("modifiers", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("pos_sku", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_order_items"),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], name="fk_order_items_order_id_orders", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["menu_item_id"], ["menu_items.id"], name="fk_order_items_menu_item_id_menu_items", ondelete="SET NULL"),
    )
    op.create_index("ix_order_items_order_id", "order_items", ["order_id"])
    op.create_index("ix_order_items_menu_item_id", "order_items", ["menu_item_id"])

    # =========================================================================
    # PAYMENT INTENTS TABLE
    # =========================================================================
    op.create_table(
        "payment_intents",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("order_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("stripe_payment_intent_id", sa.String(255), nullable=False),
        sa.Column("stripe_account_id", sa.String(255), nullable=False),
        sa.Column("status", postgresql.ENUM("requires_payment_method", "requires_confirmation", "requires_action", "processing", "succeeded", "canceled", "requires_capture", name="payment_status", create_type=False), nullable=False, server_default="requires_payment_method"),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("application_fee_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("currency", sa.String(3), nullable=False, server_default="usd"),
        sa.Column("payment_method_type", sa.String(50), nullable=True),
        sa.Column("last_four", sa.String(4), nullable=True),
        sa.Column("card_brand", sa.String(20), nullable=True),
        sa.Column("client_secret", sa.String(255), nullable=True),
        sa.Column("confirmed_at", sa.DateTime(), nullable=True),
        sa.Column("canceled_at", sa.DateTime(), nullable=True),
        sa.Column("last_error_code", sa.String(100), nullable=True),
        sa.Column("last_error_message", sa.Text(), nullable=True),
        sa.Column("stripe_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_payment_intents"),
        sa.UniqueConstraint("order_id", name="uq_payment_intents_order_id"),
        sa.UniqueConstraint("stripe_payment_intent_id", name="uq_payment_intents_stripe_payment_intent_id"),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], name="fk_payment_intents_order_id_orders", ondelete="CASCADE"),
    )
    op.create_index("ix_payment_intents_order_id", "payment_intents", ["order_id"])
    op.create_index("ix_payment_intents_stripe_id", "payment_intents", ["stripe_payment_intent_id"])
    op.create_index("ix_payment_intents_status", "payment_intents", ["status"])

    # =========================================================================
    # REFUNDS TABLE
    # =========================================================================
    op.create_table(
        "refunds",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("order_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("stripe_refund_id", sa.String(255), nullable=False),
        sa.Column("stripe_payment_intent_id", sa.String(255), nullable=False),
        sa.Column("status", postgresql.ENUM("pending", "succeeded", "failed", "canceled", name="refund_status", create_type=False), nullable=False, server_default="pending"),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="usd"),
        sa.Column("reason", postgresql.ENUM("customer_request", "restaurant_canceled", "item_unavailable", "quality_issue", "late_delivery", "wrong_order", "duplicate", "fraudulent", "other", name="refund_reason", create_type=False), nullable=False),
        sa.Column("reason_details", sa.Text(), nullable=True),
        sa.Column("initiated_by_user_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_refunds"),
        sa.UniqueConstraint("stripe_refund_id", name="uq_refunds_stripe_refund_id"),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], name="fk_refunds_order_id_orders", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["initiated_by_user_id"], ["users.id"], name="fk_refunds_initiated_by_user_id_users", ondelete="SET NULL"),
    )
    op.create_index("ix_refunds_order_id", "refunds", ["order_id"])
    op.create_index("ix_refunds_stripe_refund_id", "refunds", ["stripe_refund_id"])
    op.create_index("ix_refunds_status", "refunds", ["status"])

    # =========================================================================
    # WEBHOOK EVENTS TABLE
    # =========================================================================
    op.create_table(
        "webhook_events",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("provider", postgresql.ENUM("stripe", "square", "toast", "olo", "omnivore", name="webhook_provider", create_type=False), nullable=False),
        sa.Column("provider_event_id", sa.String(255), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("status", postgresql.ENUM("received", "processing", "processed", "failed", "ignored", name="webhook_event_status", create_type=False), nullable=False, server_default="received"),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("signature_valid", sa.Boolean(), nullable=True),
        sa.Column("signature_header", sa.String(500), nullable=True),
        sa.Column("processing_started_at", sa.DateTime(), nullable=True),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_retry_at", sa.DateTime(), nullable=True),
        sa.Column("related_order_id", sa.String(255), nullable=True),
        sa.Column("related_payment_intent_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_webhook_events"),
    )
    op.create_index("ix_webhook_events_provider_event", "webhook_events", ["provider", "provider_event_id"], unique=True)
    op.create_index("ix_webhook_events_status", "webhook_events", ["status"])
    op.create_index("ix_webhook_events_event_type", "webhook_events", ["event_type"])
    op.create_index("ix_webhook_events_next_retry", "webhook_events", ["next_retry_at"])

    # =========================================================================
    # ROW-LEVEL SECURITY POLICIES
    # =========================================================================

    # Enable RLS on tenant-scoped tables
    op.execute("ALTER TABLE users ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE restaurants ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE stripe_accounts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE pos_connections ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE menu_items ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE orders ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE order_items ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE payment_intents ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE refunds ENABLE ROW LEVEL SECURITY")

    # Users policy: users can see their own tenant's users
    op.execute("""
        CREATE POLICY users_tenant_isolation ON users
        FOR ALL
        USING (
            tenant_id IS NULL  -- Platform admins
            OR tenant_id::text = current_setting('app.tenant_id', true)
        )
        WITH CHECK (
            tenant_id IS NULL
            OR tenant_id::text = current_setting('app.tenant_id', true)
        )
    """)

    # Restaurants policy: tenant isolation
    op.execute("""
        CREATE POLICY restaurants_tenant_isolation ON restaurants
        FOR ALL
        USING (tenant_id::text = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true))
    """)

    # Stripe accounts policy: via restaurant tenant
    op.execute("""
        CREATE POLICY stripe_accounts_tenant_isolation ON stripe_accounts
        FOR ALL
        USING (
            restaurant_id IN (
                SELECT id FROM restaurants
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
        WITH CHECK (
            restaurant_id IN (
                SELECT id FROM restaurants
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
    """)

    # POS connections policy: via restaurant tenant
    op.execute("""
        CREATE POLICY pos_connections_tenant_isolation ON pos_connections
        FOR ALL
        USING (
            restaurant_id IN (
                SELECT id FROM restaurants
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
        WITH CHECK (
            restaurant_id IN (
                SELECT id FROM restaurants
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
    """)

    # Menu items policy: via restaurant tenant
    op.execute("""
        CREATE POLICY menu_items_tenant_isolation ON menu_items
        FOR ALL
        USING (
            restaurant_id IN (
                SELECT id FROM restaurants
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
        WITH CHECK (
            restaurant_id IN (
                SELECT id FROM restaurants
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
    """)

    # Orders policy: direct tenant isolation
    op.execute("""
        CREATE POLICY orders_tenant_isolation ON orders
        FOR ALL
        USING (tenant_id::text = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true))
    """)

    # Order items policy: via order tenant
    op.execute("""
        CREATE POLICY order_items_tenant_isolation ON order_items
        FOR ALL
        USING (
            order_id IN (
                SELECT id FROM orders
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
        WITH CHECK (
            order_id IN (
                SELECT id FROM orders
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
    """)

    # Payment intents policy: via order tenant
    op.execute("""
        CREATE POLICY payment_intents_tenant_isolation ON payment_intents
        FOR ALL
        USING (
            order_id IN (
                SELECT id FROM orders
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
        WITH CHECK (
            order_id IN (
                SELECT id FROM orders
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
    """)

    # Refunds policy: via order tenant
    op.execute("""
        CREATE POLICY refunds_tenant_isolation ON refunds
        FOR ALL
        USING (
            order_id IN (
                SELECT id FROM orders
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
        WITH CHECK (
            order_id IN (
                SELECT id FROM orders
                WHERE tenant_id::text = current_setting('app.tenant_id', true)
            )
        )
    """)

    # =========================================================================
    # UPDATED_AT TRIGGER FUNCTION
    # =========================================================================
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    # Apply updated_at trigger to all tables
    for table in [
        "tenants", "users", "restaurants", "stripe_accounts", "pos_connections",
        "menu_items", "orders", "order_items", "payment_intents", "refunds",
        "webhook_events"
    ]:
        op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """)


def downgrade() -> None:
    # Drop triggers
    for table in [
        "tenants", "users", "restaurants", "stripe_accounts", "pos_connections",
        "menu_items", "orders", "order_items", "payment_intents", "refunds",
        "webhook_events"
    ]:
        op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")

    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")

    # Drop RLS policies
    op.execute("DROP POLICY IF EXISTS refunds_tenant_isolation ON refunds")
    op.execute("DROP POLICY IF EXISTS payment_intents_tenant_isolation ON payment_intents")
    op.execute("DROP POLICY IF EXISTS order_items_tenant_isolation ON order_items")
    op.execute("DROP POLICY IF EXISTS orders_tenant_isolation ON orders")
    op.execute("DROP POLICY IF EXISTS menu_items_tenant_isolation ON menu_items")
    op.execute("DROP POLICY IF EXISTS pos_connections_tenant_isolation ON pos_connections")
    op.execute("DROP POLICY IF EXISTS stripe_accounts_tenant_isolation ON stripe_accounts")
    op.execute("DROP POLICY IF EXISTS restaurants_tenant_isolation ON restaurants")
    op.execute("DROP POLICY IF EXISTS users_tenant_isolation ON users")

    # Drop tables in reverse order
    op.drop_table("webhook_events")
    op.drop_table("refunds")
    op.drop_table("payment_intents")
    op.drop_table("order_items")
    op.drop_table("orders")
    op.drop_table("menu_items")
    op.drop_table("pos_connections")
    op.drop_table("stripe_accounts")

    # Drop FK from users to restaurants before dropping restaurants
    op.drop_constraint("fk_users_restaurant_id_restaurants", "users", type_="foreignkey")

    op.drop_table("restaurants")
    op.drop_table("users")
    op.drop_table("tenants")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS webhook_event_status")
    op.execute("DROP TYPE IF EXISTS webhook_provider")
    op.execute("DROP TYPE IF EXISTS refund_reason")
    op.execute("DROP TYPE IF EXISTS refund_status")
    op.execute("DROP TYPE IF EXISTS payment_status")
    op.execute("DROP TYPE IF EXISTS fulfillment_type")
    op.execute("DROP TYPE IF EXISTS order_status")
    op.execute("DROP TYPE IF EXISTS pos_connection_status")
    op.execute("DROP TYPE IF EXISTS pos_provider")
    op.execute("DROP TYPE IF EXISTS stripe_account_status")
    op.execute("DROP TYPE IF EXISTS user_role")
