"""Seed users (platform admin, tenant admins, restaurant admins, customers)."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.password import hash_password
from src.db.models.user import User, UserRole
from seed.constants import (
    DEFAULT_PASSWORD,
    PLATFORM_ADMIN,
    TENANT_USERS,
    RESTAURANT_STAFF,
)
from seed.context import SeedContext


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create all user types with proper tenant/restaurant scoping.

    Returns:
        Number of records created.
    """
    password_hash = hash_password(DEFAULT_PASSWORD)
    count = 0

    # 1. Platform admin (no tenant)
    admin = User(
        id=str(uuid4()),
        email=PLATFORM_ADMIN["email"],
        password_hash=password_hash,
        is_active=True,
        is_verified=True,
        first_name=PLATFORM_ADMIN["first_name"],
        last_name=PLATFORM_ADMIN["last_name"],
        phone=PLATFORM_ADMIN["phone"],
        role=UserRole.PLATFORM_ADMIN,
        tenant_id=None,
        restaurant_id=None,
    )
    session.add(admin)
    ctx.users[admin.email] = {
        "id": admin.id,
        "email": admin.email,
        "role": UserRole.PLATFORM_ADMIN.value,
        "tenant_id": None,
    }
    count += 1

    # 2. Per-tenant: tenant admin + customers
    for tenant_slug, user_data in TENANT_USERS.items():
        tenant = ctx.tenants[tenant_slug]

        # Tenant admin
        ta = User(
            id=str(uuid4()),
            email=user_data["admin"]["email"],
            password_hash=password_hash,
            is_active=True,
            is_verified=True,
            first_name=user_data["admin"]["first_name"],
            last_name=user_data["admin"]["last_name"],
            phone=user_data["admin"]["phone"],
            role=UserRole.TENANT_ADMIN,
            tenant_id=tenant["id"],
            restaurant_id=None,
        )
        session.add(ta)
        ctx.users[ta.email] = {
            "id": ta.id,
            "email": ta.email,
            "role": UserRole.TENANT_ADMIN.value,
            "tenant_id": tenant["id"],
            "tenant_slug": tenant_slug,
        }
        count += 1

        # Customers
        for cust_data in user_data["customers"]:
            cust = User(
                id=str(uuid4()),
                email=cust_data["email"],
                password_hash=password_hash,
                is_active=True,
                is_verified=True,
                first_name=cust_data["first_name"],
                last_name=cust_data["last_name"],
                phone=cust_data["phone"],
                role=UserRole.CUSTOMER,
                tenant_id=tenant["id"],
                restaurant_id=None,
            )
            session.add(cust)
            ctx.users[cust.email] = {
                "id": cust.id,
                "email": cust.email,
                "role": UserRole.CUSTOMER.value,
                "tenant_id": tenant["id"],
                "tenant_slug": tenant_slug,
            }
            count += 1

    # 3. Restaurant admins (linked to specific restaurants)
    for rest_slug, staff_data in RESTAURANT_STAFF.items():
        restaurant = ctx.restaurants.get(rest_slug)
        if not restaurant:
            continue

        staff = User(
            id=str(uuid4()),
            email=staff_data["email"],
            password_hash=password_hash,
            is_active=True,
            is_verified=True,
            first_name=staff_data["first_name"],
            last_name=staff_data["last_name"],
            phone=staff_data["phone"],
            role=UserRole.RESTAURANT_ADMIN,
            tenant_id=restaurant["tenant_id"],
            restaurant_id=restaurant["id"],
        )
        session.add(staff)
        ctx.users[staff.email] = {
            "id": staff.id,
            "email": staff.email,
            "role": UserRole.RESTAURANT_ADMIN.value,
            "tenant_id": restaurant["tenant_id"],
            "tenant_slug": restaurant["tenant_slug"],
            "restaurant_id": restaurant["id"],
            "restaurant_slug": rest_slug,
        }
        count += 1

    await session.flush()
    return count
