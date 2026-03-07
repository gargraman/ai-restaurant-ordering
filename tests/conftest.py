"""Global test fixtures and configuration."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from uuid import uuid4
from jose import jwt

# Import get_db for dependency override
from src.db import get_db

# Set test environment variables before importing application code
os.environ["JWT_SECRET_KEY"] = "test_secret_key_32_chars_long_for_hs256"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["ACCESS_TOKEN_EXPIRE_HOURS"] = "24"
os.environ["REFRESH_TOKEN_EXPIRE_DAYS"] = "7"
os.environ["STRIPE_SECRET_KEY"] = "sk_test_123456789"
os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_test123456789"
os.environ["OPENTELEMETRY_ENABLED"] = "false"


@pytest.fixture(scope="session", autouse=True)
def setup_jwt_service():
    """Initialize JWT service for all tests."""
    from src.auth.jwt import init_jwt_service
    init_jwt_service(
        secret_key="test_secret_key_32_chars_long_for_hs256",
        algorithm="HS256",
        access_token_expire_hours=24,
        refresh_token_expire_days=7,
    )


@pytest.fixture(scope="function", autouse=True)
def reset_jwt_service():
    """Reset JWT service before each test to ensure clean state."""
    from src.auth import jwt as jwt_module
    jwt_module._jwt_service = None
    init_jwt_service = jwt_module.init_jwt_service
    init_jwt_service(
        secret_key="test_secret_key_32_chars_long_for_hs256",
        algorithm="HS256",
        access_token_expire_hours=24,
        refresh_token_expire_days=7,
    )
    yield
    jwt_module._jwt_service = None


@pytest.fixture
def mock_db_session():
    """Create a mock database session for testing."""
    session = AsyncMock()

    # Mock execute to return async results
    async def mock_execute(*args, **kwargs):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_result.scalars = MagicMock()
        mock_result.scalars.return_value = MagicMock()
        mock_result.scalars.return_value.all = MagicMock(return_value=[])
        mock_result.one_or_none = MagicMock(return_value=None)
        mock_result.one = MagicMock(return_value=None)
        mock_result.all = MagicMock(return_value=[])
        mock_result.first = MagicMock(return_value=None)
        return mock_result

    def mock_add(obj):
        """Simulate DB behavior by assigning IDs and timestamps on add."""
        if hasattr(obj, 'id') and obj.id is None:
            obj.id = str(uuid4())
        if hasattr(obj, 'created_at') and obj.created_at is None:
            obj.created_at = datetime.utcnow()
        if hasattr(obj, 'updated_at') and obj.updated_at is None:
            obj.updated_at = datetime.utcnow()

    session.execute = mock_execute
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.flush = AsyncMock()
    session.add = mock_add

    return session


@pytest.fixture
def db_session(mock_db_session):
    """Alias for mock_db_session for compatibility with tests expecting db_session."""
    return mock_db_session


@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    from src.db.models.user import User
    user = User(
        id=1,
        email="test@example.com",
        password_hash="hashed_password",
        is_active=True,
        role="customer",
        tenant_id="tenant-123",
        first_name="Test",
        last_name="User",
    )
    return user


@pytest.fixture
def sample_admin_user():
    """Create a sample admin user for testing."""
    from src.db.models.user import User
    user = User(
        id=2,
        email="admin@example.com",
        password_hash="hashed_password",
        is_active=True,
        role="admin",
        tenant_id="tenant-123",
        first_name="Admin",
        last_name="User",
    )
    return user


@pytest.fixture
def valid_token(sample_user):
    """Create a valid JWT token for testing."""
    data = {
        "sub": str(sample_user.id),
        "email": sample_user.email,
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "tenant_id": sample_user.tenant_id,
    }
    return jwt.encode(data, "test_secret_key_32_chars_long_for_hs256", algorithm="HS256")


@pytest.fixture
def expired_token(sample_user):
    """Create an expired JWT token for testing."""
    data = {
        "sub": str(sample_user.id),
        "email": sample_user.email,
        "exp": datetime.utcnow() - timedelta(hours=1),
        "tenant_id": sample_user.tenant_id,
    }
    return jwt.encode(data, "test_secret_key_32_chars_long_for_hs256", algorithm="HS256")


@pytest.fixture
def invalid_token():
    """Return an invalid token string."""
    return "invalid_token_string"


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for testing."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hset = AsyncMock()
    redis.hdel = AsyncMock()
    
    # Mock Redis JSON commands
    json_mock = MagicMock()
    json_mock.get = AsyncMock(return_value=None)
    json_mock.set = AsyncMock()
    redis.json = MagicMock(return_value=json_mock)
    
    return redis


@pytest.fixture
def mock_stripe():
    """Mock Stripe API calls."""
    with patch("stripe.PaymentIntent.create") as mock_create, \
         patch("stripe.PaymentIntent.retrieve") as mock_retrieve, \
         patch("stripe.PaymentIntent.modify") as mock_modify, \
         patch("stripe.Webhook.construct_event") as mock_construct:
        
        mock_create.return_value = MagicMock(id="pi_test_123", client_secret="secret_123")
        mock_retrieve.return_value = MagicMock(id="pi_test_123", status="succeeded")
        mock_modify.return_value = MagicMock(id="pi_test_123")
        mock_construct.return_value = MagicMock(id="evt_test_123", type="payment_intent.succeeded")
        
        yield {
            "create": mock_create,
            "retrieve": mock_retrieve,
            "modify": mock_modify,
            "construct": mock_construct,
        }


@pytest.fixture
def mock_opensearch():
    """Mock OpenSearch client."""
    with patch("src.search.bm25.OpenSearchClient") as mock_client:
        client = MagicMock()
        client.search = AsyncMock(return_value={"hits": {"hits": []}})
        mock_client.return_value = client
        yield client


@pytest.fixture
def mock_pgvector():
    """Mock pgvector client."""
    with patch("src.search.vector.VectorSearchClient") as mock_client:
        client = MagicMock()
        client.similarity_search = AsyncMock(return_value=[])
        mock_client.return_value = client
        yield client


@pytest.fixture
def mock_pos_adapter():
    """Mock POS adapter."""
    adapter = AsyncMock()
    adapter.is_available = AsyncMock(return_value=True)
    adapter.send_order = AsyncMock(return_value={"status": "success", "order_id": "pos_123"})
    adapter.get_status = AsyncMock(return_value={"status": "connected"})
    return adapter


@pytest.fixture
def make_auth_token():
    """Factory fixture to create valid JWT tokens for testing authentication."""
    def _make_token(
        user_id: str = None,
        email: str = "test@example.com",
        role: str = "CUSTOMER",
        tenant_id: str = None,
        restaurant_id: str = None,
    ):
        from src.auth.jwt import get_jwt_service
        from src.auth.models import UserRole
        user_id = user_id or str(uuid4())
        tenant_id = tenant_id or str(uuid4())
        role_enum = UserRole(role.lower())
        jwt_service = get_jwt_service()
        access_token, refresh_token, expires_in = jwt_service.create_token_pair(
            user_id=user_id,
            email=email,
            role=role_enum,
            tenant_id=tenant_id,
            restaurant_id=restaurant_id,
        )
        return access_token, refresh_token, expires_in, user_id, tenant_id
    return _make_token


@pytest.fixture
def create_test_token():
    """Factory fixture to create JWT tokens with custom claims."""
    def _create_token(
        user_id: int = 1,
        email: str = "test@example.com",
        tenant_id: str = "tenant-123",
        expires_in_minutes: int = 30,
        secret_key: str = "test_secret_key_32_chars_long_for_hs256",
    ):
        data = {
            "sub": str(user_id),
            "email": email,
            "exp": datetime.utcnow() + timedelta(minutes=expires_in_minutes),
            "tenant_id": tenant_id,
        }
        return jwt.encode(data, secret_key, algorithm="HS256")
    return _create_token


@pytest.fixture
def client(mock_db_session):
    """Create a test client for the FastAPI app with mocked dependencies."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    # Override the get_db dependency
    async def override_get_db():
        yield mock_db_session

    app.dependency_overrides = {}
    app.dependency_overrides[get_db] = override_get_db

    test_client = TestClient(app)
    yield test_client

    # Clean up overrides
    app.dependency_overrides.clear()
