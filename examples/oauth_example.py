"""
Example usage of the OAuth module for external services like Stripe and Square.

This script demonstrates how to use the new OAuth functionality in the auth module.
"""

from src.auth.oauth import (
    OAuthConfig,
    OAuthStateData,
    create_oauth_config,
    get_oauth_provider,
)

def example_usage():
    """Example of how to use the OAuth functionality."""
    
    print("OAuth Module Usage Examples")
    print("=" * 40)
    
    # Example 1: Creating a Square OAuth configuration
    print("\n1. Creating Square OAuth configuration:")
    square_config = create_oauth_config(
        provider_name="square",
        client_id="your_square_application_id",
        client_secret="your_square_application_secret",
        redirect_uri="https://yourapp.com/callback/square",
        scopes=[
            "MERCHANT_PROFILE_READ",
            "ITEMS_READ",
            "ITEMS_WRITE",
            "ORDERS_READ",
            "ORDERS_WRITE",
            "PAYMENTS_READ",
            "PAYMENTS_WRITE",
            "INVENTORY_READ",
        ],
        environment="production",  # or "sandbox"
    )
    print(f"   Square Authorize URL: {square_config.authorize_url}")
    print(f"   Square Token URL: {square_config.token_url}")
    
    # Example 2: Creating a Stripe OAuth configuration
    print("\n2. Creating Stripe OAuth configuration:")
    stripe_config = create_oauth_config(
        provider_name="stripe",
        client_id="your_stripe_client_id",
        client_secret="your_stripe_client_secret",
        redirect_uri="https://yourapp.com/callback/stripe",
        scopes=["read_write"],  # Stripe specific scopes
        environment="production",
    )
    print(f"   Stripe Authorize URL: {stripe_config.authorize_url}")
    print(f"   Stripe Token URL: {stripe_config.token_url}")
    
    # Example 3: Getting an OAuth provider instance
    print("\n3. Getting OAuth provider instances:")
    square_provider = get_oauth_provider("square", square_config)
    stripe_provider = get_oauth_provider("stripe", stripe_config)
    generic_provider = get_oauth_provider("generic", square_config)
    
    print(f"   Square provider type: {type(square_provider).__name__}")
    print(f"   Stripe provider type: {type(stripe_provider).__name__}")
    print(f"   Generic provider type: {type(generic_provider).__name__}")
    
    # Example 4: Creating OAuth state data for CSRF protection
    print("\n4. Creating OAuth state data:")
    state_data = OAuthStateData(
        provider="square",
        restaurant_id="rest_12345",
        tenant_id="tenant_67890",
        user_id="user_abcde",
    )
    print(f"   State data provider: {state_data.provider}")
    print(f"   Restaurant ID: {state_data.restaurant_id}")
    print(f"   Tenant ID: {state_data.tenant_id}")
    
    # Example 5: Generating authorization URL (would normally be done in a web handler)
    print("\n5. Generating authorization URL:")
    # Note: This would typically be done in a web request handler
    # state, state_string = square_provider.get_authorization_url(
    #     state_data=state_data,
    #     state_secret="your_app_secret_for_signing_state",
    # )
    print("   Authorization URL would be generated here in a web handler")
    print("   Includes CSRF protection via signed state parameter")
    
    print("\nOAuth module is ready for use with external services!")
    print("Supported providers:", list(["square", "stripe", "generic"]))


if __name__ == "__main__":
    example_usage()