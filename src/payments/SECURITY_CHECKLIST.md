# Stripe Connect Security & PCI Compliance Checklist

This checklist ensures secure payment processing and PCI DSS compliance for the multi-restaurant platform.

## PCI Compliance Requirements

### 1. Card Data Handling

- [ ] **NEVER store card numbers, CVV, or magnetic stripe data**
  - Use Stripe.js/Elements to collect payment details client-side
  - Card data goes directly to Stripe, never touches your servers
  - Use `client_secret` for frontend payment confirmation

- [ ] **NEVER log sensitive payment information**
  - No card numbers in application logs
  - No CVV codes in any logs or databases
  - Sanitize payment method details before logging

- [ ] **Use Stripe's tokenization**
  - Collect payment methods using Stripe Elements
  - Pass payment method ID (pm_xxx) or token to backend
  - Never pass raw card data to backend

### 2. API Key Security

- [ ] **Protect Stripe API keys**
  - Store in environment variables, never in code
  - Use different keys for test/production environments
  - Rotate keys periodically (quarterly recommended)
  - Never commit keys to version control

- [ ] **Restrict API key permissions**
  - Use restricted keys where possible
  - Limit key access to specific operations
  - Monitor key usage for anomalies

- [ ] **Secure webhook endpoints**
  - ALWAYS verify webhook signatures using `stripe.Webhook.construct_event()`
  - Use HTTPS for webhook endpoints
  - Return 200 status only after successful verification
  - Store webhook signing secret securely

### 3. Data Storage

- [ ] **Store only necessary payment metadata**
  - Payment intent ID (pi_xxx)
  - Refund ID (re_xxx)
  - Last 4 digits of card (from payment_method_details)
  - Card brand (Visa, Mastercard, etc.)
  - Payment status
  - Order association data

- [ ] **NEVER store**
  - Full card number
  - CVV/CVC code
  - Magnetic stripe data
  - PIN codes

- [ ] **Encrypt sensitive data at rest**
  - Use database encryption for payment records
  - Encrypt customer email addresses
  - Use field-level encryption for PII

### 4. Transport Security

- [ ] **Use HTTPS everywhere**
  - All API endpoints must use TLS 1.2+
  - Enforce HTTPS redirects
  - Use HSTS headers

- [ ] **Secure API communication**
  - All Stripe API calls use HTTPS
  - Verify SSL certificates
  - Use certificate pinning in mobile apps

## Stripe Connect Specific Security

### 5. Connected Account Security

- [ ] **Validate connected accounts**
  - Verify `charges_enabled` before creating charges
  - Check `payouts_enabled` for account readiness
  - Validate account ownership (tenant_id in metadata)

- [ ] **Prevent account takeover**
  - Only allow account updates from authenticated restaurant owner
  - Verify email ownership before account creation
  - Implement MFA for restaurant dashboard

- [ ] **Monitor account status**
  - Subscribe to `account.updated` webhooks
  - Alert on account.disabled events
  - Track failed charges per account

### 6. Split Payment Security

- [ ] **Validate application fees**
  - Ensure fee doesn't exceed order total
  - Prevent negative fees
  - Cap maximum fee percentage (sanity check)

- [ ] **Verify transfer destinations**
  - Confirm connected_account_id belongs to tenant
  - Prevent unauthorized fund transfers
  - Log all payment intent creations with account mapping

### 7. Refund Security

- [ ] **Authorize refunds properly**
  - Only restaurant owner can initiate refunds
  - Require authentication for refund requests
  - Log all refund attempts with actor ID

- [ ] **Validate refund amounts**
  - Check refund doesn't exceed original charge
  - Prevent duplicate refunds (use idempotency)
  - Track partial refund totals

## Application Security

### 8. Authentication & Authorization

- [ ] **Secure restaurant authentication**
  - Use JWT or OAuth for API access
  - Implement proper session management
  - Require strong passwords (min 12 chars)

- [ ] **Role-based access control**
  - Restaurant owner: full payment/refund access
  - Staff: limited order/status access
  - Customer: no backend payment access

- [ ] **Multi-factor authentication**
  - Require MFA for restaurant owner accounts
  - Use authenticator apps (TOTP)
  - Backup codes for account recovery

### 9. Idempotency

- [ ] **Use idempotency keys**
  - Generate unique key per payment intent: `pi_{order_id}`
  - Generate unique key per refund: `refund_{order_id}`
  - Store idempotency keys with operations

- [ ] **Handle retries safely**
  - Same idempotency key returns same result
  - Prevent duplicate charges on network errors
  - Prevent duplicate refunds

### 10. Error Handling

- [ ] **Sanitize error messages**
  - Don't expose internal IDs to frontend
  - Generic error messages to customers
  - Detailed errors in server logs only

- [ ] **Handle Stripe errors properly**
  - Catch `StripeError` exceptions
  - Log error type and code
  - Return appropriate HTTP status codes

## Monitoring & Incident Response

### 11. Logging & Monitoring

- [ ] **Log all payment operations**
  - Payment intent creation (with order_id, tenant_id)
  - Refund creation
  - Webhook events received
  - Failed transactions

- [ ] **Monitor for fraud**
  - Track high-value transactions
  - Alert on unusual refund patterns
  - Monitor failed payment rates

- [ ] **Set up alerts**
  - Failed webhook deliveries
  - Payment failures > threshold
  - Account disabled events
  - Unusual fee amounts

### 12. Webhook Security

- [ ] **Verify all webhook signatures**
  ```python
  event = stripe.Webhook.construct_event(
      payload, sig_header, webhook_secret
  )
  ```

- [ ] **Use HTTPS for webhook endpoints**
  - Stripe requires HTTPS in production
  - Use valid SSL certificate

- [ ] **Handle webhook retries**
  - Stripe retries failed webhooks
  - Implement idempotent event handlers
  - Return 200 only after successful processing

- [ ] **Secure webhook endpoint**
  - Don't require authentication (Stripe signs requests)
  - Rate limit webhook endpoint
  - Validate event age (prevent replay attacks)

## Incident Response

### 13. Breach Response Plan

- [ ] **Have incident response plan**
  - Document steps for data breach
  - Contact information for security team
  - Stripe contact for security issues

- [ ] **Regular security audits**
  - Quarterly code security review
  - Annual penetration testing
  - Dependency vulnerability scanning

- [ ] **Key rotation procedures**
  - Document API key rotation process
  - Test key rotation in staging
  - Zero-downtime rotation strategy

## Production Checklist

### 14. Pre-Launch Security

- [ ] Use production Stripe keys (not test keys)
- [ ] Enable Stripe Radar for fraud detection
- [ ] Configure webhook endpoints in Stripe Dashboard
- [ ] Set up monitoring and alerting
- [ ] Test webhook signature verification
- [ ] Enable audit logging
- [ ] Configure rate limiting
- [ ] Set up DDoS protection
- [ ] Enable database encryption
- [ ] Configure CORS properly
- [ ] Set secure HTTP headers (CSP, X-Frame-Options)
- [ ] Enable access logs
- [ ] Set up backup/disaster recovery

### 15. Compliance Documentation

- [ ] Document PCI DSS compliance approach
- [ ] Maintain security policies
- [ ] Keep change logs for payment code
- [ ] Document third-party integrations
- [ ] Maintain vendor security assessments

## Code Review Checklist

When reviewing payment-related code:

- [ ] No card data stored or logged
- [ ] Webhook signatures verified
- [ ] Idempotency keys used for mutations
- [ ] Connected account ID validated
- [ ] Application fees validated
- [ ] Refund amounts validated
- [ ] Authentication required for sensitive operations
- [ ] Error messages sanitized
- [ ] HTTPS enforced
- [ ] Stripe errors handled properly

## Resources

- [Stripe Security Best Practices](https://stripe.com/docs/security/guide)
- [PCI DSS Overview](https://stripe.com/docs/security/guide#pci-dss)
- [Stripe Connect Best Practices](https://stripe.com/docs/connect/best-practices)
- [Webhook Security](https://stripe.com/docs/webhooks/best-practices)
