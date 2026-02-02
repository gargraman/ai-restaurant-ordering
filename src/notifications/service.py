"""Notification service for sending emails and SMS messages.

This module provides a centralized service for sending notifications
to customers and restaurants via email and SMS.
"""

import asyncio
import smtplib
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
import structlog
from dataclasses import dataclass

import httpx
from twilio.rest import Client

from src.config import get_settings


logger = structlog.get_logger(__name__)


@dataclass
class NotificationMessage:
    """Data class for notification message."""
    
    subject: str
    body: str
    recipients: List[str]  # List of email addresses or phone numbers
    message_type: str = "email"  # "email" or "sms"
    html_body: Optional[str] = None


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""
    
    @abstractmethod
    async def send(self, message: NotificationMessage) -> Dict[str, Any]:
        """Send a notification message.
        
        Args:
            message: NotificationMessage to send
            
        Returns:
            Dictionary with send results
        """
        pass


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider using SMTP."""
    
    def __init__(self):
        settings = get_settings()
        self.smtp_server = settings.smtp_server
        self.smtp_port = settings.smtp_port
        self.smtp_username = settings.smtp_username
        self.smtp_password = settings.smtp_password
        self.sender_email = settings.sender_email
        
    async def send(self, message: NotificationMessage) -> Dict[str, Any]:
        """Send an email notification.
        
        Args:
            message: NotificationMessage to send
            
        Returns:
            Dictionary with send results
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['Subject'] = message.subject
            
            # Add text body
            if message.html_body:
                msg.attach(MIMEText(message.html_body, 'html'))
            else:
                msg.attach(MIMEText(message.body, 'plain'))
            
            results = []
            for recipient in message.recipients:
                msg['To'] = recipient
                
                # Create a copy of the message for each recipient
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                
                server.send_message(msg)
                server.quit()
                
                results.append({
                    'recipient': recipient,
                    'status': 'sent',
                    'provider': 'smtp'
                })
                
            return {
                'status': 'success',
                'results': results,
                'message_type': 'email'
            }
            
        except Exception as e:
            logger.error("Failed to send email notification", error=str(e))
            return {
                'status': 'error',
                'error': str(e),
                'message_type': 'email'
            }


class SMSNotificationProvider(NotificationProvider):
    """SMS notification provider using Twilio."""
    
    def __init__(self):
        settings = get_settings()
        self.twilio_account_sid = settings.twilio_account_sid
        self.twilio_auth_token = settings.twilio_auth_token
        self.twilio_phone_number = settings.twilio_phone_number
        
        if self.twilio_account_sid and self.twilio_auth_token:
            self.client = Client(self.twilio_account_sid, self.twilio_auth_token)
        else:
            self.client = None
    
    async def send(self, message: NotificationMessage) -> Dict[str, Any]:
        """Send an SMS notification.
        
        Args:
            message: NotificationMessage to send
            
        Returns:
            Dictionary with send results
        """
        if not self.client:
            logger.warning("Twilio not configured, skipping SMS notification")
            return {
                'status': 'skipped',
                'error': 'Twilio not configured',
                'message_type': 'sms'
            }
        
        try:
            results = []
            for recipient in message.recipients:
                # Ensure phone number is in correct format
                if not recipient.startswith('+'):
                    # Assume US number if no country code
                    recipient = f"+1{recipient}" if len(recipient) == 10 else recipient
                
                message_obj = self.client.messages.create(
                    body=message.body,
                    from_=self.twilio_phone_number,
                    to=recipient
                )
                
                results.append({
                    'recipient': recipient,
                    'status': 'sent',
                    'message_sid': message_obj.sid,
                    'provider': 'twilio'
                })
                
            return {
                'status': 'success',
                'results': results,
                'message_type': 'sms'
            }
            
        except Exception as e:
            logger.error("Failed to send SMS notification", error=str(e))
            return {
                'status': 'error',
                'error': str(e),
                'message_type': 'sms'
            }


class MockEmailProvider(NotificationProvider):
    """Mock email provider for development/testing."""
    
    async def send(self, message: NotificationMessage) -> Dict[str, Any]:
        """Mock send email - just logs the message.
        
        Args:
            message: NotificationMessage to send
            
        Returns:
            Dictionary with mock results
        """
        logger.info(
            "Mock email sent",
            subject=message.subject,
            recipients=message.recipients,
            body=message.body
        )
        
        return {
            'status': 'success',
            'results': [{'recipient': r, 'status': 'mock_sent'} for r in message.recipients],
            'message_type': 'email'
        }


class MockSMSProvider(NotificationProvider):
    """Mock SMS provider for development/testing."""
    
    async def send(self, message: NotificationMessage) -> Dict[str, Any]:
        """Mock send SMS - just logs the message.
        
        Args:
            message: NotificationMessage to send
            
        Returns:
            Dictionary with mock results
        """
        logger.info(
            "Mock SMS sent",
            recipients=message.recipients,
            body=message.body
        )
        
        return {
            'status': 'success',
            'results': [{'recipient': r, 'status': 'mock_sent'} for r in message.recipients],
            'message_type': 'sms'
        }


class NotificationService:
    """Main notification service that orchestrates sending notifications."""
    
    def __init__(self):
        settings = get_settings()
        
        # Initialize providers based on environment
        if settings.app_env == "development":
            self.email_provider = MockEmailProvider()
            self.sms_provider = MockSMSProvider()
        else:
            self.email_provider = EmailNotificationProvider()
            self.sms_provider = SMSNotificationProvider()
    
    async def send_email(
        self,
        subject: str,
        body: str,
        recipients: List[str],
        html_body: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send an email notification.
        
        Args:
            subject: Email subject
            body: Plain text email body
            recipients: List of email addresses
            html_body: Optional HTML email body
            
        Returns:
            Dictionary with send results
        """
        message = NotificationMessage(
            subject=subject,
            body=body,
            recipients=recipients,
            message_type='email',
            html_body=html_body
        )
        
        return await self.email_provider.send(message)
    
    async def send_sms(
        self,
        body: str,
        recipients: List[str]
    ) -> Dict[str, Any]:
        """Send an SMS notification.
        
        Args:
            body: SMS message body
            recipients: List of phone numbers
            
        Returns:
            Dictionary with send results
        """
        message = NotificationMessage(
            subject="",  # SMS doesn't have subjects
            body=body,
            recipients=recipients,
            message_type='sms'
        )
        
        return await self.sms_provider.send(message)
    
    async def send_customer_notification(
        self,
        customer_email: Optional[str],
        customer_phone: Optional[str],
        subject: str,
        email_body: str,
        sms_body: str,
        html_body: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send notification to customer via email and/or SMS.
        
        Args:
            customer_email: Customer's email address
            customer_phone: Customer's phone number
            subject: Email subject
            email_body: Email message body
            sms_body: SMS message body
            html_body: Optional HTML email body
            
        Returns:
            Dictionary with results from all notification attempts
        """
        results = {}
        
        if customer_email:
            results['email'] = await self.send_email(
                subject=subject,
                body=email_body,
                recipients=[customer_email],
                html_body=html_body
            )
        
        if customer_phone:
            results['sms'] = await self.send_sms(
                body=sms_body,
                recipients=[customer_phone]
            )
        
        return results
    
    async def send_restaurant_notification(
        self,
        restaurant_email: Optional[str],
        restaurant_phone: Optional[str],
        subject: str,
        email_body: str,
        sms_body: str,
        html_body: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send notification to restaurant via email and/or SMS.
        
        Args:
            restaurant_email: Restaurant's email address
            restaurant_phone: Restaurant's phone number
            subject: Email subject
            email_body: Email message body
            sms_body: SMS message body
            html_body: Optional HTML email body
            
        Returns:
            Dictionary with results from all notification attempts
        """
        results = {}
        
        if restaurant_email:
            results['email'] = await self.send_email(
                subject=subject,
                body=email_body,
                recipients=[restaurant_email],
                html_body=html_body
            )
        
        if restaurant_phone:
            results['sms'] = await self.send_sms(
                body=sms_body,
                recipients=[restaurant_phone]
            )
        
        return results


# Global instance for convenience
_notification_service = None


def get_notification_service() -> NotificationService:
    """Get the global notification service instance.
    
    Returns:
        NotificationService instance
    """
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


# Convenience functions for common notification types
async def send_order_confirmation_notification(
    customer_email: Optional[str],
    customer_phone: Optional[str],
    order_number: str,
    restaurant_name: str,
    estimated_time: str
) -> Dict[str, Any]:
    """Send order confirmation notification to customer.
    
    Args:
        customer_email: Customer's email
        customer_phone: Customer's phone
        order_number: Order number
        restaurant_name: Name of the restaurant
        estimated_time: Estimated preparation time
        
    Returns:
        Dictionary with notification results
    """
    service = get_notification_service()
    
    email_subject = f"Order Confirmation - #{order_number}"
    email_body = f"""
    Your order #{order_number} has been confirmed!
    
    Restaurant: {restaurant_name}
    Estimated prep time: {estimated_time}
    
    Thank you for your order.
    """
    
    sms_body = f"Order #{order_number} confirmed at {restaurant_name}. ETA: {estimated_time}"
    
    return await service.send_customer_notification(
        customer_email=customer_email,
        customer_phone=customer_phone,
        subject=email_subject,
        email_body=email_body,
        sms_body=sms_body
    )


async def send_order_status_notification(
    customer_email: Optional[str],
    customer_phone: Optional[str],
    order_number: str,
    status: str,
    restaurant_name: str
) -> Dict[str, Any]:
    """Send order status update notification to customer.
    
    Args:
        customer_email: Customer's email
        customer_phone: Customer's phone
        order_number: Order number
        status: New order status
        restaurant_name: Name of the restaurant
        
    Returns:
        Dictionary with notification results
    """
    service = get_notification_service()
    
    email_subject = f"Order Update - #{order_number}"
    email_body = f"""
    Your order #{order_number} status has been updated to: {status}
    
    Restaurant: {restaurant_name}
    
    Thank you for your patience.
    """
    
    sms_body = f"Order #{order_number} status: {status} at {restaurant_name}"
    
    return await service.send_customer_notification(
        customer_email=customer_email,
        customer_phone=customer_phone,
        subject=email_subject,
        email_body=email_body,
        sms_body=sms_body
    )


async def send_order_failure_notification(
    customer_email: Optional[str],
    customer_phone: Optional[str],
    order_number: str,
    error_message: str
) -> Dict[str, Any]:
    """Send order failure notification to customer.
    
    Args:
        customer_email: Customer's email
        customer_phone: Customer's phone
        order_number: Order number
        error_message: Error message
        
    Returns:
        Dictionary with notification results
    """
    service = get_notification_service()
    
    email_subject = f"Order Issue - #{order_number}"
    email_body = f"""
    We encountered an issue with your order #{order_number}.
    
    Error: {error_message}
    
    Our team has been notified and will contact you shortly.
    """
    
    sms_body = f"Issue with order #{order_number}: {error_message}. We'll contact you."
    
    return await service.send_customer_notification(
        customer_email=customer_email,
        customer_phone=customer_phone,
        subject=email_subject,
        email_body=email_body,
        sms_body=sms_body
    )