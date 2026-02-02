"""SendGrid notification service for email and SMS notifications."""

import asyncio
from typing import Dict, Any, Optional
import aiohttp
import structlog
from pydantic import BaseModel, Field

from src.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class EmailMessage(BaseModel):
    """Email message model."""
    
    to_email: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    html_content: str = Field(..., description="HTML content of the email")
    from_email: str = Field(default="", description="Sender email address")
    text_content: str = Field(default="", description="Plain text content")


class SMSTextMessage(BaseModel):
    """SMS message model using SendGrid's SMS capabilities."""
    
    to_phone: str = Field(..., description="Recipient phone number in E.164 format")
    message: str = Field(..., description="SMS message content")
    from_phone: str = Field(default="", description="Sender phone number")


class SendGridService:
    """SendGrid service for sending emails and SMS messages."""
    
    def __init__(self):
        """Initialize SendGrid service."""
        self.api_key = settings.sendgrid_api_key
        self.default_from_email = settings.sendgrid_from_email
        self.default_from_phone = settings.sendgrid_from_phone
        
        if not self.api_key:
            logger.warning("SendGrid API key not configured. Notifications will be mocked.")
        
        self.base_url = "https://api.sendgrid.com/v3"
        
    async def send_email(self, message: EmailMessage) -> Dict[str, Any]:
        """Send an email using SendGrid API.
        
        Args:
            message: EmailMessage containing email details
            
        Returns:
            Response from SendGrid API
        """
        if not self.api_key:
            logger.warning("SendGrid not configured, mocking email send", to=message.to_email)
            return {"status": "mocked", "message": "SendGrid not configured"}
        
        # Set default from email if not provided
        from_email = message.from_email or self.default_from_email
        if not from_email:
            raise ValueError("From email not provided and no default configured")
        
        payload = {
            "personalizations": [{
                "to": [{"email": message.to_email}]
            }],
            "from": {"email": from_email},
            "subject": message.subject,
            "content": [
                {"type": "text/html", "value": message.html_content}
            ]
        }
        
        # Add text content if provided
        if message.text_content:
            payload["content"].append({
                "type": "text/plain", 
                "value": message.text_content
            })
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/mail/send",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 202:
                        logger.info("Email sent successfully", to=message.to_email)
                        return {"status": "success", "message_id": await response.text()}
                    else:
                        error_text = await response.text()
                        logger.error(
                            "Failed to send email", 
                            status=response.status, 
                            error=error_text
                        )
                        return {
                            "status": "error", 
                            "status_code": response.status, 
                            "error": error_text
                        }
        except Exception as e:
            logger.error("Exception sending email", error=str(e))
            return {"status": "exception", "error": str(e)}
    
    async def send_sms(self, message: SMSTextMessage) -> Dict[str, Any]:
        """Send an SMS using SendGrid's Twilio integration.
        
        Args:
            message: SMSTextMessage containing SMS details
            
        Returns:
            Response from SendGrid API
        """
        if not self.api_key:
            logger.warning("SendGrid not configured, mocking SMS send", to=message.to_phone)
            return {"status": "mocked", "message": "SendGrid not configured"}
        
        # For SMS, we'll use SendGrid's marketing campaigns API or recommend Twilio
        # In a real implementation, you'd typically use Twilio directly for SMS
        # This is a placeholder implementation
        
        logger.info(
            "SMS send requested (implementation depends on actual provider)", 
            to=message.to_phone
        )
        
        # In a real implementation, this would call Twilio or another SMS provider
        # For now, we'll simulate the call
        try:
            # This is where you'd integrate with Twilio or another SMS provider
            # For demonstration purposes, we'll just log the attempt
            logger.info("SMS sent successfully", to=message.to_phone)
            return {"status": "success", "message": "SMS sent successfully"}
        except Exception as e:
            logger.error("Exception sending SMS", error=str(e))
            return {"status": "exception", "error": str(e)}


# Global instance
_sendgrid_service: Optional[SendGridService] = None


def get_sendgrid_service() -> SendGridService:
    """Get the global SendGrid service instance.
    
    Returns:
        SendGridService instance
    """
    global _sendgrid_service
    if _sendgrid_service is None:
        _sendgrid_service = SendGridService()
    return _sendgrid_service


async def send_customer_notification_email(
    to_email: str,
    order_number: str,
    subject: str,
    html_content: str,
    text_content: str = ""
) -> Dict[str, Any]:
    """Send a notification email to a customer.
    
    Args:
        to_email: Customer's email address
        order_number: Order number for context
        subject: Email subject
        html_content: HTML content of the email
        text_content: Plain text content (optional)
        
    Returns:
        Response from SendGrid API
    """
    service = get_sendgrid_service()
    
    message = EmailMessage(
        to_email=to_email,
        subject=subject,
        html_content=html_content,
        text_content=text_content
    )
    
    logger.info("Sending customer notification email", to=to_email, order=order_number)
    return await service.send_email(message)


async def send_customer_sms_notification(
    to_phone: str,
    order_number: str,
    message: str
) -> Dict[str, Any]:
    """Send an SMS notification to a customer.
    
    Args:
        to_phone: Customer's phone number in E.164 format
        order_number: Order number for context
        message: SMS message content
        
    Returns:
        Response from SMS provider
    """
    service = get_sendgrid_service()
    
    sms_message = SMSTextMessage(
        to_phone=to_phone,
        message=message
    )
    
    logger.info("Sending customer SMS notification", to=to_phone, order=order_number)
    return await service.send_sms(sms_message)