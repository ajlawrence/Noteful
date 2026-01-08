# =============================================================================
# Email Alerts
# =============================================================================
# Send email notifications for important events:
#   - Pipeline failures
#   - Anomaly detection
#   - Scheduled reports
#
# Uses Gmail SMTP by default.
# =============================================================================

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# EMAIL CONFIGURATION
# =============================================================================

DEFAULT_SMTP_CONFIG = {
    "server": "smtp.gmail.com",
    "port": 587,
    "use_tls": True,
}


# =============================================================================
# EMAIL ALERTS CLASS
# =============================================================================

class EmailAlerts:
    """
    Send email alerts and notifications.

    Example usage:
        alerts = EmailAlerts(
            smtp_username="your.email@gmail.com",
            smtp_password="your-app-password",
            default_recipient="recipient@example.com"
        )

        # Send a simple alert
        alerts.send_alert(
            subject="Pipeline Failed",
            message="Error fetching tourism data."
        )

        # Send a report
        alerts.send_report(
            subject="Monthly Report",
            report_content="...",
            attachments=["data/charts/dashboard.png"]
        )
    """

    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        default_recipient: Optional[str] = None,
        sender_name: str = "Costa Sol Economy Monitor"
    ):
        """
        Initialize Email Alerts.

        Args:
            smtp_server: SMTP server hostname (default: Gmail)
            smtp_port: SMTP port (default: 587 for TLS)
            smtp_username: SMTP username (or use SMTP_USERNAME env var)
            smtp_password: SMTP password (or use SMTP_PASSWORD env var)
            default_recipient: Default email recipient (or use ALERT_RECIPIENT env var)
            sender_name: Display name for sender
        """
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", DEFAULT_SMTP_CONFIG["server"])
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", DEFAULT_SMTP_CONFIG["port"]))
        self.smtp_username = smtp_username or os.getenv("SMTP_USERNAME")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.default_recipient = default_recipient or os.getenv("ALERT_RECIPIENT")
        self.sender_name = sender_name

        self._validate_config()

        logger.info("EmailAlerts initialized")

    def _validate_config(self) -> None:
        """Validate email configuration."""
        self.is_configured = all([
            self.smtp_username,
            self.smtp_password,
            self.default_recipient
        ])

        if not self.is_configured:
            logger.warning(
                "Email alerts not fully configured. "
                "Set SMTP_USERNAME, SMTP_PASSWORD, and ALERT_RECIPIENT environment variables."
            )

    # =========================================================================
    # ALERT METHODS
    # =========================================================================

    def send_alert(
        self,
        subject: str,
        message: str,
        recipient: Optional[str] = None,
        priority: str = "normal",  # "low", "normal", "high"
        include_timestamp: bool = True
    ) -> bool:
        """
        Send a simple text alert.

        Args:
            subject: Email subject
            message: Alert message body
            recipient: Override default recipient
            priority: Email priority
            include_timestamp: Add timestamp to message

        Returns:
            True if sent successfully, False otherwise

        Example:
            alerts.send_alert(
                subject="[ALERT] Data Fetch Failed",
                message="Failed to fetch tourism data from Dataestur API.",
                priority="high"
            )
        """
        if not self.is_configured:
            logger.warning("Cannot send alert: email not configured")
            return False

        recipient = recipient or self.default_recipient

        # Build message
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"{message}\n\n---\nTimestamp: {timestamp}"

        # Add priority prefix to subject
        if priority == "high":
            subject = f"[URGENT] {subject}"
        elif priority == "low":
            subject = f"[INFO] {subject}"

        return self._send_email(
            recipient=recipient,
            subject=subject,
            body_text=message
        )

    def send_anomaly_alert(
        self,
        indicator: str,
        anomaly_details: Dict[str, Any],
        recipient: Optional[str] = None
    ) -> bool:
        """
        Send an alert about detected anomalies.

        Args:
            indicator: Name of the indicator with anomaly
            anomaly_details: Dictionary with anomaly info
            recipient: Override default recipient

        Returns:
            True if sent successfully

        Example:
            alerts.send_anomaly_alert(
                indicator="tourist_arrivals",
                anomaly_details={
                    "date": "2024-07-01",
                    "value": 50000,
                    "expected": 200000,
                    "deviation": -75.0,
                    "explanation": "Possible data error"
                }
            )
        """
        subject = f"Anomaly Detected: {indicator}"

        message = f"""
Anomaly detected in {indicator} data for Costa del Sol.

Details:
- Date: {anomaly_details.get('date', 'Unknown')}
- Value: {anomaly_details.get('value', 'N/A')}
- Expected: {anomaly_details.get('expected', 'N/A')}
- Deviation: {anomaly_details.get('deviation', 'N/A')}%

Explanation:
{anomaly_details.get('explanation', 'No explanation available')}

Recommended Action:
Please review this data point to verify if it's a real anomaly or a data issue.

---
Costa del Sol Economic Monitor
"""

        return self.send_alert(
            subject=subject,
            message=message.strip(),
            recipient=recipient,
            priority="high"
        )

    def send_pipeline_status(
        self,
        status: str,  # "success", "partial", "failed"
        summary: Dict[str, Any],
        recipient: Optional[str] = None
    ) -> bool:
        """
        Send a summary of pipeline execution.

        Args:
            status: Overall pipeline status
            summary: Dictionary with execution details
            recipient: Override default recipient

        Returns:
            True if sent successfully

        Example:
            alerts.send_pipeline_status(
                status="success",
                summary={
                    "run_time": "2024-01-15 08:00:00",
                    "duration_seconds": 45,
                    "records_fetched": 250,
                    "errors": []
                }
            )
        """
        status_emoji = {
            "success": "✅",
            "partial": "⚠️",
            "failed": "❌"
        }.get(status, "ℹ️")

        subject = f"{status_emoji} Pipeline {status.title()}"

        errors_text = ""
        if summary.get("errors"):
            errors_text = "\n\nErrors:\n" + "\n".join([f"- {e}" for e in summary["errors"]])

        message = f"""
Pipeline Execution Summary
==========================

Status: {status.upper()}
Run Time: {summary.get('run_time', 'Unknown')}
Duration: {summary.get('duration_seconds', 0)} seconds

Data Fetched:
- Records: {summary.get('records_fetched', 0)}
- Sources: {summary.get('sources_processed', 0)}
{errors_text}

---
Costa del Sol Economic Monitor
"""

        priority = "high" if status == "failed" else "normal"

        return self.send_alert(
            subject=subject,
            message=message.strip(),
            recipient=recipient,
            priority=priority
        )

    def send_report(
        self,
        subject: str,
        report_content: str,
        recipient: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        html_format: bool = False
    ) -> bool:
        """
        Send a report with optional attachments.

        Args:
            subject: Email subject
            report_content: Report text content
            recipient: Override default recipient
            attachments: List of file paths to attach
            html_format: If True, report_content is HTML

        Returns:
            True if sent successfully

        Example:
            alerts.send_report(
                subject="Monthly Economic Report - January 2024",
                report_content=report_markdown,
                attachments=["data/charts/dashboard.png"]
            )
        """
        if not self.is_configured:
            logger.warning("Cannot send report: email not configured")
            return False

        recipient = recipient or self.default_recipient

        return self._send_email(
            recipient=recipient,
            subject=subject,
            body_text=None if html_format else report_content,
            body_html=report_content if html_format else None,
            attachments=attachments
        )

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _send_email(
        self,
        recipient: str,
        subject: str,
        body_text: Optional[str] = None,
        body_html: Optional[str] = None,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """
        Internal method to send an email.

        Args:
            recipient: Email recipient
            subject: Email subject
            body_text: Plain text body
            body_html: HTML body
            attachments: List of file paths

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart("mixed")
            msg["From"] = f"{self.sender_name} <{self.smtp_username}>"
            msg["To"] = recipient
            msg["Subject"] = subject

            # Add body
            if body_text:
                text_part = MIMEText(body_text, "plain")
                msg.attach(text_part)
            if body_html:
                html_part = MIMEText(body_html, "html")
                msg.attach(html_part)

            # Add attachments
            if attachments:
                for filepath in attachments:
                    path = Path(filepath)
                    if not path.exists():
                        logger.warning(f"Attachment not found: {filepath}")
                        continue

                    with open(path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition",
                            f"attachment; filename={path.name}"
                        )
                        msg.attach(part)

            # Send
            logger.info(f"Sending email to {recipient}: {subject}")

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            logger.info("Email sent successfully")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Test the SMTP connection.

        Returns:
            True if connection successful
        """
        if not self.is_configured:
            logger.warning("Email not configured")
            return False

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                logger.info("SMTP connection test successful")
                return True
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False


# =============================================================================
# LOGGING HANDLER (for automatic alerts on errors)
# =============================================================================

class EmailLogHandler(logging.Handler):
    """
    Custom logging handler that sends emails for critical errors.

    Usage:
        alerts = EmailAlerts()
        handler = EmailLogHandler(alerts)
        handler.setLevel(logging.ERROR)
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, email_alerts: EmailAlerts):
        super().__init__()
        self.email_alerts = email_alerts
        self.setLevel(logging.ERROR)

    def emit(self, record: logging.LogRecord):
        """Send email for log record."""
        try:
            self.email_alerts.send_alert(
                subject=f"[ERROR] {record.name}",
                message=self.format(record),
                priority="high"
            )
        except Exception:
            self.handleError(record)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Email Alerts...")

    # Check configuration
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    recipient = os.getenv("ALERT_RECIPIENT")

    if not all([username, password, recipient]):
        print("\n⚠️  Email not configured. Set these environment variables:")
        print("   - SMTP_USERNAME (your Gmail address)")
        print("   - SMTP_PASSWORD (Gmail App Password)")
        print("   - ALERT_RECIPIENT (where to send alerts)")
        print("\nTo get a Gmail App Password:")
        print("1. Enable 2FA on your Google account")
        print("2. Go to https://myaccount.google.com/apppasswords")
        print("3. Generate an app password for 'Mail'")
    else:
        try:
            alerts = EmailAlerts()

            print("\n1. Testing SMTP Connection:")
            if alerts.test_connection():
                print("   ✅ Connection successful!")

                print("\n2. Sending Test Alert:")
                success = alerts.send_alert(
                    subject="Test Alert - Costa Sol Economy Monitor",
                    message="This is a test alert from the Costa del Sol Economic Monitor.",
                    priority="low"
                )
                print(f"   {'✅' if success else '❌'} Alert sent: {success}")

            else:
                print("   ❌ Connection failed")

        except Exception as e:
            print(f"❌ Error: {e}")
