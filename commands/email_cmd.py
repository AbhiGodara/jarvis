import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from commands.registry import command
from core.config import get_config

# Try importing from parent modules for interactive prompting
try:
    from tts import speak
    from stt import listen
except ImportError:
    speak = None
    listen = None

logger = logging.getLogger(__name__)

_cfg = get_config()
_CONTACTS = _cfg.email_contacts


def _get_input(prompt_msg: str) -> str | None:
    """Helper to ask user a question via speak/listen or CLI fallback."""
    if speak:
        speak(prompt_msg)
    else:
        print(prompt_msg)

    # Try voice first if available
    response = None
    if listen:
        response = listen(timeout=10, phrase_limit=15)
    
    # Fallback to CLI input if voice was empty or not available
    if not response:
        print(f"(Press Enter to skip voice) or type response for '{prompt_msg}': ")
        try:
            response = input().strip()
        except (KeyboardInterrupt, EOFError):
            return None

    return response if response else None


@command(keywords=["send email", "send an email", "send a mail", "send mail", "write an email", "compose an email"])
def send_email_cmd(text: str) -> str:
    """Send an email dynamically by looking up contacts and prompting for content."""
    
    # 1. Check SMTP credentials
    sender_email = _cfg.smtp_sender_email or _cfg.email_sender
    sender_password = _cfg.smtp_sender_password

    if not sender_email or not sender_password or "your_email" in sender_email or "your_app_password" in sender_password:
        logger.warning("Email credentials are not configured in .env file.")
        return (
            "The email feature is not configured. Please set SMTP_SENDER_EMAIL "
            "and SMTP_SENDER_PASSWORD in your dot env file."
        )

    # 2. Extract or prompt for recipient
    recipient_name = ""
    recipient_email = ""

    # Check if recipient was already in the voice command, e.g., "send email to assistant"
    for word in text.split():
        if word in _CONTACTS:
            recipient_name = word
            recipient_email = _CONTACTS[word]
            break

    if not recipient_email:
        # Prompt user for recipient
        ans = _get_input("Whom do you want to email?")
        if not ans:
            return "Email cancelled. No recipient provided."
        
        # Clean up response
        ans = ans.lower().replace("email", "").replace("to", "").strip()
        if ans in _CONTACTS:
            recipient_name = ans
            recipient_email = _CONTACTS[ans]
        elif "@" in ans:
            recipient_email = ans
        else:
            # Let them type it in console if voice failed to capture standard email address format
            if speak:
                speak("I couldn't match that name in contacts. Please enter the email address in the terminal.")
            print("Enter recipient email address: ")
            try:
                recipient_email = input().strip()
            except (KeyboardInterrupt, EOFError):
                return "Email cancelled."

    if not recipient_email or "@" not in recipient_email:
        return f"Invalid email address provided: {recipient_email}."

    # 3. Prompt for Subject
    subject = _get_input("What is the subject of the email?")
    if not subject:
        subject = "No Subject (Sent by JARVIS)"

    # 4. Prompt for Message Body
    message_body = _get_input("What is the message of the email?")
    if not message_body:
        return "Email cancelled. Message body was empty."

    # 5. Send the email
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message_body, 'plain'))

        logger.info(f"Connecting to SMTP server to send email from {sender_email} to {recipient_email}...")
        
        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.close()

        recipient_disp = recipient_name if recipient_name else recipient_email
        logger.info(f"Email sent successfully to {recipient_email}")
        return f"The email has been successfully sent to {recipient_disp}."

    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP Authentication failed. Check your App Password.")
        return "Failed to send email. There was an SMTP authentication error. Please check your app password."
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return f"I encountered an error sending the email: {e}"
