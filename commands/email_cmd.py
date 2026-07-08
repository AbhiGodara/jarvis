"""
commands/email_cmd.py — voice email.

Mk-III Phase 6: with dialog_manager enabled (default), "send an email" starts
a non-blocking slot-filling session (recipient → subject → body → confirm)
that runs inside the normal voice loop — the planner feeds each subsequent
utterance to it, and the SMTP send happens in a background thread after the
user confirms ("Sending, sir" is immediate; a failure is announced
asynchronously). With dialog_manager: false the Mk-II blocking flow
(speak/listen/input prompts) is used instead.
"""
import logging
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from commands.registry import command
from core.config import get_config

# Only the legacy blocking flow uses these interactive imports.
try:
    from tts import speak
    from stt import listen
except ImportError:
    speak = None
    listen = None

logger = logging.getLogger(__name__)

_cfg = get_config()
_CONTACTS = _cfg.email_contacts

_CONFIRM_YES = frozenset([
    "yes", "yeah", "yep", "sure", "send", "send it", "go ahead",
    "confirm", "do it", "okay", "ok", "yes send it", "affirmative",
])
_CONFIRM_NO = frozenset([
    "no", "nope", "don't", "do not", "don't send", "do not send it",
    "no don't", "negative", "hold on", "wait",
])


def _dialogs_enabled() -> bool:
    return bool(_cfg.dialog_manager)


def _credentials() -> tuple[str, str] | None:
    """(sender, password) when SMTP is usable, else None."""
    sender = _cfg.smtp_sender_email or _cfg.email_sender
    password = _cfg.smtp_sender_password
    if (
        not sender or not password
        or "your_email" in sender or "your_app_password" in password
    ):
        return None
    return sender, password


def _resolve_recipient(text: str) -> tuple[str | None, str]:
    """Match a contact name or a literal address inside an utterance.

    Returns (email_or_None, display_name).
    """
    lower = text.lower()
    for name, address in _CONTACTS.items():
        if name in lower:
            return address, name
    for word in text.replace(",", " ").split():
        if "@" in word and "." in word:
            return word.strip(".,;:!?"), word.strip(".,;:!?")
    return None, ""


def _send_via_smtp(sender: str, password: str, recipient: str, subject: str, body: str) -> None:
    """Blocking SMTP send; raises on failure. Tests inject a fake here."""
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    logger.info(f"Connecting to SMTP server to send email from {sender} to {recipient}...")
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(sender, password)
    server.sendmail(sender, recipient, msg.as_string())
    server.close()
    logger.info(f"Email sent successfully to {recipient}")


def _dispatch_send(recipient: str, display: str, subject: str, body: str) -> None:
    """Fire the send in a background thread — the voice loop never waits on
    SMTP. Failures are spoken asynchronously via the dialog announcer."""
    creds = _credentials()
    if creds is None:   # session start checked this; re-check for safety
        return

    def _worker():
        from agents.dialog import announce
        try:
            _send_via_smtp(creds[0], creds[1], recipient, subject, body)
        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP Authentication failed. Check your App Password.")
            announce(
                f"Sir, the email to {display} failed to send — "
                f"there was an SMTP authentication error."
            )
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            announce(f"Sir, the email to {display} failed to send.")

    threading.Thread(target=_worker, name="email-send", daemon=True).start()


# ── Dialog-mode session (Mk-III Phase 6) ────────────────────────────────────

def _validate_recipient(value: str, session) -> tuple[str | None, str | None]:
    address, name = _resolve_recipient(value)
    if address is None:
        return None, (
            "I couldn't find that name in your contacts, sir. "
            "Who should receive it? You can also spell out the address."
        )
    session.slots["_recipient_display"] = name or address
    return address, None


def _validate_confirm(value: str, session) -> tuple[str | None, str | None]:
    normalized = value.strip().strip("?!.,;:").lower()
    if normalized in _CONFIRM_YES:
        return "yes", None
    if normalized in _CONFIRM_NO:
        # A declined confirmation ends the session without sending — mark it
        # cancelled via the manager path by treating "no" as terminal.
        return "no", None
    return None, "Shall I send it, sir? Yes or no."


def _complete_email(session) -> str:
    if session.slots.get("confirm") != "yes":
        return "Understood, sir. I won't send it."
    display = session.slots.get("_recipient_display") or session.slots["recipient"]
    _dispatch_send(
        recipient=session.slots["recipient"],
        display=display,
        subject=session.slots["subject"],
        body=session.slots["body"],
    )
    return f"Sending the email to {display} now, sir."


def _start_email_dialog(text: str) -> str:
    from agents.dialog import DialogSession, get_manager

    if _credentials() is None:
        logger.warning("Email credentials are not configured in .env file.")
        return (
            "The email feature is not configured. Please set SMTP_SENDER_EMAIL "
            "and SMTP_SENDER_PASSWORD in your dot env file."
        )

    # "_recipient_display" is bookkeeping, pre-filled so it's never prompted.
    slots: dict = {
        "recipient": None,
        "_recipient_display": "",
        "subject": None,
        "body": None,
        "confirm": None,
    }
    # Recipient already in the trigger utterance ("send an email to assistant")
    # skips its prompt entirely.
    address, name = _resolve_recipient(text)
    if address is not None:
        slots["recipient"] = address
        slots["_recipient_display"] = name or address

    session = DialogSession(
        command_name="email",
        slots=slots,
        prompts={
            "recipient": "Whom do you want to email, sir?",
            "subject": "What's the subject?",
            "body": "What should the message say?",
            "confirm": "Shall I send it, sir?",
        },
        validators={
            "recipient": _validate_recipient,
            "confirm": _validate_confirm,
        },
        on_complete=_complete_email,
    )
    return get_manager().start(session)


# ── Legacy blocking flow (dialog_manager: false rollback) ───────────────────

def _get_input(prompt_msg: str) -> str | None:
    """Helper to ask user a question via speak/listen or CLI fallback."""
    if speak:
        speak(prompt_msg)
    else:
        print(prompt_msg)

    response = None
    if listen:
        response = listen(timeout=10, phrase_limit=15)

    if not response:
        print(f"(Press Enter to skip voice) or type response for '{prompt_msg}': ")
        try:
            response = input().strip()
        except (KeyboardInterrupt, EOFError):
            return None

    return response if response else None


def _send_email_blocking(text: str) -> str:
    """The Mk-II interactive flow — hijacks the voice loop until done."""
    creds = _credentials()
    if creds is None:
        logger.warning("Email credentials are not configured in .env file.")
        return (
            "The email feature is not configured. Please set SMTP_SENDER_EMAIL "
            "and SMTP_SENDER_PASSWORD in your dot env file."
        )
    sender_email, sender_password = creds

    recipient_email, recipient_name = None, ""
    address, name = _resolve_recipient(text)
    if address is not None:
        recipient_email, recipient_name = address, name

    if not recipient_email:
        ans = _get_input("Whom do you want to email?")
        if not ans:
            return "Email cancelled. No recipient provided."
        ans = ans.lower().replace("email", "").replace("to", "").strip()
        if ans in _CONTACTS:
            recipient_name = ans
            recipient_email = _CONTACTS[ans]
        elif "@" in ans:
            recipient_email = ans
        else:
            if speak:
                speak("I couldn't match that name in contacts. Please enter the email address in the terminal.")
            print("Enter recipient email address: ")
            try:
                recipient_email = input().strip()
            except (KeyboardInterrupt, EOFError):
                return "Email cancelled."

    if not recipient_email or "@" not in recipient_email:
        return f"Invalid email address provided: {recipient_email}."

    subject = _get_input("What is the subject of the email?")
    if not subject:
        subject = "No Subject (Sent by JARVIS)"

    message_body = _get_input("What is the message of the email?")
    if not message_body:
        return "Email cancelled. Message body was empty."

    try:
        _send_via_smtp(sender_email, sender_password, recipient_email, subject, message_body)
        recipient_disp = recipient_name if recipient_name else recipient_email
        return f"The email has been successfully sent to {recipient_disp}."
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP Authentication failed. Check your App Password.")
        return "Failed to send email. There was an SMTP authentication error. Please check your app password."
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return f"I encountered an error sending the email: {e}"


@command(
    keywords=["send email", "send an email", "send a mail", "send mail", "write an email", "compose an email"],
    examples=[
        "send an email to my assistant",
        "compose a mail for me",
        "i need to email someone",
        "shoot an email to john",
        "draft an email about the meeting",
    ],
)
def send_email_cmd(text: str) -> str:
    """Send an email — as a non-blocking dialog (default) or the legacy
    interactive flow when dialog_manager is disabled."""
    if _dialogs_enabled():
        return _start_email_dialog(text)
    return _send_email_blocking(text)
