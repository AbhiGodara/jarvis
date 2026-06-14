import os
import datetime
import pickle
import logging
import pytz
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from commands.registry import command
from core.config import get_config

# Try importing from parent modules for speaking feedback
try:
    from tts import speak
except ImportError:
    speak = None

logger = logging.getLogger(__name__)

_TIMEZONE_STR = get_config().calendar_timezone or "UTC"
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY_EXTENSIONS = ["rd", "th", "st", "nd"]


def _authenticate_google() -> tuple[any, str | None]:
    """
    Authenticate with Google Calendar API.
    Returns:
        (service_instance, error_message)
    """
    creds = None
    token_path = 'token.pickle'
    creds_path = 'credentials.json'

    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists(token_path):
        try:
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            logger.warning(f"Failed to load token.pickle: {e}")

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning(f"Failed to refresh credentials: {e}")
                creds = None

        if not creds:
            if not os.path.exists(creds_path):
                logger.error("credentials.json is missing.")
                return None, (
                    "Google Calendar credentials file is missing. "
                    "Please download credentials.json from the Google Cloud Console "
                    "and place it in the root folder of the project."
                )
            
            try:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                # Run local server to authenticate
                creds = flow.run_local_server(port=0)
            except Exception as e:
                logger.error(f"Google Calendar OAuth flow failed: {e}")
                return None, f"Google Calendar login failed: {e}."

        # Save the credentials for the next run
        try:
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
        except Exception as e:
            logger.warning(f"Failed to save token.pickle: {e}")

    try:
        service = build('calendar', 'v3', credentials=creds)
        return service, None
    except Exception as e:
        logger.error(f"Failed to build Google Calendar service client: {e}")
        return None, f"Failed to initialize Calendar API: {e}."


def _parse_date(text: str) -> datetime.date:
    """Parse date from voice command text."""
    today = datetime.date.today()
    lower_text = text.lower()

    if "today" in lower_text:
        return today

    day = -1
    day_of_week = -1
    month = -1
    year = today.year

    for word in lower_text.split():
        if word in MONTHS:
            month = MONTHS.index(word) + 1
        elif word in DAYS:
            day_of_week = DAYS.index(word)
        elif word.isdigit():
            day = int(word)
        else:
            for ext in DAY_EXTENSIONS:
                found = word.find(ext)
                if found > 0:
                    try:
                        day = int(word[:found])
                    except ValueError:
                        pass

    if month < today.month and month != -1:  
        year = year + 1

    if month == -1 and day != -1:  
        if day < today.day:
            month = today.month + 1
        else:
            month = today.month

    if month == -1 and day == -1 and day_of_week != -1:
        current_day_of_week = today.weekday()
        dif = day_of_week - current_day_of_week

        if dif < 0:
            dif += 7
            if "next" in lower_text:
                dif += 7

        return today + datetime.timedelta(dif)

    if day != -1:
        try:
            return datetime.date(month=month, day=day, year=year)
        except ValueError:
            return today

    return today


@command(keywords=["what do i have", "do i have plans", "am i busy", "google calendar", "calendar events", "schedule on"])
def check_google_calendar(text: str) -> str:
    """List Google Calendar events for a specific day."""
    service, err_msg = _authenticate_google()
    if err_msg:
        return err_msg

    target_date = _parse_date(text)
    
    # Calculate start and end times for the target day
    date_min = datetime.datetime.combine(target_date, datetime.datetime.min.time())
    date_max = datetime.datetime.combine(target_date, datetime.datetime.max.time())
    
    # Apply timezone
    try:
        tz = pytz.timezone(_TIMEZONE_STR)
    except Exception:
        tz = pytz.UTC
        
    date_min = tz.localize(date_min)
    date_max = tz.localize(date_max)

    try:
        logger.info(f"Fetching calendar events for {target_date}...")
        events_result = service.events().list(
            calendarId='primary', 
            timeMin=date_min.isoformat(), 
            timeMax=date_max.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        date_str = target_date.strftime("%A, %B %d")
        if not events:
            return f"You have no upcoming events on your calendar for {date_str}."

        response = f"You have {len(events)} event{'s' if len(events) != 1 else ''} scheduled for {date_str}. "
        event_list = []
        
        for event in events:
            summary = event.get('summary', 'Untitled Event')
            start = event['start'].get('dateTime', event['start'].get('date'))
            
            # Format event start time
            if 'T' in start:
                time_part = start.split("T")[1].split("+")[0].split("-")[0]
                hour = int(time_part.split(":")[0])
                minute = time_part.split(":")[1]
                ampm = "a.m." if hour < 12 else "p.m."
                disp_hour = hour if hour <= 12 else hour - 12
                if disp_hour == 0:
                    disp_hour = 12
                start_time_str = f"at {disp_hour}:{minute} {ampm}"
            else:
                start_time_str = "all day"
                
            event_list.append(f"{summary} {start_time_str}")

        response += " ".join(event_list)
        return response

    except Exception as e:
        logger.error(f"Failed to fetch calendar events: {e}")
        return "I was unable to retrieve events from your Google Calendar."
