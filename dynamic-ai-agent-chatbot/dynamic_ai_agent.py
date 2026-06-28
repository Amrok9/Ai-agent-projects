"""
Dynamic AI Agent Chatbot

A lightweight AI agent project designed for Google Colab and Gradio demos.

Core capabilities:
- Classifies user intent into scheduling, location/timezone lookup, analytics, or general chat.
- Manages an in-memory schedule with create, update, delete, list, and conflict detection.
- Parses natural date/time expressions such as "tomorrow", "next Tuesday", "3 pm", and "2:30 pm".
- Displays all user-facing schedule timestamps in DD/MM/YYYY HH:MM format.
- Answers supported location/timezone questions using a local lookup table.
- Runs simple analytics such as count, sum, average, median, minimum, and maximum.

Implementation notes:
- Rule-based parsing is used first for reliable demo behavior.
- The local Qwen model is loaded lazily and is only used for unknown/general requests.
- Schedule data is stored in memory and resets when the runtime restarts.
"""


from __future__ import annotations

import json
import math
import re
import statistics
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from pydantic import BaseModel, Field


# =============================================================================
# 1. Model configuration
# =============================================================================

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

tokenizer = None
model = None


def load_model() -> None:
    """Load the local Hugging Face model only once."""
    global tokenizer, model

    if tokenizer is not None and model is not None:
        return

    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError(
            "Missing model dependencies. Install torch and transformers first."
        )

    print(f"Loading model: {MODEL_ID}")
    print(f"Running on: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    model.eval()
    print("Model loaded successfully.\n")


def call_llm(messages: List[Dict[str, str]], max_new_tokens: int = 500) -> str:
    """Call the local Qwen model using Hugging Face Transformers."""
    load_model()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


# =============================================================================
# 2. Agent decision schema
# =============================================================================

class AgentDecision(BaseModel):
    """Structured decision used by the agent."""

    intent: Literal[
        "schedule_management",
        "location_info",
        "simple_analytics",
        "general_response",
    ]

    action: str = Field(
        ...,
        description=(
            "Tool action such as create, update, delete, list, "
            "get_location_info, calculate, or answer."
        ),
    )

    params: Dict[str, Any] = Field(default_factory=dict)
    reasoning_summary: Optional[str] = ""


ALLOWED_INTENTS = {
    "schedule_management",
    "location_info",
    "simple_analytics",
    "general_response",
}

try:
    AgentDecision.model_rebuild()
except Exception:
    pass


# =============================================================================
# 3. General helper functions
# =============================================================================

def clean_text(value: Any) -> str:
    """Convert a value to clean text."""
    return str(value).strip() if value is not None else ""


def normalize_spaces(text: str) -> str:
    """Remove extra spaces."""
    return re.sub(r"\s+", " ", text).strip()


def title_event_name(text: str) -> str:
    """Make event names readable."""
    text = normalize_spaces(text.strip(" .?!,-_"))

    if not text:
        return ""

    return text[0].upper() + text[1:]


def format_number(value: float) -> str:
    """Format numbers nicely."""
    if float(value).is_integer():
        return str(int(value))
    return str(round(value, 4))


# =============================================================================
# 4. Date/time formatting
# =============================================================================

INTERNAL_DATE_FORMAT = "%Y-%m-%d"
INTERNAL_TIME_FORMAT = "%H:%M"
DISPLAY_DATETIME_FORMAT = "%d/%m/%Y %H:%M"


def make_display_datetime(date_value: str, time_value: str) -> str:
    """
    Convert internal date/time into DD/MM/YYYY HH:MM.
    Example:
    2026-06-29 + 10:00 -> 29/06/2026 10:00
    """
    try:
        dt = datetime.strptime(
            f"{date_value} {time_value}",
            f"{INTERNAL_DATE_FORMAT} {INTERNAL_TIME_FORMAT}",
        )
        return dt.strftime(DISPLAY_DATETIME_FORMAT)
    except Exception:
        return f"{date_value} {time_value}"


def make_display_datetime_from_datetime(dt: datetime) -> str:
    """Format a datetime object as DD/MM/YYYY HH:MM."""
    return dt.strftime(DISPLAY_DATETIME_FORMAT)


def is_valid_date(date_value: str) -> bool:
    """Validate YYYY-MM-DD."""
    try:
        datetime.strptime(date_value, INTERNAL_DATE_FORMAT)
        return True
    except ValueError:
        return False


def is_valid_time(time_value: str) -> bool:
    """Validate HH:MM."""
    try:
        datetime.strptime(time_value, INTERNAL_TIME_FORMAT)
        return True
    except ValueError:
        return False


# =============================================================================
# 5. Natural date and time parsing
# =============================================================================

WEEKDAY_MAP = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "tuseday": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

WEEKDAY_PATTERN = "|".join(sorted(WEEKDAY_MAP.keys(), key=len, reverse=True))


def parse_natural_date(user_input: str, base_date: Optional[date] = None) -> str:
    """
    Convert date phrases into internal YYYY-MM-DD.

    Supported examples:
    - 2026-07-01
    - 01/07/2026
    - today
    - tomorrow
    - next Tuesday
    - the next tuseday
    - Tuesday
    """
    text = user_input.lower()
    today = base_date or date.today()

    # YYYY-MM-DD
    explicit_iso = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if explicit_iso:
        return explicit_iso.group(0)

    # DD/MM/YYYY
    explicit_display = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if explicit_display:
        day = int(explicit_display.group(1))
        month = int(explicit_display.group(2))
        year = int(explicit_display.group(3))

        try:
            parsed = date(year, month, day)
            return parsed.strftime(INTERNAL_DATE_FORMAT)
        except ValueError:
            return ""

    if re.search(r"\btoday\b", text):
        return today.strftime(INTERNAL_DATE_FORMAT)

    if re.search(r"\btomorrow\b", text):
        return (today + timedelta(days=1)).strftime(INTERNAL_DATE_FORMAT)

    # next Tuesday / the next Tuesday / next tuseday
    next_weekday = re.search(
        rf"\b(?:the\s+)?next\s+({WEEKDAY_PATTERN})\b",
        text,
    )

    if next_weekday:
        target_weekday = WEEKDAY_MAP[next_weekday.group(1)]
        days_ahead = target_weekday - today.weekday()

        if days_ahead <= 0:
            days_ahead += 7

        return (today + timedelta(days=days_ahead)).strftime(INTERNAL_DATE_FORMAT)

    # Tuesday / Friday / etc.
    plain_weekday = re.search(rf"\b({WEEKDAY_PATTERN})\b", text)

    if plain_weekday:
        target_weekday = WEEKDAY_MAP[plain_weekday.group(1)]
        days_ahead = target_weekday - today.weekday()

        if days_ahead < 0:
            days_ahead += 7

        return (today + timedelta(days=days_ahead)).strftime(INTERNAL_DATE_FORMAT)

    return ""


def parse_natural_time(user_input: str) -> str:
    """
    Convert time phrases into internal HH:MM.

    Supported examples:
    - 14:00
    - 3 pm -> 15:00
    - 2:30 pm -> 14:30
    - 11 am -> 11:00
    - noon -> 12:00
    - midnight -> 00:00

    Important:
    12-hour time is checked before 24-hour time so that
    '2:30 pm' becomes 14:30, not 02:30.
    """
    text = user_input.lower()

    if re.search(r"\bnoon\b", text):
        return "12:00"

    if re.search(r"\bmidnight\b", text):
        return "00:00"

    # Parse 12-hour time before 24-hour time to preserve AM/PM meaning.
    twelve_hour = re.search(
        r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm)\b",
        text,
    )

    if twelve_hour:
        hour = int(twelve_hour.group(1))
        minute = int(twelve_hour.group(2) or 0)
        meridiem = twelve_hour.group(3)

        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    # Parse standard 24-hour time after AM/PM patterns.
    twenty_four_hour = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)

    if twenty_four_hour:
        hour = int(twenty_four_hour.group(1))
        minute = int(twenty_four_hour.group(2))
        return f"{hour:02d}:{minute:02d}"

    return ""


def remove_date_time_phrases(text: str) -> str:
    """Remove common date and time expressions from event-name text."""
    cleaned = text

    cleaned = re.sub(
        r"\b\d{4}-\d{2}-\d{2}\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\b(today|tomorrow)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        rf"\b(?:on|for|to)?\s*(?:the\s+)?next\s+(?:{WEEKDAY_PATTERN})\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        rf"\b(?:on|for|to)?\s*(?:{WEEKDAY_PATTERN})\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\b(?:at\s+)?(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s*(?:am|pm)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\b(?:at\s+)?(?:[01]?\d|2[0-3]):[0-5]\d\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\b(?:at\s+)?(?:noon|midnight)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    return normalize_spaces(cleaned)


def extract_event_name(user_input: str, action: str) -> str:
    """Extract a clean event name from a scheduling request."""
    text = user_input.strip()

    called_match = re.search(
        r"\b(?:called|named|event called|event named)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if called_match:
        name = remove_date_time_phrases(called_match.group(1))
        return title_event_name(name)

    working = text

    if action == "update":
        # Example: "Move the meeting to next Friday at 2:30 pm"
        # Keep the original event name before the new target date/time.
        before_to = re.split(
            r"\bto\b",
            working,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        if before_to:
            working = before_to

    working = remove_date_time_phrases(working)

    working = re.sub(
        r"\b(please|can you|could you|would you|for me|my)\b",
        " ",
        working,
        flags=re.IGNORECASE,
    )

    working = re.sub(
        r"\b(schedule|book|create|add|set up|make|arrange|cancel|delete|remove|move|reschedule|update|change|show|list|view)\b",
        " ",
        working,
        flags=re.IGNORECASE,
    )

    working = re.sub(
        r"\b(an|a|the)\b",
        " ",
        working,
        flags=re.IGNORECASE,
    )

    working = normalize_spaces(working)

    if not working:
        if action == "create":
            return "Untitled event"
        return ""

    return title_event_name(working)


# =============================================================================
# 6. Schedule management tool
# =============================================================================

SCHEDULE_DB: List[Dict[str, str]] = []


def normalize_event_name(params: Dict[str, Any]) -> str:
    """Read event name from different possible parameter names."""
    return clean_text(
        params.get("event_name")
        or params.get("event")
        or params.get("title")
        or params.get("name")
    )


def build_event(event_name: str, date_value: str, time_value: str) -> Dict[str, str]:
    """Create an event dictionary."""
    return {
        "event_name": event_name,
        "date": date_value,
        "time": time_value,
        "datetime": make_display_datetime(date_value, time_value),
    }


def find_event_index_by_name(event_name: str) -> int:
    """Find an event by name."""
    target = event_name.lower().strip()

    for index, event in enumerate(SCHEDULE_DB):
        if event["event_name"].lower().strip() == target:
            return index

    return -1


def find_event_index_by_datetime(date_value: str, time_value: str) -> int:
    """Find an event by exact date and time."""
    for index, event in enumerate(SCHEDULE_DB):
        if event["date"] == date_value and event["time"] == time_value:
            return index

    return -1


def has_conflict(
    date_value: str,
    time_value: str,
    ignore_index: Optional[int] = None,
) -> bool:
    """Check whether another event exists at the same date and time."""
    for index, event in enumerate(SCHEDULE_DB):
        if ignore_index is not None and index == ignore_index:
            continue

        if event["date"] == date_value and event["time"] == time_value:
            return True

    return False


def schedule_management_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create, update, delete, list events, and detect scheduling conflicts."""
    action = clean_text(params.get("action") or params.get("_action")).lower()
    event_name = normalize_event_name(params)
    date_value = clean_text(params.get("date"))
    time_value = clean_text(params.get("time"))

    if action in {"add", "book", "schedule"}:
        action = "create"
    elif action in {"cancel", "remove"}:
        action = "delete"
    elif action in {"change", "move", "reschedule", "modify"}:
        action = "update"
    elif action in {"show", "view", "all"}:
        action = "list"

    if action == "list":
        return {
            "status": "success",
            "message": "Current schedule returned successfully.",
            "current_schedule": SCHEDULE_DB,
        }

    if action == "create":
        missing = []

        if not event_name:
            missing.append("event name")

        if not date_value:
            missing.append("date")

        if not time_value:
            missing.append("time")

        if missing:
            return {
                "status": "error",
                "message": "To create an event, provide: " + ", ".join(missing) + ".",
                "current_schedule": SCHEDULE_DB,
            }

        if not is_valid_date(date_value):
            return {
                "status": "error",
                "message": "Invalid date. Use DD/MM/YYYY, YYYY-MM-DD, or natural language like tomorrow.",
                "current_schedule": SCHEDULE_DB,
            }

        if not is_valid_time(time_value):
            return {
                "status": "error",
                "message": "Invalid time. Use HH:MM, 3 pm, 2:30 pm, noon, or midnight.",
                "current_schedule": SCHEDULE_DB,
            }

        if has_conflict(date_value, time_value):
            return {
                "status": "conflict",
                "message": (
                    "Another event already exists on "
                    f"{make_display_datetime(date_value, time_value)}."
                ),
                "current_schedule": SCHEDULE_DB,
            }

        new_event = build_event(event_name, date_value, time_value)
        SCHEDULE_DB.append(new_event)

        return {
            "status": "success",
            "message": "Event created successfully.",
            "event": new_event,
            "current_schedule": SCHEDULE_DB,
        }

    if action == "update":
        index = find_event_index_by_name(event_name) if event_name else -1

        if index == -1 and date_value and time_value:
            index = find_event_index_by_datetime(date_value, time_value)

        if index == -1:
            return {
                "status": "not_found",
                "message": (
                    "Could not find the event to update. "
                    "Provide the event name or its original date and time."
                ),
                "current_schedule": SCHEDULE_DB,
            }

        old_event = SCHEDULE_DB[index].copy()

        new_event_name = clean_text(
            params.get("new_event_name")
            or params.get("new_event")
            or params.get("new_title")
        ) or old_event["event_name"]

        new_date = clean_text(params.get("new_date")) or old_event["date"]
        new_time = clean_text(params.get("new_time")) or old_event["time"]

        if not is_valid_date(new_date):
            return {
                "status": "error",
                "message": "Invalid new date. Use DD/MM/YYYY, YYYY-MM-DD, or natural language.",
                "current_schedule": SCHEDULE_DB,
            }

        if not is_valid_time(new_time):
            return {
                "status": "error",
                "message": "Invalid new time. Use HH:MM, 3 pm, 2:30 pm, noon, or midnight.",
                "current_schedule": SCHEDULE_DB,
            }

        if has_conflict(new_date, new_time, ignore_index=index):
            return {
                "status": "conflict",
                "message": (
                    "Another event already exists on "
                    f"{make_display_datetime(new_date, new_time)}."
                ),
                "old_event": old_event,
                "current_schedule": SCHEDULE_DB,
            }

        updated_event = build_event(new_event_name, new_date, new_time)
        SCHEDULE_DB[index] = updated_event

        return {
            "status": "success",
            "message": "Event updated successfully.",
            "old_event": old_event,
            "updated_event": updated_event,
            "current_schedule": SCHEDULE_DB,
        }

    if action == "delete":
        index = find_event_index_by_name(event_name) if event_name else -1

        if index == -1 and date_value and time_value:
            index = find_event_index_by_datetime(date_value, time_value)

        if index == -1:
            return {
                "status": "not_found",
                "message": (
                    "Could not find the event to delete. "
                    "Provide the event name or its date and time."
                ),
                "current_schedule": SCHEDULE_DB,
            }

        deleted_event = SCHEDULE_DB.pop(index)

        return {
            "status": "success",
            "message": "Event deleted successfully.",
            "deleted_event": deleted_event,
            "current_schedule": SCHEDULE_DB,
        }

    return {
        "status": "error",
        "message": f"Unknown schedule action: {action}. Use create, update, delete, or list.",
        "current_schedule": SCHEDULE_DB,
    }


# =============================================================================
# 7. Location info tool
# =============================================================================

LOCATION_DB = {
    "cairo": {"country": "Egypt", "timezone": "Africa/Cairo"},
    "egypt": {"country": "Egypt", "timezone": "Africa/Cairo"},
    "alexandria": {"country": "Egypt", "timezone": "Africa/Cairo"},
    "tokyo": {"country": "Japan", "timezone": "Asia/Tokyo"},
    "japan": {"country": "Japan", "timezone": "Asia/Tokyo"},
    "london": {"country": "United Kingdom", "timezone": "Europe/London"},
    "paris": {"country": "France", "timezone": "Europe/Paris"},
    "new york": {"country": "United States", "timezone": "America/New_York"},
    "los angeles": {"country": "United States", "timezone": "America/Los_Angeles"},
    "dubai": {"country": "United Arab Emirates", "timezone": "Asia/Dubai"},
    "riyadh": {"country": "Saudi Arabia", "timezone": "Asia/Riyadh"},
    "berlin": {"country": "Germany", "timezone": "Europe/Berlin"},
    "madrid": {"country": "Spain", "timezone": "Europe/Madrid"},
    "rome": {"country": "Italy", "timezone": "Europe/Rome"},
    "sydney": {"country": "Australia", "timezone": "Australia/Sydney"},
    "toronto": {"country": "Canada", "timezone": "America/Toronto"},
    "singapore": {"country": "Singapore", "timezone": "Asia/Singapore"},
}


def extract_location(user_input: str) -> str:
    """Extract a location from common location/timezone questions."""
    text = user_input.strip()

    patterns = [
        r"\b(?:current\s+)?(?:local\s+)?time\s+in\s+(.+)$",
        r"\btimezone\s+(?:in|of|for)\s+(.+)$",
        r"\bcountry\s+(?:of|for)\s+(.+)$",
        r"\blocation\s+info\s+(?:for|about)\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return match.group(1).strip(" ?.!")

    return text.strip(" ?.!")


def location_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return country, timezone, and current local time for supported locations."""
    location = clean_text(
        params.get("location")
        or params.get("place")
        or params.get("city")
        or params.get("country")
    )

    if not location:
        return {
            "status": "error",
            "message": "No location was provided.",
        }

    key = location.lower().strip()

    if key not in LOCATION_DB:
        return {
            "status": "not_found",
            "message": (
                f"Unknown location: {location}. "
                "Try one of the supported locations."
            ),
            "supported_examples": sorted(LOCATION_DB.keys()),
        }

    info = LOCATION_DB[key]
    timezone_name = info["timezone"]

    try:
        current_dt = datetime.now(ZoneInfo(timezone_name))
        current_time = make_display_datetime_from_datetime(current_dt)
    except Exception:
        current_time = "Unavailable"

    return {
        "status": "success",
        "location": location.title(),
        "country": info["country"],
        "timezone": timezone_name,
        "current_local_time": current_time,
    }


# =============================================================================
# 8. Simple analytics tool
# =============================================================================

def extract_numbers(text: str) -> List[float]:
    """Extract numbers from text."""
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def simple_analytics_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate count, sum, average, median, min, max."""
    values = params.get("values") or params.get("numbers") or params.get("data") or []

    if isinstance(values, str):
        values = re.findall(r"-?\d+(?:\.\d+)?", values)

    if not isinstance(values, list) or len(values) == 0:
        return {
            "status": "error",
            "message": "Provide a non-empty list of numbers.",
        }

    numeric_values = []
    skipped_values = []

    for value in values:
        try:
            number = float(value)

            if math.isfinite(number):
                numeric_values.append(number)
            else:
                skipped_values.append(value)

        except (TypeError, ValueError):
            skipped_values.append(value)

    if not numeric_values:
        return {
            "status": "error",
            "message": "No valid numeric values were found.",
            "skipped_values": skipped_values,
        }

    return {
        "status": "success",
        "count": len(numeric_values),
        "sum": round(sum(numeric_values), 4),
        "average": round(sum(numeric_values) / len(numeric_values), 4),
        "median": round(statistics.median(numeric_values), 4),
        "maximum": max(numeric_values),
        "minimum": min(numeric_values),
        "valid_values": numeric_values,
        "skipped_values": skipped_values,
    }


# =============================================================================
# 9. Tool registry
# =============================================================================

TOOLS = {
    "schedule_management": schedule_management_tool,
    "location_info": location_info_tool,
    "simple_analytics": simple_analytics_tool,
}


# =============================================================================
# 10. LLM decision engine
# =============================================================================

DECISION_SYSTEM_PROMPT = """
You are the decision engine for a dynamic AI agent.

Your job:
1. Understand the user request.
2. Select exactly one intent.
3. Select the correct action.
4. Extract the needed parameters.
5. Return JSON only. No markdown. No explanation outside JSON.

Allowed intent values:
- "schedule_management"
- "location_info"
- "simple_analytics"
- "general_response"

Rules:
- For schedule/calendar/event requests, intent must be "schedule_management".
- Schedule actions:
  - create/add/schedule/book => action "create"
  - update/change/move/reschedule => action "update"
  - cancel/delete/remove => action "delete"
  - show/list/view schedule => action "list"
- For location/timezone/current local time/country of a place, intent must be "location_info" and action must be "get_location_info".
- For average/max/min/count/sum/median/statistics of numbers, intent must be "simple_analytics" and action must be "calculate".
- If no tool is suitable, use intent "general_response" and action "answer".

Required JSON format:
{
  "intent": "schedule_management | location_info | simple_analytics | general_response",
  "action": "create | update | delete | list | get_location_info | calculate | answer",
  "params": {},
  "reasoning_summary": "short explanation"
}
"""


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from model output."""
    if not isinstance(text, str):
        raise ValueError("LLM output is not a string.")

    text = text.strip()

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL,
    )

    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")

    return json.loads(match.group(0))


def repair_common_json_issues(text: str) -> str:
    """Clean common JSON formatting mistakes made by small local models."""
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("```", "")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text.strip()


def normalize_decision_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common wrong intents/actions into the valid schema."""
    if not isinstance(data, dict):
        raise ValueError("Decision must be a JSON object.")

    intent = str(data.get("intent", "")).strip().lower()
    action = str(data.get("action", "")).strip().lower()
    params = data.get("params") or {}

    if intent in {"create", "add", "schedule", "book"}:
        action = "create"
        intent = "schedule_management"
    elif intent in {"update", "change", "move", "reschedule", "modify"}:
        action = "update"
        intent = "schedule_management"
    elif intent in {"cancel", "delete", "remove"}:
        action = "delete"
        intent = "schedule_management"
    elif intent in {"time", "timezone", "country", "location"}:
        intent = "location_info"
        action = "get_location_info"
    elif intent in {"analytics", "analysis", "statistics", "calculate"}:
        intent = "simple_analytics"
        action = "calculate"

    if action in {"add", "schedule", "book"}:
        action = "create"
    elif action in {"cancel", "remove"}:
        action = "delete"
    elif action in {"change", "move", "reschedule", "modify"}:
        action = "update"
    elif action in {"show", "view", "all"}:
        action = "list"

    if intent not in ALLOWED_INTENTS:
        intent = "general_response"
        action = "answer"

    if intent == "schedule_management" and not action:
        action = "list"
    elif intent == "location_info":
        action = "get_location_info"
    elif intent == "simple_analytics":
        action = "calculate"
    elif intent == "general_response":
        action = "answer"

    data["intent"] = intent
    data["action"] = action
    data["params"] = params
    data.setdefault("reasoning_summary", "")

    return data


def llm_json_decision(user_input: str) -> AgentDecision:
    """Ask the LLM to classify unknown requests."""
    messages = [
        {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": f"User request: {user_input}\nReturn JSON only."},
    ]

    raw_output = call_llm(messages, max_new_tokens=450)
    data = extract_json_object(repair_common_json_issues(raw_output))
    data = normalize_decision_data(data)
    data.setdefault("params", {})
    data["params"]["_action"] = data.get("action", "")

    return AgentDecision(**data)


# =============================================================================
# 11. Reliable rule-based decision layer
# =============================================================================

def rule_based_decision(user_input: str) -> Optional[AgentDecision]:
    """
    Handle known project features deterministically.

    This prevents the model from misclassifying simple actions such as:
    - Show my schedule
    - Schedule a project review tomorrow at 10 am
    - Move the meeting to next Friday at 2:30 pm
    """
    text = user_input.lower().strip()

    analytics_words = {
        "average",
        "avg",
        "maximum",
        "max",
        "minimum",
        "min",
        "count",
        "sum",
        "median",
        "analytics",
        "statistics",
        "calculate",
    }

    if any(word in text for word in analytics_words):
        numbers = extract_numbers(user_input)

        return AgentDecision(
            intent="simple_analytics",
            action="calculate",
            params={"values": numbers, "_action": "calculate"},
            reasoning_summary="Rule-based parser detected an analytics request.",
        )

    location_words = [
        "time in",
        "timezone",
        "country",
        "local time",
        "location info",
    ]

    if any(word in text for word in location_words):
        location = extract_location(user_input)

        return AgentDecision(
            intent="location_info",
            action="get_location_info",
            params={"location": location.title(), "_action": "get_location_info"},
            reasoning_summary="Rule-based parser detected a location/time request.",
        )

    is_list_request = text.startswith(("show ", "list ", "view ")) and "schedule" in text
    is_create_request = text.startswith(("schedule ", "book ", "add ", "create ", "set up ", "arrange "))
    is_update_request = text.startswith(("move ", "reschedule ", "update ", "change "))
    is_delete_request = text.startswith(("cancel ", "delete ", "remove "))

    schedule_words = [
        "schedule",
        "book",
        "event",
        "meeting",
        "appointment",
        "calendar",
        "cancel",
        "delete",
        "remove",
        "update",
        "reschedule",
        "move",
    ]

    if is_list_request or is_create_request or is_update_request or is_delete_request or any(word in text for word in schedule_words):
        if is_list_request:
            action = "list"
        elif is_update_request:
            action = "update"
        elif is_delete_request:
            action = "delete"
        elif is_create_request:
            action = "create"
        else:
            action = "create"

        params: Dict[str, Any] = {"_action": action}

        if action != "list":
            event_name = extract_event_name(user_input, action)

            if event_name:
                params["event_name"] = event_name

        parsed_date = parse_natural_date(user_input)
        parsed_time = parse_natural_time(user_input)

        if action == "update":
            if parsed_date:
                params["new_date"] = parsed_date

            if parsed_time:
                params["new_time"] = parsed_time

        elif action in {"create", "delete"}:
            if parsed_date:
                params["date"] = parsed_date

            if parsed_time:
                params["time"] = parsed_time

        return AgentDecision(
            intent="schedule_management",
            action=action,
            params=params,
            reasoning_summary="Rule-based parser detected a schedule request.",
        )

    return None


def make_safe_general_decision() -> AgentDecision:
    """Fallback when neither rules nor LLM can classify the request."""
    return AgentDecision(
        intent="general_response",
        action="answer",
        params={
            "message": (
                "I can help with schedule management, location information, "
                "and simple analytics. Try asking me to create an event, "
                "check a timezone, or analyze a list of numbers."
            ),
            "_action": "answer",
        },
        reasoning_summary="Safe general response.",
    )


def make_decision(user_input: str) -> AgentDecision:
    """Use rules first, then the LLM for unknown requests."""
    rule_decision = rule_based_decision(user_input)

    if rule_decision is not None:
        return rule_decision

    try:
        return llm_json_decision(user_input)
    except Exception as error:
        print("\nLLM decision failed. Using safe general response.")
        print(f"Reason: {error}\n")
        return make_safe_general_decision()


def apply_natural_datetime_overrides(
    decision: AgentDecision,
    user_input: str,
) -> AgentDecision:
    """Always fix natural dates/times before calling schedule tools."""
    if decision.intent != "schedule_management":
        return decision

    parsed_date = parse_natural_date(user_input)
    parsed_time = parse_natural_time(user_input)

    if decision.action == "create":
        if parsed_date:
            decision.params["date"] = parsed_date

        if parsed_time:
            decision.params["time"] = parsed_time

        if not decision.params.get("event_name"):
            decision.params["event_name"] = (
                extract_event_name(user_input, "create") or "Untitled event"
            )

    elif decision.action == "update":
        if parsed_date:
            decision.params["new_date"] = parsed_date

        if parsed_time:
            decision.params["new_time"] = parsed_time

        if not decision.params.get("event_name"):
            event_name = extract_event_name(user_input, "update")

            if event_name:
                decision.params["event_name"] = event_name

    elif decision.action == "delete":
        if parsed_date:
            decision.params["date"] = parsed_date

        if parsed_time:
            decision.params["time"] = parsed_time

        if not decision.params.get("event_name"):
            event_name = extract_event_name(user_input, "delete")

            if event_name:
                decision.params["event_name"] = event_name

    decision.params["_action"] = decision.action

    return decision


# =============================================================================
# 12. Deterministic final response generator
# =============================================================================

def format_event(event: Dict[str, str]) -> str:
    """Format one event for user-visible output."""
    event_name = event.get("event_name", "Untitled event")
    event_datetime = event.get("datetime")

    if not event_datetime:
        event_datetime = make_display_datetime(
            event.get("date", ""),
            event.get("time", ""),
        )

    return f"{event_name} on {event_datetime}"


def format_schedule(schedule: List[Dict[str, str]]) -> str:
    """Format the in-memory schedule for display."""
    if not schedule:
        return "Your schedule is currently empty."

    rows = []

    for item in schedule:
        rows.append(f"- {format_event(item)}")

    return "Here is your current schedule:\n" + "\n".join(rows)


def deterministic_final_response(tool_result: Dict[str, Any]) -> str:
    """Generate a reliable final answer directly from the tool result."""
    status = tool_result.get("status", "unknown")
    message = tool_result.get("message", "Operation completed.")

    if status == "success":
        if "event" in tool_result:
            event = tool_result["event"]
            return f"Done. I created '{event['event_name']}' on {event['datetime']}."

        if "updated_event" in tool_result:
            event = tool_result["updated_event"]
            return f"Done. I updated it to '{event['event_name']}' on {event['datetime']}."

        if "deleted_event" in tool_result:
            event = tool_result["deleted_event"]
            return f"Done. I deleted '{format_event(event)}' from the schedule."

        if "current_schedule" in tool_result:
            return format_schedule(tool_result["current_schedule"])

        if "timezone" in tool_result:
            return (
                f"{tool_result['location']} is in {tool_result['country']}. "
                f"Timezone: {tool_result['timezone']}. "
                f"Current local time: {tool_result['current_local_time']}."
            )

        if "average" in tool_result:
            return (
                f"Analytics completed. Count: {tool_result['count']}, "
                f"Sum: {format_number(tool_result['sum'])}, "
                f"Average: {format_number(tool_result['average'])}, "
                f"Median: {format_number(tool_result['median'])}, "
                f"Min: {format_number(tool_result['minimum'])}, "
                f"Max: {format_number(tool_result['maximum'])}."
            )

    if status == "conflict":
        schedule_text = ""

        if "current_schedule" in tool_result:
            schedule_text = "\n\n" + format_schedule(tool_result["current_schedule"])

        return f"There is a scheduling conflict. {message}{schedule_text}"

    if status == "not_found":
        schedule_text = ""

        if "current_schedule" in tool_result:
            schedule_text = "\n\n" + format_schedule(tool_result["current_schedule"])

        return f"I could not find that event. {message}{schedule_text}"

    if status == "error":
        return f"Error: {message}"

    return f"Status: {status}. {message}"


def generate_final_response(
    user_input: str,
    decision: AgentDecision,
    tool_result: Dict[str, Any],
) -> str:
    """Use deterministic responses for all tool results."""
    if decision.intent == "general_response" and tool_result.get("message"):
        return tool_result["message"]

    return deterministic_final_response(tool_result)


# =============================================================================
# 13. Main agent function
# =============================================================================

def run_agent(user_input: str) -> Dict[str, Any]:
    """Main dynamic AI agent function."""
    user_input = clean_text(user_input)

    if not user_input:
        return {
            "user_input": user_input,
            "decision": {},
            "tool_result": {
                "status": "error",
                "message": "No input provided.",
            },
            "final_response": "Please type a message first.",
        }

    decision = make_decision(user_input)
    decision = apply_natural_datetime_overrides(decision, user_input)

    if decision.intent == "general_response":
        try:
            response = call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Keep answers short and clear.",
                    },
                    {
                        "role": "user",
                        "content": user_input,
                    },
                ],
                max_new_tokens=250,
            )

            tool_result = {
                "status": "success",
                "message": response,
            }

        except Exception:
            tool_result = {
                "status": "success",
                "message": decision.params.get(
                    "message",
                    "I can help with schedule management, location info, and simple analytics.",
                ),
            }

    else:
        tool = TOOLS.get(decision.intent)

        if tool is None:
            tool_result = {
                "status": "error",
                "message": f"No tool found for intent: {decision.intent}",
            }

        else:
            params = dict(decision.params)
            params["_action"] = decision.action
            params["action"] = decision.action
            tool_result = tool(params)

    final_response = generate_final_response(user_input, decision, tool_result)

    return {
        "user_input": user_input,
        "decision": decision.model_dump(),
        "tool_result": tool_result,
        "final_response": final_response,
    }


def print_agent_result(result: Dict[str, Any]) -> None:
    """Pretty-print the full agent result for debugging."""
    print("=" * 80)
    print("USER INPUT")
    print(result["user_input"])

    print("\nAGENT DECISION")
    print(json.dumps(result["decision"], indent=2, default=str))

    print("\nTOOL RESULT")
    print(json.dumps(result["tool_result"], indent=2, default=str))

    print("\nFINAL RESPONSE")
    print(result["final_response"])
    print("=" * 80)


# =============================================================================
# 14. Interactive terminal loop
# =============================================================================

def interactive_loop() -> None:
    """Run a terminal-based chat loop."""
    print("\nDynamic AI Agent is ready.")
    print("Capabilities: schedule management, location info, simple analytics.")
    print("Type 'help' for examples or 'exit' to stop.\n")

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in {"exit", "quit", "stop"}:
            print("Agent stopped.")
            break

        if user_text.lower() == "help":
            print(
                "\nTry these examples:\n"
                "- Schedule a meeting tomorrow at 10 am\n"
                "- Schedule a project review tomorrow at 10 am\n"
                "- Show my schedule\n"
                "- Move the meeting to next Friday at 2:30 pm\n"
                "- Cancel the event called meeting\n"
                "- What is the current local time in Paris?\n"
                "- Calculate the average and maximum for 10, 20, 30, 40\n"
            )
            continue

        if not user_text:
            continue

        result = run_agent(user_text)
        print_agent_result(result)


# =============================================================================
# 15. Gradio chat UI
# =============================================================================

def chat_agent(message: str, history: Optional[List[Any]]) -> tuple[str, List[Any]]:
    """Gradio chat handler using tuple history for broad Gradio compatibility."""
    history = history or []
    message = clean_text(message)

    if not message:
        return "", history

    result = run_agent(message)
    reply = result["final_response"]

    history = history + [(message, reply)]

    return "", history


def reset_schedule(history: Optional[List[Any]]) -> List[Any]:
    """Clear the in-memory schedule and add a confirmation to the chat."""
    SCHEDULE_DB.clear()

    history = history or []
    history = history + [
        (
            "Reset schedule",
            "Done. I cleared the in-memory schedule.",
        )
    ]

    return history


def launch_ui(share: bool = True):
    """Launch the Gradio UI."""
    import gradio as gr

    with gr.Blocks(title="Dynamic AI Agent Chatbot") as demo:
        gr.Markdown(
            """
            # Dynamic AI Agent Chatbot

            Chat with your local Qwen-powered agent. It can manage an in-memory schedule,
            answer supported location/timezone questions, and run simple analytics.
            """
        )

        chatbot = gr.Chatbot(
            label="Chat",
            height=460,
        )

        message_box = gr.Textbox(
            label="Your message",
            placeholder="Type your message here...",
            lines=2,
        )

        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear chat")
            reset_button = gr.Button("Reset schedule")

        send_button.click(
            fn=chat_agent,
            inputs=[message_box, chatbot],
            outputs=[message_box, chatbot],
        )

        message_box.submit(
            fn=chat_agent,
            inputs=[message_box, chatbot],
            outputs=[message_box, chatbot],
        )

        clear_button.click(
            fn=lambda: [],
            outputs=chatbot,
        )

        reset_button.click(
            fn=reset_schedule,
            inputs=chatbot,
            outputs=chatbot,
        )

    demo.launch(share=share)
    return demo
