"""
src/openf1_data.py

OpenF1 API integration layer for f1-race-replay.
Enriches FastF1 telemetry with live/real-time data from https://api.openf1.org/v1

What this adds on top of FastF1:
  - Intervals (gap_to_leader, gap to car ahead) — timestamped per driver
  - Race control messages (flags, SC, VSC) — more granular than FastF1 track_status
  - Pit stop data — precise lap number + duration
  - Team radio — URLs to audio clips per driver
  - Weather — supplemental (FastF1 already provides this)

Requires: requests
Only works for sessions from 2023 onwards (OpenF1 data availability).

Usage in f1_data.py:
    from src import openf1_data
    session_key = openf1_data.get_session_key_from_session(fastf1_session)
    if session_key:
        intervals = openf1_data.get_intervals(session_key)
        ...
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
REQUEST_TIMEOUT = 30  # seconds

# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str, params: dict | None = None) -> list:
    """
    Make a GET request to the OpenF1 API.
    Returns parsed JSON list, or [] on any error.
    """
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        log.warning("[OpenF1] HTTP %s for %s: %s", resp.status_code, url, e)
    except requests.RequestException as e:
        log.warning("[OpenF1] Request failed for %s: %s", url, e)
    except Exception as e:
        log.warning("[OpenF1] Unexpected error for %s: %s", url, e)
    return []


def _parse_openf1_date(date_str: str | None) -> float | None:
    """
    Parse an ISO 8601 date string (with or without timezone) to a UTC unix
    timestamp (float seconds).  Returns None if parsing fails.
    """
    if not date_str:
        return None
    try:
        s = str(date_str).strip()
        # Normalise trailing 'Z' → '+00:00'
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

# Mapping from FastF1 session name → OpenF1 session_name
_SESSION_NAME_MAP: dict[str, str] = {
    "Race":              "Race",
    "Qualifying":        "Qualifying",
    "Sprint":            "Sprint",
    "Sprint Qualifying": "Sprint Qualifying",
    "Sprint Shootout":   "Sprint Shootout",
    "Practice 1":        "Practice 1",
    "Practice 2":        "Practice 2",
    "Practice 3":        "Practice 3",
    # Short codes (used in main.py flags)
    "R":                 "Race",
    "Q":                 "Qualifying",
    "S":                 "Sprint",
    "SQ":                "Sprint Qualifying",
    "FP1":               "Practice 1",
    "FP2":               "Practice 2",
    "FP3":               "Practice 3",
}


def get_session_key_from_session(fastf1_session) -> int | None:
    """
    Reliably resolve an OpenF1 session_key from a loaded FastF1 session object.

    Strategy: match by session start date (±12 hours) rather than by round
    number, since OpenF1's round ordering can differ from FastF1's.

    Args:
        fastf1_session: A loaded fastf1.core.Session object.

    Returns:
        Integer session_key, or None if not found / year < 2023.
    """
    try:
        year = int(fastf1_session.event["EventDate"].year)
    except Exception:
        return None

    if year < 2023:
        log.info("[OpenF1] Year %d < 2023 — skipping enrichment.", year)
        return None

    # Map FastF1 session name → OpenF1 session name
    ff1_name = getattr(fastf1_session, "name", "") or ""
    openf1_name = _SESSION_NAME_MAP.get(ff1_name, ff1_name)

    # UTC timestamp of FastF1 session start
    try:
        target_ts = fastf1_session.date.timestamp()
    except Exception:
        return None

    # Fetch all sessions of this type + year
    sessions = _get("sessions", {"year": year, "session_name": openf1_name})
    if not sessions:
        log.warning("[OpenF1] No sessions found for year=%d name=%s", year, openf1_name)
        return None

    # Pick the session whose date_start is closest to the FastF1 session start
    best_key: int | None = None
    best_diff = float("inf")

    for s in sessions:
        ts = _parse_openf1_date(s.get("date_start"))
        if ts is None:
            continue
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_key = s.get("session_key")

    if best_key and best_diff < 43200:  # within 12 hours
        log.info("[OpenF1] Resolved session_key=%d (diff=%.0fs)", best_key, best_diff)
        return best_key

    log.warning("[OpenF1] Could not resolve session_key (closest diff=%.0fs)", best_diff)
    return None


# ---------------------------------------------------------------------------
# Raw data fetchers
# ---------------------------------------------------------------------------

def get_race_control_messages(session_key: int) -> list:
    """
    Race control messages: flags, SC deployments, DRS open/close, etc.

    Key response fields per entry:
        date, category, flag, message, lap_number, scope, sector, driver_number
    """
    return _get("race_control", {"session_key": session_key})


def get_intervals(session_key: int) -> list:
    """
    Timestamped gap data for every driver throughout the session.

    Key response fields per entry:
        date, driver_number, gap_to_leader, interval
    gap_to_leader / interval can be:
        - A float string like "+1.234"
        - "LAP" / "LAPS" for drivers who are lapped
        - None before data is available
    """
    return _get("intervals", {"session_key": session_key})


def get_weather(session_key: int) -> list:
    """
    Weather samples throughout the session.

    Key fields: date, air_temperature, track_temperature, humidity,
                pressure, rainfall, wind_direction, wind_speed
    """
    return _get("weather", {"session_key": session_key})


def get_pit_stops(session_key: int) -> list:
    """
    Pit stop events for all drivers.

    Key fields: date, driver_number, lap_number, pit_duration
    """
    return _get("pits", {"session_key": session_key})


def get_team_radio(session_key: int, driver_number: int | None = None) -> list:
    """
    Team radio events (links to audio clips).

    Key fields: date, driver_number, recording_url
    """
    params: dict = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    return _get("team_radio", params)


def get_drivers(session_key: int) -> list:
    """Driver metadata for the session (name, team, number, etc.)."""
    return _get("drivers", {"session_key": session_key})


# ---------------------------------------------------------------------------
# Data processing / transformation helpers
# ---------------------------------------------------------------------------

def _parse_gap(val) -> float:
    """
    Parse a gap value from the OpenF1 intervals endpoint.
    Returns seconds as float.  Lapped drivers get a sentinel value of 999.0.
    """
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lstrip("+")
    if not s or s.upper() in ("LAP", "LAPS"):
        return 999.0  # sentinel: driver is lapped
    try:
        return float(s)
    except ValueError:
        return 0.0


def build_driver_number_map(fastf1_session) -> dict[str, str]:
    """
    Build a mapping from OpenF1 driver_number (as str) → FastF1 abbreviation.

    e.g. {"44": "HAM", "1": "VER", ...}
    """
    mapping: dict[str, str] = {}
    for num in fastf1_session.drivers:
        try:
            abbr = fastf1_session.get_driver(num)["Abbreviation"]
            mapping[str(num)] = abbr
        except Exception:
            pass
    return mapping


def build_openf1_track_statuses(
    race_control_messages: list,
    session_start_utc: float,
    global_t_min: float,
) -> list:
    """
    Convert OpenF1 race control flag messages into the FastF1 track_status
    format so they can be used as a drop-in replacement / enrichment.

    FastF1 status codes used by the visualiser:
        "1" = Green  "2" = Yellow  "4" = Safety Car
        "5" = Red    "6" = VSC     "7" = VSC ending

    Args:
        race_control_messages:  List from get_race_control_messages().
        session_start_utc:      UTC unix timestamp when the session started.
                                Use fastf1_session.date.timestamp().
        global_t_min:           FastF1 global_t_min (seconds) — the offset
                                applied to FastF1's SessionTime timeline.

    Returns:
        List of dicts [{status, start_time, end_time}] compatible with the
        existing track_status format.  Returns [] if no flag messages found.
    """
    FLAG_TO_STATUS: dict[str, str] = {
        "GREEN":              "1",
        "YELLOW":             "2",
        "DOUBLE YELLOW":      "2",
        "RED":                "5",
        "SAFETY CAR":         "4",
        "VIRTUAL SAFETY CAR": "6",
        "VSC ENDING":         "7",
        "CHEQUERED":          "1",
        "CLEAR":              "1",
        "BLUE":               "1",  # blue flag doesn't affect track status colour
    }

    # Only keep flag-change events
    flag_msgs = [
        m for m in race_control_messages
        if m.get("flag") and str(m["flag"]).upper() in FLAG_TO_STATUS
    ]

    if not flag_msgs:
        return []

    # Sort chronologically
    flag_msgs.sort(key=lambda m: m.get("date", ""))

    statuses: list[dict] = []

    for msg in flag_msgs:
        utc_ts = _parse_openf1_date(msg.get("date"))
        if utc_ts is None:
            continue

        # Convert wall-clock UTC → FastF1 timeline seconds
        # FastF1 timeline[0] = 0  corresponds to wall time (session_start_utc + global_t_min)
        t = (utc_ts - session_start_utc) - global_t_min
        status_code = FLAG_TO_STATUS[str(msg["flag"]).upper()]

        # Close the previous entry
        if statuses:
            statuses[-1]["end_time"] = t

        statuses.append({
            "status": status_code,
            "start_time": t,
            "end_time": None,
        })

    return statuses


def build_openf1_intervals(
    intervals_data: list,
    driver_number_to_code: dict[str, str],
    session_start_utc: float,
    global_t_min: float,
    timeline: "np.ndarray",
) -> dict[str, dict[str, "np.ndarray"]]:
    """
    Resample OpenF1 interval data onto the FastF1 common timeline.

    Args:
        intervals_data:         List from get_intervals().
        driver_number_to_code:  {"44": "HAM", ...}  from build_driver_number_map().
        session_start_utc:      UTC unix timestamp for session start.
        global_t_min:           FastF1 global_t_min offset.
        timeline:               The resampled np.ndarray of time values (from f1_data.py).

    Returns:
        {driver_code: {"gap_to_leader": np.ndarray, "interval": np.ndarray}}
        Arrays are the same length as `timeline`.
    """
    if not intervals_data or len(timeline) == 0:
        return {}

    # Accumulate raw samples per driver
    raw: dict[str, dict] = {}

    for entry in intervals_data:
        code = driver_number_to_code.get(str(entry.get("driver_number", "")))
        if not code:
            continue

        utc_ts = _parse_openf1_date(entry.get("date"))
        if utc_ts is None:
            continue

        t = (utc_ts - session_start_utc) - global_t_min

        if code not in raw:
            raw[code] = {"times": [], "gap": [], "interval": []}

        raw[code]["times"].append(t)
        raw[code]["gap"].append(_parse_gap(entry.get("gap_to_leader")))
        raw[code]["interval"].append(_parse_gap(entry.get("interval")))

    result: dict[str, dict[str, np.ndarray]] = {}

    for code, data in raw.items():
        if len(data["times"]) < 2:
            continue

        times = np.array(data["times"])
        order = np.argsort(times)
        times = times[order]
        gap_arr = np.array(data["gap"])[order]
        int_arr = np.array(data["interval"])[order]

        # Clamp timeline extrapolation (hold last value at edges)
        t_min_data, t_max_data = float(times[0]), float(times[-1])
        tl_clamped = np.clip(timeline, t_min_data, t_max_data)

        result[code] = {
            "gap_to_leader": np.interp(tl_clamped, times, gap_arr),
            "interval":      np.interp(tl_clamped, times, int_arr),
        }

    return result


def build_openf1_pit_events(
    pit_data: list,
    driver_number_to_code: dict[str, str],
) -> dict[str, list]:
    """
    Convert OpenF1 pit stop data to a per-driver list of pit events.

    Returns:
        {driver_code: [{"lap_number": int, "pit_duration": float, "date": str}]}
    """
    result: dict[str, list] = {}
    for entry in pit_data:
        code = driver_number_to_code.get(str(entry.get("driver_number", "")))
        if not code:
            continue
        result.setdefault(code, []).append({
            "lap_number":   entry.get("lap_number"),
            "pit_duration": entry.get("pit_duration"),
            "date":         entry.get("date", ""),
        })
    return result


def build_openf1_radio_events(
    radio_data: list,
    driver_number_to_code: dict[str, str],
    session_start_utc: float,
    global_t_min: float,
) -> dict[str, list]:
    """
    Convert OpenF1 team radio data to per-driver events with timeline-aligned
    timestamps so the UI can show notifications at the right moment.

    Returns:
        {driver_code: [{"t": float, "recording_url": str}]}
    where `t` is on the same timeline as the replay frames.
    """
    result: dict[str, list] = {}
    for entry in radio_data:
        code = driver_number_to_code.get(str(entry.get("driver_number", "")))
        if not code:
            continue

        utc_ts = _parse_openf1_date(entry.get("date"))
        if utc_ts is None:
            continue

        t = (utc_ts - session_start_utc) - global_t_min
        result.setdefault(code, []).append({
            "t":             round(t, 3),
            "recording_url": entry.get("recording_url", ""),
        })
    return result


def build_openf1_race_control_for_frames(
    race_control_messages: list,
    session_start_utc: float,
    global_t_min: float,
    fps: int,
    num_frames: int,
) -> dict[int, list]:
    """
    Map each race control message to a frame index so the visualiser can
    display banners / notifications at the correct moment in the replay.

    Returns:
        {frame_index: [{"message": str, "flag": str, "category": str, "lap": int|None}]}
    """
    frame_events: dict[int, list] = {}

    for msg in race_control_messages:
        utc_ts = _parse_openf1_date(msg.get("date"))
        if utc_ts is None:
            continue

        t = (utc_ts - session_start_utc) - global_t_min
        idx = int(round(t * fps))

        if 0 <= idx < num_frames:
            frame_events.setdefault(idx, []).append({
                "message":  msg.get("message", ""),
                "flag":     msg.get("flag", ""),
                "category": msg.get("category", ""),
                "lap":      msg.get("lap_number"),
            })

    return frame_events