"""
Time utility functions for the crypto algorithmic trading system.

Provides time-related operations:
- Timestamp parsing and formatting
- Timezone handling
- Time range calculations
- Market hours and trading sessions
"""

import logging
import re
from datetime import UTC, datetime, timedelta

import pytz

logger = logging.getLogger(__name__)


class TimeUtilsError(Exception):
    """Custom exception for time utility errors."""
    pass


def get_utc_now() -> datetime:
    """
    Get current UTC time.

    Returns:
        Current UTC datetime
    """
    return datetime.now(UTC)


def parse_timestamp(
    timestamp: str | int | float | datetime,
    timezone_name: str = 'UTC',
    default_timezone: str = 'UTC'
) -> datetime:
    """
    Parse timestamp from various formats to timezone-aware datetime.

    Args:
        timestamp: Timestamp to parse
        timezone_name: Target timezone for output
        default_timezone: Default timezone if input is naive

    Returns:
        Timezone-aware datetime in specified timezone

    Raises:
        TimeUtilsError: If parsing fails
    """
    try:
        if isinstance(timestamp, datetime):
            dt = timestamp
        elif isinstance(timestamp, int | float):
            # Assume Unix timestamp (seconds since epoch)
            if timestamp > 1e10:  # Likely milliseconds
                timestamp = timestamp / 1000
            dt = datetime.fromtimestamp(timestamp, tz=UTC)
        elif isinstance(timestamp, str):
            # Try various string formats
            dt = _parse_timestamp_string(timestamp)
        else:
            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")

        # Handle timezone
        if dt.tzinfo is None:
            # Naive datetime, assume default timezone
            default_tz = pytz.timezone(default_timezone)
            dt = default_tz.localize(dt)

        # Convert to target timezone
        target_tz = pytz.timezone(timezone_name)
        dt = dt.astimezone(target_tz)

        return dt

    except Exception as e:
        error_msg = f"Failed to parse timestamp {timestamp}: {e}"
        logger.error(error_msg)
        raise TimeUtilsError(error_msg) from e


def _parse_timestamp_string(timestamp_str: str) -> datetime:
    """
    Parse timestamp string in various formats.

    Args:
        timestamp_str: Timestamp string

    Returns:
        Parsed datetime

    Raises:
        TimeUtilsError: If parsing fails
    """
    # Common timestamp formats
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%d',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y'
    ]

    # Try each format
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    # Try ISO format with regex for more flexible parsing
    iso_pattern = r'(\d{4})-(\d{2})-(\d{2})T?(\d{2}):?(\d{2}):?(\d{2})(?:\.(\d+))?(?:Z|([+-]\d{2}:?\d{2}))?'
    match = re.match(iso_pattern, timestamp_str)

    if match:
        groups = match.groups()
        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
        hour = int(groups[3]) if groups[3] else 0
        minute = int(groups[4]) if groups[4] else 0
        second = int(groups[5]) if groups[5] else 0
        microsecond = int(groups[6]) if groups[6] else 0

        dt = datetime(year, month, day, hour, minute, second, microsecond)

        # Handle timezone offset
        if groups[7]:  # Timezone offset
            offset_str = groups[7].replace(':', '')
            hours = int(offset_str[:3])
            minutes = int(offset_str[3:])
            offset = timedelta(hours=hours, minutes=minutes)
            dt = dt - offset
            dt = dt.replace(tzinfo=UTC)

        return dt

    raise TimeUtilsError(f"Could not parse timestamp string: {timestamp_str}")


def format_timestamp(
    dt: datetime,
    format_str: str = 'iso',
    timezone_name: str = 'UTC'
) -> str:
    """
    Format datetime to string.

    Args:
        dt: Datetime to format
        format_str: Output format ('iso', 'human', 'short', or custom format)
        timezone_name: Target timezone for output

    Returns:
        Formatted timestamp string

    Raises:
        TimeUtilsError: If formatting fails
    """
    try:
        # Convert to target timezone
        if dt.tzinfo is None:
            # Assume UTC if naive
            dt = dt.replace(tzinfo=UTC)

        target_tz = pytz.timezone(timezone_name)
        dt = dt.astimezone(target_tz)

        # Apply format
        if format_str == 'iso':
            return dt.isoformat()
        elif format_str == 'human':
            return dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        elif format_str == 'short':
            return dt.strftime('%Y-%m-%d %H:%M')
        else:
            return dt.strftime(format_str)

    except Exception as e:
        error_msg = f"Failed to format timestamp {dt}: {e}"
        logger.error(error_msg)
        raise TimeUtilsError(error_msg) from e


def get_time_range(
    start_time: str | datetime,
    end_time: str | datetime,
    interval_minutes: int = 1,
    timezone_name: str = 'UTC'
) -> list[datetime]:
    """
    Generate list of timestamps within a time range.

    Args:
        start_time: Start time
        end_time: End time
        interval_minutes: Interval between timestamps in minutes
        timezone_name: Timezone for output

    Returns:
        List of datetime objects
    """
    try:
        start_dt = parse_timestamp(start_time, timezone_name)
        end_dt = parse_timestamp(end_time, timezone_name)

        if start_dt >= end_dt:
            raise TimeUtilsError("Start time must be before end time")

        timestamps = []
        current_dt = start_dt

        while current_dt <= end_dt:
            timestamps.append(current_dt)
            current_dt += timedelta(minutes=interval_minutes)

        return timestamps

    except Exception as e:
        error_msg = f"Failed to generate time range: {e}"
        logger.error(error_msg)
        raise TimeUtilsError(error_msg) from e


def is_market_open(
    timestamp: str | datetime,
    market: str = 'crypto',
    timezone_name: str = 'UTC'
) -> bool:
    """
    Check if market is open at given timestamp.

    Args:
        timestamp: Timestamp to check
        market: Market type ('crypto', 'forex', 'stocks')
        timezone_name: Timezone for timestamp

    Returns:
        True if market is open, False otherwise
    """
    try:
        dt = parse_timestamp(timestamp, timezone_name)

        if market == 'crypto':
            # Crypto markets are always open
            return True
        elif market == 'forex':
            # Forex is closed on weekends
            return dt.weekday() < 5
        elif market == 'stocks':
            # Stock markets have specific hours (simplified)
            if dt.weekday() >= 5:  # Weekend
                return False

            # US market hours (9:30 AM - 4:00 PM EST)
            est_tz = pytz.timezone('US/Eastern')
            dt_est = dt.astimezone(est_tz)

            market_open = dt_est.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = dt_est.replace(hour=16, minute=0, second=0, microsecond=0)

            return market_open <= dt_est <= market_close
        else:
            logger.warning(f"Unknown market type: {market}, assuming open")
            return True

    except Exception as e:
        logger.error(f"Failed to check market status: {e}")
        return True  # Assume open on error


def get_next_market_open(
    timestamp: str | datetime,
    market: str = 'crypto',
    timezone_name: str = 'UTC'
) -> datetime:
    """
    Get next market open time.

    Args:
        timestamp: Current timestamp
        market: Market type
        timezone_name: Timezone for timestamp

    Returns:
        Next market open datetime
    """
    try:
        dt = parse_timestamp(timestamp, timezone_name)

        if market == 'crypto':
            # Crypto is always open, return current time
            return dt
        elif market == 'stocks':
            # Find next weekday at 9:30 AM EST
            est_tz = pytz.timezone('US/Eastern')
            dt_est = dt.astimezone(est_tz)

            # If it's weekend, move to next Monday
            while dt_est.weekday() >= 5:
                dt_est += timedelta(days=1)

            # Set to 9:30 AM
            next_open = dt_est.replace(hour=9, minute=30, second=0, microsecond=0)

            # If current time is after market open today, move to next day
            if dt_est.time() >= next_open.time():
                next_open += timedelta(days=1)
                # Handle weekend
                while next_open.weekday() >= 5:
                    next_open += timedelta(days=1)

            return next_open.astimezone(pytz.timezone(timezone_name))
        else:
            # For other markets, return current time
            return dt

    except Exception as e:
        logger.error(f"Failed to get next market open: {e}")
        return dt


def calculate_time_difference(
    time1: str | datetime,
    time2: str | datetime,
    timezone_name: str = 'UTC',
    unit: str = 'minutes'
) -> float:
    """
    Calculate time difference between two timestamps.

    Args:
        time1: First timestamp
        time2: Second timestamp
        timezone_name: Timezone for calculation
        unit: Output unit ('seconds', 'minutes', 'hours', 'days')

    Returns:
        Time difference in specified unit
    """
    try:
        dt1 = parse_timestamp(time1, timezone_name)
        dt2 = parse_timestamp(time2, timezone_name)

        diff = abs(dt2 - dt1)

        # Convert to requested unit
        if unit == 'seconds':
            return diff.total_seconds()
        elif unit == 'minutes':
            return diff.total_seconds() / 60
        elif unit == 'hours':
            return diff.total_seconds() / 3600
        elif unit == 'days':
            return diff.total_seconds() / 86400
        else:
            raise ValueError(f"Unsupported unit: {unit}")

    except Exception as e:
        error_msg = f"Failed to calculate time difference: {e}"
        logger.error(error_msg)
        raise TimeUtilsError(error_msg) from e


def round_timestamp(
    timestamp: str | datetime,
    interval_minutes: int = 1,
    timezone_name: str = 'UTC'
) -> datetime:
    """
    Round timestamp to nearest interval.

    Args:
        timestamp: Timestamp to round
        interval_minutes: Interval in minutes
        timezone_name: Timezone for timestamp

    Returns:
        Rounded datetime
    """
    try:
        dt = parse_timestamp(timestamp, timezone_name)

        # Convert to minutes since epoch
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        minutes_since_epoch = int((dt - epoch).total_seconds() / 60)

        # Round to nearest interval
        rounded_minutes = round(minutes_since_epoch / interval_minutes) * interval_minutes

        # Convert back to datetime
        rounded_dt = epoch + timedelta(minutes=rounded_minutes)

        # Convert to target timezone
        target_tz = pytz.timezone(timezone_name)
        return rounded_dt.astimezone(target_tz)

    except Exception as e:
        error_msg = f"Failed to round timestamp: {e}"
        logger.error(error_msg)
        raise TimeUtilsError(error_msg) from e
