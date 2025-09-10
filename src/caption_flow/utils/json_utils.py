"""JSON serialization utilities for handling special types like datetime."""

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely serialize objects to JSON, handling special types.

    Args:
    ----
        obj: Object to serialize
        **kwargs: Additional arguments to pass to json.dumps

    Returns:
    -------
        JSON string representation

    """
    return json.dumps(obj, default=json_serializer, **kwargs)


def safe_dict(obj: Any) -> Dict[str, Any]:
    """Convert an object to a dictionary, handling special types.

    Args:
    ----
        obj: Object to convert (dataclass, dict, etc.)

    Returns:
    -------
        Dictionary with JSON-serializable values

    """
    if is_dataclass(obj):
        data = asdict(obj)
    elif hasattr(obj, "__dict__"):
        data = obj.__dict__.copy()
    elif isinstance(obj, dict):
        data = obj.copy()
    else:
        return obj

    return sanitize_dict(data)


def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize a dictionary to ensure all values are JSON-serializable.

    Args:
    ----
        data: Dictionary to sanitize

    Returns:
    -------
        Sanitized dictionary

    """
    result = {}

    for key, value in data.items():
        if value is None:
            result[key] = None
        elif isinstance(value, (datetime, date)):
            result[key] = value.isoformat()
        elif isinstance(value, Decimal):
            result[key] = float(value)
        elif isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, Enum):
            result[key] = value.value
        elif isinstance(value, (list, tuple)):
            result[key] = [sanitize_value(item) for item in value]
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif is_dataclass(value):
            result[key] = sanitize_dict(asdict(value))
        elif hasattr(value, "__dict__"):
            result[key] = sanitize_dict(value.__dict__)
        else:
            result[key] = value

    return result


def sanitize_value(value: Any) -> Any:
    """Sanitize a single value for JSON serialization.

    Args:
    ----
        value: Value to sanitize

    Returns:
    -------
        JSON-serializable value

    """
    if value is None:
        return None
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, Decimal):
        return float(value)
    elif isinstance(value, Path):
        return str(value)
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, dict):
        return sanitize_dict(value)
    elif isinstance(value, (list, tuple)):
        return [sanitize_value(item) for item in value]
    elif is_dataclass(value):
        return sanitize_dict(asdict(value))
    elif hasattr(value, "__dict__"):
        return sanitize_dict(value.__dict__)
    else:
        return value


def json_serializer(obj: Any) -> Any:
    """Default JSON serializer for special types.

    Args:
    ----
        obj: Object to serialize

    Returns:
    -------
        JSON-serializable representation

    Raises:
    ------
        TypeError: If object type is not supported

    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif type(obj).__name__ == "int64":
        return int(obj)
    elif is_dataclass(obj):
        return sanitize_dict(asdict(obj))
    elif hasattr(obj, "__dict__"):
        return sanitize_dict(obj.__dict__)
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def parse_datetime(dt_string: Union[str, datetime, None]) -> Union[datetime, None]:
    """Parse a datetime string or return existing datetime.

    Args:
    ----
        dt_string: ISO format datetime string, datetime object, or None

    Returns:
    -------
        datetime object or None

    """
    if dt_string is None:
        return None
    elif isinstance(dt_string, datetime):
        return dt_string
    elif isinstance(dt_string, str):
        try:
            return datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
        except ValueError:
            # Try parsing without timezone
            return datetime.fromisoformat(dt_string)
    else:
        raise ValueError(f"Cannot parse datetime from {type(dt_string).__name__}")


# Convenience functions for common use cases
def to_json_dict(obj: Any) -> Dict[str, Any]:
    """Convert any object to a JSON-serializable dictionary.

    This is a convenience wrapper around safe_dict.

    Args:
    ----
        obj: Object to convert

    Returns:
    -------
        JSON-serializable dictionary

    """
    return safe_dict(obj)


def to_json_string(obj: Any, indent: int = None) -> str:
    """Convert any object to a JSON string.

    This is a convenience wrapper around safe_json_dumps.

    Args:
    ----
        obj: Object to convert
        indent: Number of spaces for indentation (None for compact)

    Returns:
    -------
        JSON string

    """
    return safe_json_dumps(obj, indent=indent)
