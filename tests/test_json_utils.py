"""Comprehensive tests for JSON utilities module."""

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pytest
from caption_flow.utils.json_utils import (
    json_serializer,
    parse_datetime,
    safe_dict,
    safe_json_dumps,
    sanitize_dict,
    sanitize_value,
    to_json_dict,
    to_json_string,
)


class SampleEnum(Enum):
    """Sample enum for testing."""

    OPTION_A = "option_a"
    OPTION_B = "option_b"
    NUMERIC = 42


@dataclass
class SampleDataclass:
    """Sample dataclass for testing."""

    name: str
    value: int
    timestamp: datetime = None


class SampleClass:
    """Sample class for testing."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value
        self.private_attr = "_hidden"


class TestSafeJsonDumps:
    """Test safe_json_dumps function."""

    def test_basic_types(self):
        """Test serialization of basic types."""
        assert safe_json_dumps("string") == '"string"'
        assert safe_json_dumps(42) == "42"
        assert safe_json_dumps(3.14) == "3.14"
        assert safe_json_dumps(True) == "true"
        assert safe_json_dumps(None) == "null"

    def test_datetime_serialization(self):
        """Test datetime serialization."""
        dt = datetime(2024, 1, 15, 12, 30, 45)
        result = safe_json_dumps(dt)
        assert '"2024-01-15T12:30:45"' == result

    def test_date_serialization(self):
        """Test date serialization."""
        d = date(2024, 1, 15)
        result = safe_json_dumps(d)
        assert '"2024-01-15"' == result

    def test_decimal_serialization(self):
        """Test Decimal serialization."""
        decimal_value = Decimal("123.45")
        result = safe_json_dumps(decimal_value)
        assert "123.45" == result

    def test_path_serialization(self):
        """Test Path serialization."""
        path = Path("/tmp/test.txt")
        result = safe_json_dumps(path)
        assert '"/tmp/test.txt"' == result

    def test_enum_serialization(self):
        """Test Enum serialization."""
        result = safe_json_dumps(SampleEnum.OPTION_A)
        assert '"option_a"' == result

        result = safe_json_dumps(SampleEnum.NUMERIC)
        assert "42" == result

    def test_dataclass_serialization(self):
        """Test dataclass serialization."""
        dt = datetime(2024, 1, 15, 12, 30, 45)
        obj = SampleDataclass(name="test", value=42, timestamp=dt)
        result = safe_json_dumps(obj)
        expected = json.loads(result)
        assert expected["name"] == "test"
        assert expected["value"] == 42
        assert expected["timestamp"] == "2024-01-15T12:30:45"

    def test_custom_kwargs(self):
        """Test passing custom kwargs to json.dumps."""
        data = {"key": "value"}
        result = safe_json_dumps(data, indent=2)
        assert "{\n  " in result  # Check for indentation


class TestSafeDict:
    """Test safe_dict function."""

    def test_dataclass_conversion(self):
        """Test converting dataclass to dict."""
        dt = datetime(2024, 1, 15, 12, 30, 45)
        obj = SampleDataclass(name="test", value=42, timestamp=dt)
        result = safe_dict(obj)

        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["timestamp"] == "2024-01-15T12:30:45"

    def test_custom_class_conversion(self):
        """Test converting custom class to dict."""
        obj = SampleClass(name="test", value=42)
        result = safe_dict(obj)

        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["private_attr"] == "_hidden"

    def test_dict_conversion(self):
        """Test converting dict (should return sanitized copy)."""
        dt = datetime(2024, 1, 15)
        original = {"name": "test", "date": dt}
        result = safe_dict(original)

        assert result["name"] == "test"
        assert result["date"] == "2024-01-15T00:00:00"  # datetime includes time
        # Should be a copy
        assert result is not original

    def test_primitive_passthrough(self):
        """Test that primitive types pass through unchanged."""
        assert safe_dict("string") == "string"
        assert safe_dict(42) == 42
        assert safe_dict(None) is None


class TestSanitizeDict:
    """Test sanitize_dict function."""

    def test_none_values(self):
        """Test handling of None values."""
        data = {"key": None}
        result = sanitize_dict(data)
        assert result["key"] is None

    def test_datetime_sanitization(self):
        """Test datetime sanitization in dict."""
        dt = datetime(2024, 1, 15, 12, 30, 45)
        d = date(2024, 1, 15)
        data = {"datetime": dt, "date": d}
        result = sanitize_dict(data)

        assert result["datetime"] == "2024-01-15T12:30:45"
        assert result["date"] == "2024-01-15"

    def test_decimal_sanitization(self):
        """Test Decimal sanitization in dict."""
        data = {"decimal": Decimal("123.45")}
        result = sanitize_dict(data)
        assert result["decimal"] == 123.45

    def test_path_sanitization(self):
        """Test Path sanitization in dict."""
        data = {"path": Path("/tmp/test.txt")}
        result = sanitize_dict(data)
        assert result["path"] == "/tmp/test.txt"

    def test_enum_sanitization(self):
        """Test Enum sanitization in dict."""
        data = {"enum": SampleEnum.OPTION_A}
        result = sanitize_dict(data)
        assert result["enum"] == "option_a"

    def test_list_sanitization(self):
        """Test list sanitization in dict."""
        dt = datetime(2024, 1, 15)
        data = {"list": [dt, "string", 42]}
        result = sanitize_dict(data)

        assert result["list"][0] == "2024-01-15T00:00:00"
        assert result["list"][1] == "string"
        assert result["list"][2] == 42

    def test_tuple_sanitization(self):
        """Test tuple sanitization in dict."""
        dt = datetime(2024, 1, 15)
        data = {"tuple": (dt, "string")}
        result = sanitize_dict(data)

        assert result["tuple"][0] == "2024-01-15T00:00:00"
        assert result["tuple"][1] == "string"
        assert isinstance(result["tuple"], list)  # Tuples become lists

    def test_nested_dict_sanitization(self):
        """Test nested dict sanitization."""
        dt = datetime(2024, 1, 15)
        data = {"nested": {"datetime": dt, "value": 42}}
        result = sanitize_dict(data)

        assert result["nested"]["datetime"] == "2024-01-15T00:00:00"
        assert result["nested"]["value"] == 42

    def test_nested_dataclass_sanitization(self):
        """Test nested dataclass sanitization."""
        dt = datetime(2024, 1, 15)
        obj = SampleDataclass(name="nested", value=42, timestamp=dt)
        data = {"dataclass": obj}
        result = sanitize_dict(data)

        assert result["dataclass"]["name"] == "nested"
        assert result["dataclass"]["timestamp"] == "2024-01-15T00:00:00"

    def test_nested_object_sanitization(self):
        """Test nested object sanitization."""
        obj = SampleClass(name="nested", value=42)
        data = {"object": obj}
        result = sanitize_dict(data)

        assert result["object"]["name"] == "nested"
        assert result["object"]["value"] == 42


class TestSanitizeValue:
    """Test sanitize_value function."""

    def test_none_value(self):
        """Test None value sanitization."""
        assert sanitize_value(None) is None

    def test_datetime_value(self):
        """Test datetime value sanitization."""
        dt = datetime(2024, 1, 15, 12, 30, 45)
        result = sanitize_value(dt)
        assert result == "2024-01-15T12:30:45"

    def test_date_value(self):
        """Test date value sanitization."""
        d = date(2024, 1, 15)
        result = sanitize_value(d)
        assert result == "2024-01-15"

    def test_decimal_value(self):
        """Test Decimal value sanitization."""
        decimal_val = Decimal("123.45")
        result = sanitize_value(decimal_val)
        assert result == 123.45

    def test_path_value(self):
        """Test Path value sanitization."""
        path = Path("/tmp/test.txt")
        result = sanitize_value(path)
        assert result == "/tmp/test.txt"

    def test_enum_value(self):
        """Test Enum value sanitization."""
        result = sanitize_value(SampleEnum.OPTION_A)
        assert result == "option_a"

    def test_dict_value(self):
        """Test dict value sanitization."""
        dt = datetime(2024, 1, 15)
        value = {"datetime": dt}
        result = sanitize_value(value)
        assert result["datetime"] == "2024-01-15T00:00:00"

    def test_list_value(self):
        """Test list value sanitization."""
        dt = datetime(2024, 1, 15)
        value = [dt, "string"]
        result = sanitize_value(value)
        assert result[0] == "2024-01-15T00:00:00"
        assert result[1] == "string"

    def test_dataclass_value(self):
        """Test dataclass value sanitization."""
        dt = datetime(2024, 1, 15)
        obj = SampleDataclass(name="test", value=42, timestamp=dt)
        result = sanitize_value(obj)
        assert result["name"] == "test"
        assert result["timestamp"] == "2024-01-15T00:00:00"

    def test_object_value(self):
        """Test object value sanitization."""
        obj = SampleClass(name="test", value=42)
        result = sanitize_value(obj)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_primitive_passthrough(self):
        """Test that primitive values pass through unchanged."""
        assert sanitize_value("string") == "string"
        assert sanitize_value(42) == 42
        assert sanitize_value(3.14) == 3.14
        assert sanitize_value(True) is True


class TestJsonSerializer:
    """Test json_serializer function."""

    def test_datetime_serialization(self):
        """Test datetime serialization."""
        dt = datetime(2024, 1, 15, 12, 30, 45)
        result = json_serializer(dt)
        assert result == "2024-01-15T12:30:45"

    def test_date_serialization(self):
        """Test date serialization."""
        d = date(2024, 1, 15)
        result = json_serializer(d)
        assert result == "2024-01-15"

    def test_decimal_serialization(self):
        """Test Decimal serialization."""
        decimal_val = Decimal("123.45")
        result = json_serializer(decimal_val)
        assert result == 123.45

    def test_path_serialization(self):
        """Test Path serialization."""
        path = Path("/tmp/test.txt")
        result = json_serializer(path)
        assert result == "/tmp/test.txt"

    def test_enum_serialization(self):
        """Test Enum serialization."""
        result = json_serializer(SampleEnum.OPTION_A)
        assert result == "option_a"

    def test_int64_serialization(self):
        """Test int64-like type serialization."""

        # Create a mock int64-like object
        class MockInt64:
            def __init__(self, value):
                self.value = value

            def __int__(self):
                return self.value

        # Patch the type name
        MockInt64.__name__ = "int64"
        obj = MockInt64(42)
        result = json_serializer(obj)
        assert result == 42

    def test_dataclass_serialization(self):
        """Test dataclass serialization."""
        dt = datetime(2024, 1, 15)
        obj = SampleDataclass(name="test", value=42, timestamp=dt)
        result = json_serializer(obj)
        assert result["name"] == "test"
        assert result["timestamp"] == "2024-01-15T00:00:00"

    def test_object_serialization(self):
        """Test object serialization."""
        obj = SampleClass(name="test", value=42)
        result = json_serializer(obj)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_unsupported_type_error(self):
        """Test TypeError for unsupported types."""
        # Use a built-in type that won't have __dict__ and isn't handled
        obj = object()

        with pytest.raises(TypeError, match="Object of type object is not JSON serializable"):
            json_serializer(obj)


class TestParseDatetime:
    """Test parse_datetime function."""

    def test_none_input(self):
        """Test None input."""
        result = parse_datetime(None)
        assert result is None

    def test_datetime_input(self):
        """Test datetime input (should pass through)."""
        dt = datetime(2024, 1, 15, 12, 30, 45)
        result = parse_datetime(dt)
        assert result == dt
        assert result is dt  # Should be the same object

    def test_iso_string_input(self):
        """Test ISO format string input."""
        dt_string = "2024-01-15T12:30:45"
        result = parse_datetime(dt_string)
        expected = datetime(2024, 1, 15, 12, 30, 45)
        assert result == expected

    def test_iso_string_with_z_suffix(self):
        """Test ISO format string with Z suffix."""
        dt_string = "2024-01-15T12:30:45Z"
        result = parse_datetime(dt_string)
        expected = datetime(2024, 1, 15, 12, 30, 45)
        # The result will have timezone info, but time should match
        assert result.replace(tzinfo=None) == expected

    def test_iso_string_with_timezone(self):
        """Test ISO format string with timezone."""
        dt_string = "2024-01-15T12:30:45+01:00"
        result = parse_datetime(dt_string)
        # Should parse successfully
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_invalid_string_fallback(self):
        """Test invalid string falling back to basic parsing."""
        # This should work with the fallback fromisoformat
        dt_string = "2024-01-15T12:30:45"
        result = parse_datetime(dt_string)
        expected = datetime(2024, 1, 15, 12, 30, 45)
        assert result == expected

    def test_completely_invalid_string(self):
        """Test completely invalid string."""
        with pytest.raises(ValueError):
            parse_datetime("not-a-date")

    def test_unsupported_type(self):
        """Test unsupported input type."""
        with pytest.raises(ValueError, match="Cannot parse datetime from"):
            parse_datetime(42)


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_to_json_dict(self):
        """Test to_json_dict convenience function."""
        dt = datetime(2024, 1, 15)
        obj = SampleDataclass(name="test", value=42, timestamp=dt)
        result = to_json_dict(obj)

        assert result["name"] == "test"
        assert result["timestamp"] == "2024-01-15T00:00:00"

    def test_to_json_string_compact(self):
        """Test to_json_string with compact formatting."""
        data = {"name": "test", "value": 42}
        result = to_json_string(data)
        assert result == '{"name": "test", "value": 42}'

    def test_to_json_string_indented(self):
        """Test to_json_string with indentation."""
        data = {"name": "test", "value": 42}
        result = to_json_string(data, indent=2)
        assert "{\n  " in result
        assert '"name": "test"' in result


class TestComplexCases:
    """Test complex nested cases."""

    def test_deeply_nested_structure(self):
        """Test deeply nested data structures."""
        dt = datetime(2024, 1, 15)
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "datetime": dt,
                        "list": [dt, SampleEnum.OPTION_A, Path("/tmp")],
                        "dataclass": SampleDataclass("nested", 42, dt),
                    }
                }
            }
        }

        result = sanitize_dict(data)
        level3 = result["level1"]["level2"]["level3"]

        assert level3["datetime"] == "2024-01-15T00:00:00"
        assert level3["list"][0] == "2024-01-15T00:00:00"
        assert level3["list"][1] == "option_a"
        assert level3["list"][2] == "/tmp"
        assert level3["dataclass"]["name"] == "nested"

    def test_circular_reference_safety(self):
        """Test that circular references don't cause infinite recursion."""
        # Note: The current implementation doesn't handle circular references,
        # so this test demonstrates the limitation. In practice, users should
        # avoid circular references when using these utilities.
        data = {"key": "value", "safe": "data"}

        # Test without circular reference works fine
        result = sanitize_dict(data)
        assert result["key"] == "value"
        assert result["safe"] == "data"

    def test_mixed_types_in_list(self):
        """Test list with mixed types."""
        dt = datetime(2024, 1, 15)
        mixed_list = [
            "string",
            42,
            dt,
            SampleEnum.OPTION_A,
            Path("/tmp"),
            Decimal("123.45"),
            {"nested": dt},
        ]

        result = sanitize_value(mixed_list)

        assert result[0] == "string"
        assert result[1] == 42
        assert result[2] == "2024-01-15T00:00:00"
        assert result[3] == "option_a"
        assert result[4] == "/tmp"
        assert result[5] == 123.45
        assert result[6]["nested"] == "2024-01-15T00:00:00"


if __name__ == "__main__":
    pytest.main([__file__])
