"""Lossless escaping of surrogate code points at UTF-8 storage boundaries."""

import re
from typing import Any

_SURROGATES = re.compile("[\ud800-\udfff]")


def escape_surrogates(text: str) -> str:
    r"""Escape unencodable code points as literal \uXXXX, preserving valid Unicode.

    Python's JSON decoder accepts lone surrogate escapes. Canonicalizing those
    strings with ensure_ascii=False creates text that Arrow cannot encode. An
    escaped spelling preserves the code point without replacing or dropping it;
    inside a JSON caption it also preserves the parsed document's value.
    """
    return _SURROGATES.sub(lambda match: f"\\u{ord(match.group()):04x}", text)


def utf8_safe_values(value: Any) -> Any:
    """Copy row values with safe strings; schema field names are unchanged."""
    if isinstance(value, str):
        return escape_surrogates(value)
    if isinstance(value, list):
        return [utf8_safe_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(utf8_safe_values(item) for item in value)
    if isinstance(value, dict):
        return {key: utf8_safe_values(item) for key, item in value.items()}
    return value
