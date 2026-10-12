"""Malformed model Unicode must not poison subsequent durable caption writes."""

import json
from datetime import datetime, timezone

import lance
import pytest

from caption_flow.storage import StorageManager
from caption_flow.utils.output_policy import OutputPolicy
from caption_flow.utils.unicode_utils import escape_surrogates, utf8_safe_values


@pytest.mark.parametrize("text", ["", "plain", "café 日本語 🌍", r"already \udc80"])
def test_valid_utf8_is_unchanged(text):
    assert escape_surrogates(text) == text


@pytest.mark.parametrize("codepoint", [0xD800, 0xDBFF, 0xDC00, 0xDC80, 0xDFFF])
def test_surrogates_are_preserved_as_literal_escapes(codepoint):
    assert escape_surrogates(chr(codepoint)) == f"\\u{codepoint:04x}"


def test_recursive_values_preserve_types_and_do_not_mutate_input():
    timestamp = datetime.now(timezone.utc)
    value = {"nested": ["\udc80", ("café", "\ud800")], "timestamp": timestamp, "n": 3}
    assert utf8_safe_values(value) == {
        "nested": [r"\udc80", ("café", r"\ud800")],
        "timestamp": timestamp,
        "n": 3,
    }
    assert value["nested"][0] == "\udc80"


@pytest.mark.parametrize("canonicalize", [False, True])
def test_json_output_remains_parse_equivalent_and_utf8_encodable(canonicalize):
    document = {"text": "\udc80", "\ud800": "café 🌍"}
    policy = OutputPolicy(validate_json_output=True, canonicalize_json_output=canonicalize)
    result = policy.process(json.dumps(document, ensure_ascii=False))
    assert result is not None
    assert json.loads(result) == document
    result.encode("utf-8")


def test_plain_output_escapes_surrogates():
    assert OutputPolicy().process("A \udc80 sign") == r"A \udc80 sign"


@pytest.mark.asyncio
async def test_flush_surrogate_buffer_then_keep_writing(tmp_path):
    storage = StorageManager(tmp_path)
    await storage.initialize()
    try:
        caption = json.dumps({"text": "\udc80", "valid": "café 🌍"}, ensure_ascii=False)
        storage.known_output_fields.add("captions")
        storage.caption_buffer.append({"job_id": "00000:chunk:0:sample:0", "captions": [caption]})
        await storage._flush_captions()
        assert not storage.caption_buffer
        stored = lance.dataset(str(storage.captions_path)).to_table().to_pylist()
        assert len(stored) == 1
        assert json.loads(stored[0]["captions"][0]) == json.loads(caption)
        storage.caption_buffer.append(
            {"job_id": "00000:chunk:0:sample:1", "captions": ["Next image."]}
        )
        await storage._flush_captions()
        assert not storage.caption_buffer
        assert lance.dataset(str(storage.captions_path)).count_rows() == 2
    finally:
        await storage.close()
