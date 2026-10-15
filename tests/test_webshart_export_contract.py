"""Exercise structured exports against the installed Webshart writer."""

import copy
import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from caption_flow.models import StorageContents
from caption_flow.storage.exporter import ExportError, StorageExporter


def exporter_for(rows, *, columns=None, output_fields=None):
    return StorageExporter(
        StorageContents(
            rows=rows,
            columns=columns if columns is not None else ["filename", "captions"],
            output_fields=output_fields if output_fields is not None else ["captions"],
            total_rows=len(rows),
            metadata={},
        )
    )


@pytest.fixture
def metadata_file(tmp_path):
    path = tmp_path / "shard.json"
    path.write_text(
        json.dumps(
            {
                "filesize": 1024,
                "files": {
                    "sample.jpg": {
                        "offset": 512,
                        "length": 32,
                        "captions": ["old caption"],
                        "metadata": {"width": 64, "height": 32},
                    },
                    "other.jpg": {"offset": 1024, "length": 24, "captions": ["untouched"]},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "value,expected",
    [
        ('{unfinished "caption"', ['{unfinished "caption"']),
        ("[a sign on a wall]", ["[a sign on a wall]"]),
        ("[1,2]", ["[1,2]"]),
        ('[{"description":"tree"},null]', ['[{"description":"tree"},null]']),
        ('  {"description":"café"}', [{"description": "café"}]),
        ('[{"description":"tree"},"alternate"]', [{"description": "tree"}, "alternate"]),
        ([None, "", "a tree", None], ["a tree"]),
        ((None, {"description": "tree"}, "alternate"), [{"description": "tree"}, "alternate"]),
        ([["first"], [None, ["second"]]], ["first", "second"]),
        ({}, [{}]),
    ],
)
def test_normalized_export_preserves_metadata_and_input(metadata_file, value, expected):
    before = json.loads(metadata_file.read_text(encoding="utf-8"))
    rows = [{"filename": "sample.jpg", "captions": value}]
    original_rows = copy.deepcopy(rows)

    assert exporter_for(rows).to_webshart_metadata(metadata_file) == 1

    after = json.loads(metadata_file.read_text(encoding="utf-8"))
    before["files"]["sample.jpg"]["captions"] = expected
    assert after == before
    assert rows == original_rows


def test_structured_export_serializes_nested_scientific_and_datetime_values(metadata_file):
    value = {
        "bbox": np.array([1, 2, 3, 4]),
        "width": np.int64(64),
        "confidence": np.float32(0.5),
        "nested": [{"timestamp": datetime(2026, 1, 2, tzinfo=timezone.utc)}],
    }
    exporter = exporter_for([{"filename": "sample.jpg", "captions": value}])
    assert exporter.to_webshart_metadata(metadata_file) == 1
    assert json.loads(metadata_file.read_text())["files"]["sample.jpg"]["captions"] == [
        {
            "bbox": [1, 2, 3, 4],
            "width": 64,
            "confidence": 0.5,
            "nested": [{"timestamp": "2026-01-02T00:00:00+00:00"}],
        }
    ]


@pytest.mark.parametrize("value", [None, "", [], (), [None, ""], "[]"])
def test_empty_captions_do_not_erase_existing_metadata(metadata_file, value, caplog):
    before = metadata_file.read_bytes()
    exporter = exporter_for([{"filename": "sample.jpg", "captions": value}])
    assert exporter.to_webshart_metadata(metadata_file) == 0
    assert metadata_file.read_bytes() == before
    assert "no captions content" in caplog.text


def test_unidentifiable_rows_are_skipped_without_creating_metadata(tmp_path, caplog):
    exporter = exporter_for([{"captions": "A tree."}])
    path = tmp_path / "missing.json"
    assert exporter.to_webshart_metadata(path) == 0
    assert not path.exists()
    assert "no extractable filename" in caplog.text


@pytest.mark.parametrize(
    "identity",
    [
        {"item_key": "sample"},
        {"url": "https://example.com/images/sample.jpg?download=1"},
        {"custom_filename": "sample.jpg"},
    ],
)
def test_custom_output_field_and_filename_fallbacks(metadata_file, identity):
    exporter = exporter_for(
        [{**identity, "json_label": '{"description":"a tree"}'}],
        columns=list(identity),
        output_fields=["json_label"],
    )
    assert (
        exporter.to_webshart_metadata(
            metadata_file, filename_column="custom_filename", export_column="json_label"
        )
        == 1
    )
    assert json.loads(metadata_file.read_text())["files"]["sample.jpg"]["captions"] == [
        {"description": "a tree"}
    ]


def test_missing_output_column_fails_before_modifying_metadata(metadata_file):
    before = metadata_file.read_bytes()
    exporter = exporter_for([{"filename": "sample.jpg", "captions": "A tree."}])
    with pytest.raises(ExportError, match="Column 'missing' not found"):
        exporter.to_webshart_metadata(metadata_file, export_column="missing")
    assert metadata_file.read_bytes() == before


@pytest.mark.parametrize("value", [123, False, 1.5, ["valid caption", 123]])
def test_invalid_captions_fail_before_modifying_metadata(metadata_file, value):
    before = metadata_file.read_bytes()
    exporter = exporter_for([{"filename": "sample.jpg", "captions": value}])
    with pytest.raises(ExportError, match="Column 'captions' must contain"):
        exporter.to_webshart_metadata(metadata_file)
    assert metadata_file.read_bytes() == before


def test_directory_is_not_accepted_as_a_metadata_file(tmp_path):
    exporter = exporter_for([{"filename": "sample.jpg", "captions": "A tree."}])
    with pytest.raises(ExportError, match="not a file"):
        exporter.to_webshart_metadata(tmp_path)


@pytest.mark.parametrize(
    "module,message",
    [(None, "webshart is required"), (SimpleNamespace(), "upgrade webshart")],
)
def test_missing_webshart_api_reports_actionable_error(metadata_file, monkeypatch, module, message):
    before = metadata_file.read_bytes()
    monkeypatch.setitem(sys.modules, "webshart", module)
    exporter = exporter_for([{"filename": "sample.jpg", "captions": "A tree."}])
    with pytest.raises(ExportError, match=message):
        exporter.to_webshart_metadata(metadata_file)
    assert metadata_file.read_bytes() == before


def test_writer_failure_propagates_instead_of_reporting_success(metadata_file, monkeypatch):
    writer = Mock(side_effect=OSError("metadata write failed"))
    monkeypatch.setitem(sys.modules, "webshart", SimpleNamespace(write_captions_to_metadata=writer))
    exporter = exporter_for([{"filename": "sample.jpg", "captions": "A tree."}])
    with pytest.raises(OSError, match="metadata write failed"):
        exporter.to_webshart_metadata(metadata_file)
    writer.assert_called_once_with(metadata_file, {"sample.jpg": ["A tree."]})
