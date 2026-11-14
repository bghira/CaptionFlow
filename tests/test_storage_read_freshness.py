"""Exports must include captions appended after an earlier storage read."""

from datetime import datetime

import pytest

from caption_flow.models import Caption
from caption_flow.storage import StorageManager


@pytest.mark.asyncio
async def test_repeated_exports_include_appends_without_schema_changes(tmp_path):
    storage = StorageManager(tmp_path)
    await storage.initialize()
    try:
        for index in range(3):
            await storage.save_caption(
                Caption(
                    job_id=f"job-{index}",
                    dataset="test",
                    shard="shard",
                    item_key=str(index),
                    contributor_id="tester",
                    timestamp=datetime.now(),
                    captions=[f"Caption {index}"],
                    outputs={"captions": [f"Caption {index}"]},
                )
            )
            contents = await storage.get_shard_contents("shard")
            assert {row["job_id"] for row in contents.rows} == {
                f"job-{i}" for i in range(index + 1)
            }
    finally:
        await storage.close()
        for connection in storage.duckdb_shard_connections.values():
            connection.close()
