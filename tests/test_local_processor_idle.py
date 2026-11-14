"""An exhausted local dataset must not leave its producer spinning."""

import threading
from unittest.mock import Mock

from caption_flow.processors.local_filesystem import LocalFilesystemOrchestratorProcessor


def test_exhausted_dataset_waits_and_stops_promptly():
    processor = LocalFilesystemOrchestratorProcessor()
    processor.min_buffer = 1
    processor.buffer_multiplier = 1
    entered_wait = threading.Event()
    original_wait = processor.stop_creation.wait

    def wait_for_shutdown(timeout=None):
        entered_wait.set()
        return original_wait(timeout)

    processor.stop_creation.wait = Mock(side_effect=wait_for_shutdown)
    producer = threading.Thread(target=processor._create_units_background, daemon=True)
    producer.start()
    try:
        assert entered_wait.wait(timeout=1), "Exhausted datasets must wait instead of busy-looping"
    finally:
        processor.stop_creation.set()
        producer.join(timeout=1)
    assert not producer.is_alive()
