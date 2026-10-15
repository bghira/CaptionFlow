"""Bounded retry delays without treating failures as completed work."""

import heapq
import math
import time
from collections.abc import Callable


class RetryQueue:
    """Used under the processor lock; failed units never consume fresh-work capacity."""

    def __init__(
        self, initial: float = 5.0, maximum: float = 300.0, clock: Callable = time.monotonic
    ):
        if not (math.isfinite(initial) and math.isfinite(maximum) and 0 < initial <= maximum):
            raise ValueError("retry delays must be finite and satisfy 0 < initial <= maximum")
        self.initial = initial
        self.maximum = maximum
        self.clock = clock
        self.attempts: dict[str, int] = {}
        self.deadlines: dict[str, float] = {}
        self._heap: list[tuple[float, str]] = []

    def defer(self, unit_id: str) -> None:
        attempt = self.attempts.get(unit_id, 0) + 1
        self.attempts[unit_id] = attempt
        delay = min(self.maximum, self.initial * (2 ** min(attempt - 1, 20)))
        deadline = self.clock() + delay
        self.deadlines[unit_id] = deadline
        heapq.heappush(self._heap, (deadline, unit_id))

    def ready(self) -> list[str]:
        now = self.clock()
        result = []
        while self._heap and self._heap[0][0] <= now:
            deadline, unit_id = heapq.heappop(self._heap)
            if self.deadlines.get(unit_id) == deadline:
                del self.deadlines[unit_id]
                result.append(unit_id)
        return result

    def complete(self, unit_id: str) -> None:
        self.attempts.pop(unit_id, None)
        self.deadlines.pop(unit_id, None)

    def is_retry(self, unit_id: str) -> bool:
        return unit_id in self.attempts

    def __len__(self) -> int:
        return len(self.deadlines)
