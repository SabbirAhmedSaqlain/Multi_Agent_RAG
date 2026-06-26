from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class StepTiming:
    name: str
    start: float = field(default_factory=time.time)
    end: float = 0.0

    def stop(self):
        self.end = time.time()

    @property
    def elapsed(self) -> float:
        return self.end - self.start


class PipelineMetrics:
    def __init__(self):
        self._steps: list[StepTiming] = []
        self._current: StepTiming | None = None
        self.revision_count: int = 0

    def start_step(self, name: str):
        self._current = StepTiming(name=name)
        self._steps.append(self._current)

    def end_step(self):
        if self._current:
            self._current.stop()

    @property
    def total_elapsed(self) -> float:
        if not self._steps:
            return 0.0
        first = self._steps[0].start
        last = max(s.end for s in self._steps if s.end > 0)
        return last - first

    def report(self) -> dict:
        return {
            "total_seconds": round(self.total_elapsed, 2),
            "revision_cycles": self.revision_count,
            "steps": {s.name: round(s.elapsed, 2) for s in self._steps if s.end > 0},
        }
