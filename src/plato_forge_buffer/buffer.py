"""Forge buffer — priority ring buffer with overflow handling, backpressure, and batch drain."""
import time
import heapq
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import deque

class BufferOverflowPolicy(Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_LOWEST = "drop_lowest_priority"
    BLOCK = "block"
    EXPAND = "expand"

class EntryPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class BufferEntry:
    id: str
    data: dict
    priority: int = 1
    timestamp: float = field(default_factory=time.time)
    size: int = 1  # abstract size units
    source: str = ""
    room: str = ""

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority  # higher priority first
        return self.timestamp < other.timestamp  # older first

class ForgeBuffer:
    def __init__(self, capacity: int = 1000, overflow: str = "drop_oldest",
                 backpressure_threshold: float = 0.8):
        self.capacity = capacity
        self.overflow_policy = BufferOverflowPolicy(overflow)
        self.backpressure_threshold = backpressure_threshold
        self._heap: list[BufferEntry] = []
        self._entry_map: dict[str, BufferEntry] = {}
        self._total_size: int = 0
        self._dropped: int = 0
        self._drained: int = 0
        self._peak_usage: int = 0
        self._history: deque = deque(maxlen=200)

    def push(self, entry_id: str, data: dict, priority: int = 1,
             source: str = "", room: str = "", size: int = 1) -> bool:
        if len(self._heap) >= self.capacity:
            if not self._handle_overflow():
                self._dropped += 1
                return False
        entry = BufferEntry(id=entry_id, data=data, priority=priority,
                           source=source, room=room, size=size)
        heapq.heappush(self._heap, entry)
        self._entry_map[entry_id] = entry
        self._total_size += size
        self._peak_usage = max(self._peak_usage, len(self._heap))
        return True

    def push_batch(self, entries: list[dict]) -> int:
        pushed = 0
        for e in entries:
            if self.push(e.get("id", ""), e.get("data", {}),
                        e.get("priority", 1), e.get("source", ""),
                        e.get("room", ""), e.get("size", 1)):
                pushed += 1
        return pushed

    def pop(self) -> Optional[BufferEntry]:
        if not self._heap:
            return None
        entry = heapq.heappop(self._heap)
        self._entry_map.pop(entry.id, None)
        self._total_size -= entry.size
        self._drained += 1
        self._history.append({"action": "pop", "id": entry.id, "priority": entry.priority,
                              "timestamp": time.time()})
        return entry

    def peek(self) -> Optional[BufferEntry]:
        return self._heap[0] if self._heap else None

    def drain(self, n: int = 0) -> list[BufferEntry]:
        results = []
        count = n if n > 0 else len(self._heap)
        for _ in range(min(count, len(self._heap))):
            entry = self.pop()
            if entry:
                results.append(entry)
        return results

    def drain_by_room(self, room: str, limit: int = 50) -> list[BufferEntry]:
        remaining = []
        results = []
        while self._heap:
            entry = heapq.heappop(self._heap)
            self._entry_map.pop(entry.id, None)
            self._total_size -= entry.size
            if entry.room == room:
                results.append(entry)
                self._drained += 1
            else:
                remaining.append(entry)
        for entry in remaining:
            heapq.heappush(self._heap, entry)
            self._entry_map[entry.id] = entry
            self._total_size += entry.size
        return results[:limit]

    def get(self, entry_id: str) -> Optional[BufferEntry]:
        return self._entry_map.get(entry_id)

    def remove(self, entry_id: str) -> bool:
        entry = self._entry_map.pop(entry_id, None)
        if not entry:
            return False
        self._total_size -= entry.size
        # Rebuild heap without this entry
        self._heap = [e for e in self._heap if e.id != entry_id]
        heapq.heapify(self._heap)
        return True

    def _handle_overflow(self) -> bool:
        if not self._heap:
            return False
        if self.overflow_policy == BufferOverflowPolicy.DROP_OLDEST:
            # Find oldest entry (lowest priority, oldest timestamp)
            oldest = min(self._heap, key=lambda e: (e.priority, -e.timestamp))
            self.remove(oldest.id)
            return True
        elif self.overflow_policy == BufferOverflowPolicy.DROP_LOWEST:
            lowest = min(self._heap, key=lambda e: e.priority)
            self.remove(lowest.id)
            return True
        elif self.overflow_policy == BufferOverflowPolicy.EXPAND:
            self.capacity = int(self.capacity * 1.5)
            return True
        return False  # BLOCK

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def total_size(self) -> int:
        return self._total_size

    @property
    def utilization(self) -> float:
        return len(self._heap) / max(self.capacity, 1)

    @property
    def is_under_pressure(self) -> bool:
        return self.utilization >= self.backpressure_threshold

    def by_room(self) -> dict[str, int]:
        counts = {}
        for entry in self._heap:
            counts[entry.room] = counts.get(entry.room, 0) + 1
        return counts

    def by_priority(self) -> dict[int, int]:
        counts = {}
        for entry in self._heap:
            counts[entry.priority] = counts.get(entry.priority, 0) + 1
        return dict(sorted(counts.items(), reverse=True))

    def resize(self, new_capacity: int):
        self.capacity = new_capacity
        while len(self._heap) > new_capacity:
            self._handle_overflow()

    @property
    def stats(self) -> dict:
        return {"capacity": self.capacity, "size": len(self._heap),
                "utilization": round(self.utilization, 3),
                "total_size_units": self._total_size,
                "dropped": self._dropped, "drained": self._drained,
                "peak_usage": self._peak_usage,
                "overflow_policy": self.overflow_policy.value,
                "by_priority": self.by_priority(),
                "by_room": self.by_room()}
