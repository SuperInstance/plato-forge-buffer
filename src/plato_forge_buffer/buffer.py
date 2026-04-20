"""Prioritized experience replay buffer with curriculum-balanced sampling."""

import json, time, random, math
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Experience:
    prompt: str
    completion: str
    quality: float = 0.5
    priority: str = "P2"
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    use_count: int = 0

class ExperienceBuffer:
    def __init__(self, max_size: int = 10000, dedup_threshold: float = 0.9):
        self._buffer: list[Experience] = []
        self.max_size = max_size
        self.dedup_threshold = dedup_threshold
        self._seen_prompts: list[str] = []

    def _jaccard(self, a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa and not sb: return 1.0
        if not sa or not sb: return 0.0
        return len(sa & sb) / len(sa | sb)

    def _is_dup(self, prompt: str) -> bool:
        for seen in self._seen_prompts[-500:]:
            if self._jaccard(prompt, seen) >= self.dedup_threshold:
                return True
        return False

    def add(self, exp: Experience) -> bool:
        if self._is_dup(exp.prompt):
            return False
        if len(self._buffer) >= self.max_size:
            self._buffer.pop(0)
            self._seen_prompts.pop(0)
        self._buffer.append(exp)
        self._seen_prompts.append(exp.prompt)
        return True

    def add_batch(self, experiences: list[Experience]) -> int:
        return sum(1 for e in experiences if self.add(e))

    def sample(self, batch_size: int, curriculum_balance: bool = True) -> list[Experience]:
        if not self._buffer:
            return []
        if not curriculum_balance:
            weights = [e.quality + 0.1 for e in self._buffer]
            return random.choices(self._buffer, weights=weights, k=min(batch_size, len(self._buffer)))

        p01 = [e for e in self._buffer if e.priority in ("P0", "P1")]
        stale = sorted([e for e in self._buffer if e.use_count < 2], key=lambda e: e.use_count)
        normal = [e for e in self._buffer if e.priority == "P2" and e not in stale]

        n_normal = int(batch_size * 0.7)
        n_priority = int(batch_size * 0.2)
        n_stale = batch_size - n_normal - n_priority

        result = []
        if normal:
            result.extend(random.sample(normal, min(n_normal, len(normal))))
        if p01:
            result.extend(random.sample(p01, min(n_priority, len(p01))))
        if stale:
            result.extend(stale[:n_stale])
        random.shuffle(result)
        for e in result:
            e.use_count += 1
        return result[:batch_size]

    def decay_priorities(self, rate: float = 0.96):
        for e in self._buffer:
            e.use_count = max(0, int(e.use_count * rate))

    def expire_stale(self, max_age_seconds: float = 86400.0):
        cutoff = time.time() - max_age_seconds
        before = len(self._buffer)
        self._buffer = [e for e in self._buffer if e.timestamp >= cutoff]
        self._seen_prompts = self._seen_prompts[before - len(self._buffer):]

    def dedup_merge(self, other: list[Experience]) -> int:
        return self.add_batch(other)

    @property
    def stats(self) -> dict:
        p_counts = {"P0": 0, "P1": 0, "P2": 0}
        total_q = 0.0
        stale = 0
        for e in self._buffer:
            p_counts[e.priority] = p_counts.get(e.priority, 0) + 1
            total_q += e.quality
            if e.use_count < 2:
                stale += 1
        return {"total": len(self._buffer), "by_priority": p_counts,
                "avg_quality": total_q / max(len(self._buffer), 1), "stale_count": stale}

    def clear(self):
        self._buffer.clear()
        self._seen_prompts.clear()

    def export_jsonl(self, path: str):
        with open(path, "w") as f:
            for e in self._buffer:
                f.write(json.dumps({"prompt": e.prompt, "completion": e.completion,
                    "quality": e.quality, "priority": e.priority, "source": e.source,
                    "timestamp": e.timestamp, "use_count": e.use_count}) + "\n")

    def import_jsonl(self, path: str) -> int:
        count = 0
        with open(path) as f:
            for line in f:
                d = json.loads(line.strip())
                if self.add(Experience(**{k: v for k, v in d.items() if k in Experience.__dataclass_fields__})):
                    count += 1
        return count
