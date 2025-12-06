"""Utility helpers that simulate KV cache reuse across workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CacheMetrics:
	reused: int = 0
	recomputed: int = 0
	dropped: int = 0


class KVCacheManager:
	"""Tracks KV cache ownership and reuse across workers."""

	def __init__(self) -> None:
		self._cache: Dict[str, Dict[int, int]] = {}
		self.metrics = CacheMetrics()

	def initialize_from_assignment(self, assignments: Dict[str, List[int]], sequence_length: int) -> None:
		for worker, heads in assignments.items():
			self._cache.setdefault(worker, {})
			for head in heads:
				self._cache[worker][head] = sequence_length

	def plan_updates(
		self,
		previous: Dict[str, List[int]],
		updated: Dict[str, List[int]],
		sequence_length: int,
	) -> Dict[str, Dict[str, List[int]]]:
		plan: Dict[str, Dict[str, List[int]]] = {}

		for worker, prev_heads in previous.items():
			current_cache = self._cache.get(worker, {})
			if worker not in updated:
				removed = list(current_cache.keys())
				self.metrics.dropped += len(removed)
				self._cache.pop(worker, None)
				continue

			new_heads = set(updated[worker])
			prev_set = set(prev_heads)

			reuse = sorted(prev_set & new_heads)
			recompute = sorted(new_heads - prev_set)
			dropped = sorted(prev_set - new_heads)

			for head in dropped:
				current_cache.pop(head, None)

			for head in recompute:
				current_cache[head] = sequence_length

			plan[worker] = {"reuse": reuse, "recompute": recompute}

			self.metrics.reused += len(reuse)
			self.metrics.recomputed += len(recompute)
			self.metrics.dropped += len(dropped)

		for worker, heads in updated.items():
			if worker in previous:
				continue
			cache_entry = self._cache.setdefault(worker, {})
			for head in heads:
				cache_entry[head] = sequence_length
			plan[worker] = {"reuse": [], "recompute": sorted(heads)}
			self.metrics.recomputed += len(heads)

		return plan

	def get_worker_cache(self, worker: str) -> Dict[int, int]:
		return self._cache.setdefault(worker, {})

