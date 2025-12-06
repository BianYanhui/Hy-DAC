"""Task reassignment helpers for the distributed inference demo."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


class TaskReassigner:
	"""Assigns attention heads to workers and redistributes them after failures."""

	def __init__(self, worker_ratios: Dict[str, float], seed: int | None = None) -> None:
		if not worker_ratios:
			raise ValueError("worker_ratios cannot be empty")

		total_ratio = sum(worker_ratios.values())
		if total_ratio <= 0:
			raise ValueError("sum of worker ratios must be > 0")

		self._ratios = {worker: ratio / total_ratio for worker, ratio in worker_ratios.items() if ratio > 0}
		if not self._ratios:
			raise ValueError("at least one worker ratio must be > 0")

		self._rng = random.Random(seed)

	def initial_assignment(self, total_heads: int) -> Dict[str, List[int]]:
		"""Creates an initial mapping of heads to workers."""

		if total_heads <= 0:
			raise ValueError("total_heads must be > 0")

		assignments: Dict[str, List[int]] = {worker: [] for worker in self._ratios}
		proportional_counts = {worker: int(math.floor(total_heads * ratio)) for worker, ratio in self._ratios.items()}

		assigned = sum(proportional_counts.values())
		leftovers = total_heads - assigned

		# Distribute leftover heads by descending ratio to keep ordering deterministic.
		if leftovers > 0:
			for worker, _ in sorted(self._ratios.items(), key=lambda item: item[1], reverse=True):
				if leftovers == 0:
					break
				proportional_counts[worker] += 1
				leftovers -= 1

		head_id = 0
		for worker, count in proportional_counts.items():
			assignments[worker] = list(range(head_id, head_id + count))
			head_id += count

		return assignments

	def reassign(
		self,
		offline_worker: str,
		current_assignments: Dict[str, List[int]],
	) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
		"""Redistributes heads from the offline worker to remaining workers."""

		if offline_worker not in current_assignments:
			return current_assignments, {}

		offline_heads = current_assignments.pop(offline_worker)
		if not offline_heads:
			return current_assignments, {}

		destinations = [worker for worker in current_assignments if worker != offline_worker]
		if not destinations:
			return current_assignments, {}

		new_assignments: Dict[str, List[int]] = {worker: [] for worker in destinations}

		for head in offline_heads:
			target = self._select_worker_for_head(destinations)
			current_assignments.setdefault(target, []).append(head)
			new_assignments[target].append(head)

		# Keep head ordering deterministic for readability.
		for worker in current_assignments:
			current_assignments[worker] = sorted(current_assignments[worker])

		return current_assignments, {worker: heads for worker, heads in new_assignments.items() if heads}

	def _select_worker_for_head(self, candidates: List[str]) -> str:
		weights = [self._ratios.get(worker, 0.0) for worker in candidates]
		total = sum(weights)
		if total == 0:
			return self._rng.choice(candidates)
		threshold = self._rng.random() * total
		cumulative = 0.0
		for worker, weight in zip(candidates, weights):
			cumulative += weight
			if cumulative >= threshold:
				return worker
		return candidates[-1]

