"""Heartbeat monitoring utilities for the distributed inference demo."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class WorkerHeartbeat:
	"""Stores heartbeat state for a single worker."""

	last_seen: float
	is_online: bool = True
	consecutive_misses: int = 0


class HeartbeatMonitor:
	"""Background monitor that detects worker liveness via heartbeats."""

	def __init__(
		self,
		heartbeat_interval: float = 1.0,
		tolerance: float = 3.0,
		on_worker_timeout: Optional[Callable[[str], None]] = None,
	) -> None:
		if heartbeat_interval <= 0:
			raise ValueError("heartbeat_interval must be > 0")
		if tolerance <= heartbeat_interval:
			raise ValueError("tolerance must be greater than heartbeat_interval")

		self.heartbeat_interval = heartbeat_interval
		self.tolerance = tolerance
		self._on_worker_timeout = on_worker_timeout

		self._lock = threading.Lock()
		self._workers: Dict[str, WorkerHeartbeat] = {}
		self._stop_event = threading.Event()
		self._thread: Optional[threading.Thread] = None

	def register_worker(self, worker_id: str) -> None:
		"""Adds a worker to the monitor."""

		now = time.monotonic()
		with self._lock:
			self._workers[worker_id] = WorkerHeartbeat(last_seen=now)

	def unregister_worker(self, worker_id: str) -> None:
		"""Removes a worker from monitoring."""

		with self._lock:
			self._workers.pop(worker_id, None)

	def heartbeat(self, worker_id: str) -> None:
		"""Marks a heartbeat received from a worker."""

		with self._lock:
			status = self._workers.get(worker_id)
			if not status:
				return
			status.last_seen = time.monotonic()
			if not status.is_online:
				status.is_online = True
				status.consecutive_misses = 0

	def start(self) -> None:
		"""Starts the monitoring thread."""

		if self._thread and self._thread.is_alive():
			return
		self._stop_event.clear()
		self._thread = threading.Thread(target=self._monitor_loop, name="HeartbeatMonitor", daemon=True)
		self._thread.start()

	def stop(self) -> None:
		"""Stops the monitoring thread."""

		self._stop_event.set()
		if self._thread:
			self._thread.join(timeout=2.0)
		self._thread = None

	def _monitor_loop(self) -> None:
		check_interval = self.heartbeat_interval / 2.0
		while not self._stop_event.is_set():
			time.sleep(check_interval)
			now = time.monotonic()
			timed_out: Dict[str, WorkerHeartbeat] = {}
			with self._lock:
				for worker_id, status in self._workers.items():
					if not status.is_online:
						continue
					if now - status.last_seen > self.tolerance:
						status.is_online = False
						status.consecutive_misses += 1
						timed_out[worker_id] = status

			for worker_id in timed_out:
				if self._on_worker_timeout:
					try:
						self._on_worker_timeout(worker_id)
					except Exception:
						# Avoid crashing the monitor because of user callbacks.
						pass

