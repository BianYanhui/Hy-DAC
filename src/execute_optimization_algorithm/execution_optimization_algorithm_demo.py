"""End-to-end demo that simulates head reassignment with KV cache reuse."""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from heartbeat_detection import HeartbeatMonitor
from kv_cache_reused import KVCacheManager
from task_reassign import TaskReassigner


@dataclass
class SimulationConfig:
	workers: List[str]
	worker_ratios: Dict[str, float]
	total_heads: int
	prompt_length: int
	generation_steps: int
	failure_step: int
	offline_worker: str
	heartbeat_interval: float = 0.5
	model_path: Optional[str] = None
	tokenizer_path: Optional[str] = None
	params_path: Optional[str] = None
	use_real_model: bool = False

	def __post_init__(self) -> None:
		if self.failure_step >= self.generation_steps:
			raise ValueError("failure_step must be less than generation_steps")
		if self.offline_worker not in self.workers:
			raise ValueError("offline_worker must be part of workers")


class ModelExecutor:
	"""Lightweight executor that mimics per-head compute cost."""

	def __init__(
		self,
		model_path: Optional[str],
		tokenizer_path: Optional[str],
		params_path: Optional[str],
		use_real_model: bool,
	) -> None:
		self.model_path = model_path
		self.tokenizer_path = tokenizer_path
		self.params_path = params_path
		self.use_real_model = use_real_model
		self._real_model = None
		self._tokenizer = None
		self._torch = None

		if use_real_model:
			self._load_real_model()

	def _load_real_model(self) -> None:
		try:
			import json
			import importlib

			torch_module = importlib.import_module("torch")
			safetensors_module = importlib.import_module("safetensors.torch")
			transformers_module = importlib.import_module("transformers")

			LlamaConfig = getattr(transformers_module, "LlamaConfig")
			LlamaForCausalLM = getattr(transformers_module, "LlamaForCausalLM")
			AutoTokenizer = getattr(transformers_module, "AutoTokenizer")
			load_safetensors = getattr(safetensors_module, "load_file")
		except ImportError as exc:  # pragma: no cover - optional dependency
			logging.error("Unable to load real model because dependencies are missing: %s", exc)
			self.use_real_model = False
			return

		if not (self.model_path and self.tokenizer_path and self.params_path):
			logging.error("Real model requested but paths are incomplete. Falling back to synthetic executor.")
			self.use_real_model = False
			return

		try:
			with open(self.params_path, "r", encoding="utf-8") as params_fp:
				params = json.load(params_fp)

			config = LlamaConfig(**params)
			model = LlamaForCausalLM(config)
			state_dict = load_safetensors(self.model_path)
			model.load_state_dict(state_dict)
			model.eval()

			tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=False)

			self._real_model = model
			self._tokenizer = tokenizer
			self._torch = torch_module
		except Exception as exc:  # pragma: no cover - runtime guard
			logging.error("Failed to load real model: %s", exc)
			self.use_real_model = False

	def process(self, tokens: List[int], head_ids: List[int], worker_id: str) -> float:
		if self.use_real_model and self._real_model and self._tokenizer and self._torch:  # pragma: no cover - heavy path
			input_ids = self._torch.tensor([tokens], dtype=self._torch.long)
			with self._torch.no_grad():
				_ = self._real_model(input_ids=input_ids)
			# Using time spent as latency proxy is complex; for the demo we just return a constant.
			return 0.0

		# Synthetic workload: proportional latency to heads * tokens.
		simulated_latency = 0.01 * len(tokens) * max(len(head_ids), 1)
		time.sleep(min(simulated_latency, 0.05))
		return simulated_latency


class WorkerNode(threading.Thread):
	"""Simulated worker that processes assigned heads and emits heartbeats."""

	def __init__(
		self,
		worker_id: str,
		heartbeat_interval: float,
		monitor: HeartbeatMonitor,
		task_queue: "queue.Queue[tuple[str, dict]]",
		result_queue: "queue.Queue[dict]",
		executor: ModelExecutor,
	) -> None:
		super().__init__(name=f"Worker-{worker_id}", daemon=True)
		self.worker_id = worker_id
		self._heartbeat_interval = heartbeat_interval
		self._monitor = monitor
		self._task_queue = task_queue
		self._result_queue = result_queue
		self._executor = executor

		self._stop_event = threading.Event()
		self._failed_event = threading.Event()
		self._current_heads: List[int] = []

	def update_heads(self, heads: List[int]) -> None:
		self._current_heads = list(heads)

	def simulate_failure(self) -> None:
		self._failed_event.set()

	def shutdown(self) -> None:
		self._stop_event.set()
		self._task_queue.put(("stop", {}))

	@property
	def is_available(self) -> bool:
		return not self._failed_event.is_set()

	def run(self) -> None:
		next_heartbeat = time.monotonic()
		self._monitor.heartbeat(self.worker_id)

		while not self._stop_event.is_set():
			now = time.monotonic()
			if self.is_available and now >= next_heartbeat:
				self._monitor.heartbeat(self.worker_id)
				next_heartbeat = now + self._heartbeat_interval

			try:
				task, payload = self._task_queue.get(timeout=0.1)
			except queue.Empty:
				continue

			if task == "stop":
				self._task_queue.task_done()
				break

			if task == "process":
				if not self.is_available:
					self._result_queue.put(
						{
							"worker": self.worker_id,
							"status": "failed",
							"step": payload.get("step"),
							"reuse": [],
							"recompute": payload.get("cache_plan", {}).get("recompute", []),
							"latency": 0.0,
						}
					)
					self._task_queue.task_done()
					continue

				tokens = payload.get("tokens", [])
				head_ids = payload.get("head_ids", self._current_heads)
				cache_plan = payload.get("cache_plan", {"reuse": head_ids, "recompute": []})

				latency = self._executor.process(tokens, head_ids, self.worker_id)

				self._result_queue.put(
					{
						"worker": self.worker_id,
						"status": "ok",
						"step": payload.get("step"),
						"reuse": cache_plan.get("reuse", []),
						"recompute": cache_plan.get("recompute", []),
						"latency": latency,
						"token_count": len(tokens),
					}
				)

			self._task_queue.task_done()


class DistributedInferenceDemo:
	"""Coordinates the simulation for head reassignment and cache reuse."""

	def __init__(self, config: SimulationConfig) -> None:
		self.config = config
		self.reassigner = TaskReassigner(config.worker_ratios)
		self.cache_manager = KVCacheManager()
		self.offline_events: "queue.Queue[str]" = queue.Queue()
		self.monitor = HeartbeatMonitor(
			heartbeat_interval=config.heartbeat_interval,
			tolerance=config.heartbeat_interval * 3,
			on_worker_timeout=self.offline_events.put,
		)
		self.executor = ModelExecutor(
			model_path=config.model_path,
			tokenizer_path=config.tokenizer_path,
			params_path=config.params_path,
			use_real_model=config.use_real_model,
		)

		self.assignments: Dict[str, List[int]] = {}
		self.task_queues: Dict[str, "queue.Queue[tuple[str, dict]]"] = {}
		self.workers: Dict[str, WorkerNode] = {}
		self.result_queue: "queue.Queue[dict]" = queue.Queue()
		self.timeline: List[str] = []
		self.step_results: List[dict] = []
		self.cache_plan: Dict[str, Dict[str, List[int]]] = {}
		self.current_sequence_length = 0

	def setup(self) -> None:
		self.assignments = self.reassigner.initial_assignment(self.config.total_heads)
		self.cache_manager.initialize_from_assignment(self.assignments, self.current_sequence_length)

		self.cache_plan = {
			worker: {"reuse": [], "recompute": heads}
			for worker, heads in self.assignments.items()
		}

		self.monitor.start()

		for worker_id in self.config.workers:
			self.monitor.register_worker(worker_id)
			task_queue: "queue.Queue[tuple[str, dict]]" = queue.Queue()
			worker = WorkerNode(
				worker_id=worker_id,
				heartbeat_interval=self.config.heartbeat_interval,
				monitor=self.monitor,
				task_queue=task_queue,
				result_queue=self.result_queue,
				executor=self.executor,
			)
			worker.update_heads(self.assignments.get(worker_id, []))
			worker.start()

			self.workers[worker_id] = worker
			self.task_queues[worker_id] = task_queue

		self.timeline.append(
			f"Initial assignment: {self.assignments} (prompt length {self.config.prompt_length})"
		)

	def teardown(self) -> None:
		for worker in self.workers.values():
			worker.shutdown()
		for worker in self.workers.values():
			worker.join(timeout=2.0)
		self.monitor.stop()

	def _await_offline_detection(self, expected_worker: str) -> None:
		deadline = time.time() + 10.0
		while time.time() < deadline:
			try:
				worker_id = self.offline_events.get(timeout=0.5)
			except queue.Empty:
				continue
			if worker_id == expected_worker:
				self._process_offline_worker(worker_id)
				return
			self._process_offline_worker(worker_id)
		raise RuntimeError(f"Timeout waiting for offline detection of {expected_worker}")

	def _process_offline_worker(self, worker_id: str) -> None:
		if worker_id not in self.assignments:
			return

		previous = {k: list(v) for k, v in self.assignments.items()}
		working_copy = {k: list(v) for k, v in self.assignments.items()}
		updated_assignments, delta = self.reassigner.reassign(worker_id, working_copy)

		self.assignments = updated_assignments
		plan = self.cache_manager.plan_updates(previous, self.assignments, self.current_sequence_length)
		self.cache_plan = plan

		self.timeline.append(
			f"Leader detected offline worker {worker_id}; reassigned heads {delta}"
		)

		self.monitor.unregister_worker(worker_id)

		worker = self.workers.pop(worker_id, None)
		if worker:
			worker.shutdown()
		self.task_queues.pop(worker_id, None)

		for worker_id, heads in self.assignments.items():
			worker_node = self.workers.get(worker_id)
			if worker_node:
				worker_node.update_heads(heads)

	def _dispatch_step(self, tokens: List[int], step: int, phase: str) -> None:
		expected_results = 0
		for worker_id, worker in list(self.workers.items()):
			if not worker.is_available:
				continue
			heads = self.assignments.get(worker_id, [])
			if not heads:
				continue

			plan = self.cache_plan.get(worker_id, {"reuse": heads, "recompute": []})
			payload = {
				"tokens": tokens,
				"head_ids": heads,
				"step": step,
				"phase": phase,
				"cache_plan": plan,
			}
			self.task_queues[worker_id].put(("process", payload))
			expected_results += 1

		results = []
		for _ in range(expected_results):
			result = self.result_queue.get(timeout=5.0)
			results.append(result)

		self.timeline.append(
			f"Step {phase}-{step}: {results}"
		)
		self.step_results.extend(results)

		self.cache_plan = {
			worker_id: {"reuse": self.assignments.get(worker_id, []), "recompute": []}
			for worker_id in self.assignments
		}

		self.current_sequence_length += len(tokens)

	def run(self) -> None:
		self.setup()

		try:
			prompt_tokens = list(range(self.config.prompt_length))
			self._dispatch_step(prompt_tokens, step=0, phase="prompt")

			generation_tokens = [self.config.prompt_length + idx for idx in range(self.config.generation_steps)]

			for step_index, token in enumerate(generation_tokens, start=1):
				if step_index - 1 == self.config.failure_step:
					worker = self.workers.get(self.config.offline_worker)
					if worker:
						worker.simulate_failure()
					self.timeline.append(
						f"Simulated failure for {self.config.offline_worker} before generation step {step_index}"
					)
					self._await_offline_detection(self.config.offline_worker)

				self._dispatch_step([token], step=step_index, phase="generation")

		finally:
			self.teardown()

		self._report()

	def _report(self) -> None:
		logging.info("=== Simulation timeline ===")
		for entry in self.timeline:
			logging.info(entry)

		metrics = self.cache_manager.metrics
		logging.info(
			"KV cache stats: reused=%s recomputed=%s dropped=%s",
			metrics.reused,
			metrics.recomputed,
			metrics.dropped,
		)


def main() -> None:
	parser = argparse.ArgumentParser(description="Distributed inference optimization demo")
	parser.add_argument("--use-real-model", action="store_true", help="Enable loading the actual LLaMA model")
	args = parser.parse_args()

	config = SimulationConfig(
		workers=["worker-0", "worker-1", "worker-2", "worker-3", "worker-4"],
		worker_ratios={
			"worker-0": 0.25,
			"worker-1": 0.20,
			"worker-2": 0.20,
			"worker-3": 0.20,
			"worker-4": 0.15,
		},
		total_heads=16,
		prompt_length=8,
		generation_steps=10,
		failure_step=3,
		offline_worker="worker-2",
		heartbeat_interval=0.5,
		model_path="/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/model.safetensors",
		tokenizer_path="/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/tokenizer.model",
		params_path="/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/params.json",
		use_real_model=args.use_real_model,
	)

	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	demo = DistributedInferenceDemo(config)
	demo.run()


if __name__ == "__main__":
	main()

