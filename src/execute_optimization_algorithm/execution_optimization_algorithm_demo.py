"""
分布式推理系统设备离线优化Demo (Distributed Inference Device Offline Optimization Demo)

该Demo演示了在分布式张量并行推理系统中，当设备离线时的优化处理策略。

主要功能：
1. 模拟多设备分布式推理环境（使用多线程）
2. Leader节点进行心跳检测和任务分配
3. Worker节点持有各自分配的Heads和对应的KV-Cache
4. 设备离线时，使用KV-Cache复用策略进行优化重计算
5. 与传统全量重计算策略进行性能对比

系统架构：
- 1个Leader（同时也参与计算）
- N-1个Workers
- 使用心跳机制检测设备存活
- 支持动态任务重分配

使用方法：
    python execution_optimization_algorithm_demo.py
"""

import os
import sys
import time
import json
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import argparse

import torch
import torch.nn as nn
import numpy as np

# 导入本地模块
from task_reassign import TaskReassignManager, create_task_manager
from heartbeat_detection import HeartbeatDetector, DeviceStatus
from kv_cache_reused import KVCacheManager, KVCacheReuseEngine
from performance_comparator import PerformanceComparator, ComparisonResult


class DeviceRole(Enum):
    """设备角色"""
    LEADER = "leader"
    WORKER = "worker"


@dataclass
class DeviceConfig:
    """设备配置"""
    device_id: str
    role: DeviceRole
    assigned_heads: List[int] = field(default_factory=list)
    is_online: bool = True


@dataclass
class InferenceTask:
    """推理任务"""
    task_id: str
    input_ids: torch.Tensor
    created_time: float = field(default_factory=time.time)


class SimulatedDevice(threading.Thread):
    """
    模拟的设备节点
    
    每个设备运行在独立的线程中，模拟分布式环境。
    """
    
    def __init__(self, 
                 device_id: str,
                 role: DeviceRole,
                 n_layers: int,
                 n_kv_heads: int,
                 head_dim: int,
                 message_queue: queue.Queue,
                 result_queue: queue.Queue):
        super().__init__(name=f"Device-{device_id}")
        
        self.device_id = device_id
        self.role = role
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        
        # 通信队列
        self.message_queue = message_queue  # 接收消息
        self.result_queue = result_queue    # 发送结果
        
        # KV-Cache管理器
        self.kv_manager = KVCacheManager(
            device_id=device_id,
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim
        )
        
        # 控制标志
        self.running = True
        self.is_responsive = True  # 用于模拟离线
        
        # 统计
        self.tasks_completed = 0
        self.computation_time = 0.0
    
    def set_assigned_heads(self, heads: List[int]):
        """设置分配的Heads"""
        self.kv_manager.set_assigned_heads(heads)
    
    def add_heads(self, new_heads: List[int]):
        """添加新的Heads"""
        self.kv_manager.add_heads(new_heads)
    
    def go_offline(self):
        """模拟设备离线"""
        self.is_responsive = False
        print(f"[{self.device_id}] Going OFFLINE")
    
    def go_online(self):
        """恢复在线"""
        self.is_responsive = True
        print(f"[{self.device_id}] Back ONLINE")
    
    def compute_kv_cache(self, input_tensor: torch.Tensor, 
                          head_ids: List[int] = None) -> float:
        """
        计算KV-Cache
        
        Args:
            input_tensor: 输入张量
            head_ids: 要计算的heads，如果不指定则计算需要计算的heads
            
        Returns:
            计算时间
        """
        if head_ids is None:
            head_ids = self.kv_manager.get_heads_needing_computation()
        
        if not head_ids:
            return 0.0
        
        start_time = time.time()
        
        batch_size, seq_len, dim = input_tensor.shape
        
        # 模拟计算每层的KV
        for layer_id in range(self.n_layers):
            for head_id in head_ids:
                # 模拟K和V的计算
                k = torch.randn(batch_size, seq_len, self.head_dim)
                v = torch.randn(batch_size, seq_len, self.head_dim)
                
                # 存储到缓存
                self.kv_manager.set_cache(layer_id, head_id, k, v)
            
            # 模拟层间延迟
            time.sleep(0.002)  # 2ms per layer
        
        computation_time = time.time() - start_time
        self.computation_time += computation_time
        
        return computation_time
    
    def respond_to_heartbeat(self) -> bool:
        """响应心跳请求"""
        return self.is_responsive
    
    def process_message(self, message: Dict) -> Optional[Dict]:
        """处理接收到的消息"""
        msg_type = message.get("type")
        
        if msg_type == "heartbeat":
            if self.is_responsive:
                return {"type": "heartbeat_ack", "device_id": self.device_id}
            return None
        
        elif msg_type == "compute_kv":
            if not self.is_responsive:
                return None
            
            input_tensor = message.get("input_tensor")
            head_ids = message.get("head_ids")
            
            comp_time = self.compute_kv_cache(input_tensor, head_ids)
            self.tasks_completed += 1
            
            return {
                "type": "compute_kv_done",
                "device_id": self.device_id,
                "heads_computed": head_ids if head_ids else self.kv_manager.get_assigned_heads(),
                "computation_time": comp_time
            }
        
        elif msg_type == "add_heads":
            new_heads = message.get("heads", [])
            self.add_heads(new_heads)
            return {
                "type": "add_heads_done",
                "device_id": self.device_id,
                "new_heads": new_heads
            }
        
        elif msg_type == "get_status":
            return {
                "type": "status",
                "device_id": self.device_id,
                "assigned_heads": self.kv_manager.get_assigned_heads(),
                "cached_heads": self.kv_manager.get_heads_with_cache(),
                "needs_computation": self.kv_manager.get_heads_needing_computation(),
                "is_responsive": self.is_responsive
            }
        
        elif msg_type == "shutdown":
            self.running = False
            return {"type": "shutdown_ack", "device_id": self.device_id}
        
        return None
    
    def run(self):
        """设备主循环"""
        print(f"[{self.device_id}] Started as {self.role.value.upper()}")
        
        while self.running:
            try:
                # 非阻塞方式获取消息
                message = self.message_queue.get(timeout=0.1)
                
                result = self.process_message(message)
                
                if result:
                    self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{self.device_id}] Error: {e}")
        
        print(f"[{self.device_id}] Stopped")


class DistributedInferenceSystem:
    """
    分布式推理系统
    
    协调多个模拟设备进行分布式推理，支持：
    - 设备管理和任务分配
    - 心跳检测和设备存活监控
    - 设备离线处理和任务重分配
    - KV-Cache复用优化
    """
    
    def __init__(self,
                 model_params_path: str,
                 num_devices: int = 4,
                 head_distribution: Dict[str, List[int]] = None):
        """
        初始化分布式推理系统
        
        Args:
            model_params_path: 模型参数文件路径
            num_devices: 设备数量
            head_distribution: 可选的Head分配方案
        """
        # 加载模型参数
        with open(model_params_path, 'r') as f:
            self.params = json.load(f)
        
        self.dim = self.params['dim']
        self.n_layers = self.params['n_layers']
        self.n_heads = self.params['n_heads']
        self.n_kv_heads = self.params['n_kv_heads']
        self.head_dim = self.dim // self.n_heads
        
        self.num_devices = num_devices
        
        # 创建任务管理器
        device_ids = [f"Device_{i}" for i in range(num_devices)]
        self.task_manager = TaskReassignManager(
            total_heads=self.n_kv_heads,
            device_ids=device_ids,
            head_distribution=head_distribution
        )
        
        # 创建设备
        self.devices: Dict[str, SimulatedDevice] = {}
        self.message_queues: Dict[str, queue.Queue] = {}
        self.result_queue = queue.Queue()  # 共享的结果队列
        
        for i, device_id in enumerate(device_ids):
            msg_queue = queue.Queue()
            self.message_queues[device_id] = msg_queue
            
            device = SimulatedDevice(
                device_id=device_id,
                role=DeviceRole.LEADER if i == 0 else DeviceRole.WORKER,
                n_layers=self.n_layers,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                message_queue=msg_queue,
                result_queue=self.result_queue
            )
            
            # 设置初始分配的heads
            assigned_heads = self.task_manager.get_device_heads(device_id)
            device.set_assigned_heads(assigned_heads)
            
            self.devices[device_id] = device
        
        # 创建心跳检测器
        self.heartbeat_detector = HeartbeatDetector(
            heartbeat_interval=0.5,
            timeout=0.2,
            max_failures=2,
            on_device_offline=self._on_device_offline
        )
        
        # 注册设备到心跳检测
        for device_id in device_ids[1:]:  # Leader不需要被检测
            self.heartbeat_detector.register_device(device_id)
        
        # 创建性能对比器
        self.comparator = PerformanceComparator(
            n_kv_heads=self.n_kv_heads,
            n_layers=self.n_layers,
            head_dim=self.head_dim
        )
        
        # 统计信息
        self.offline_events: List[Dict] = []
        self.reuse_results: List[Dict] = []
        self.full_recompute_results: List[Dict] = []
        
        print(f"\n{'='*70}")
        print("Distributed Inference System Initialized")
        print(f"{'='*70}")
        print(f"  Model Parameters:")
        print(f"    - Dimensions: {self.dim}")
        print(f"    - Layers: {self.n_layers}")
        print(f"    - Attention Heads: {self.n_heads}")
        print(f"    - KV Heads: {self.n_kv_heads}")
        print(f"    - Head Dimension: {self.head_dim}")
        print(f"  System Configuration:")
        print(f"    - Number of Devices: {num_devices}")
        print(f"    - Leader: Device_0")
        print(f"{'='*70}\n")
    
    def _on_device_offline(self, device_id: str):
        """设备离线回调"""
        print(f"\n[ALERT] Device {device_id} detected as OFFLINE!")
        
        # 记录离线事件
        self.offline_events.append({
            "device_id": device_id,
            "time": time.time(),
            "assigned_heads": self.task_manager.get_device_heads(device_id)
        })
    
    def start(self):
        """启动系统"""
        print("[System] Starting all devices...")
        
        # 启动所有设备
        for device in self.devices.values():
            device.start()
        
        # 启动心跳检测
        self.heartbeat_detector.start()
        
        print("[System] All devices started")
        time.sleep(0.5)  # 等待设备初始化
    
    def stop(self):
        """停止系统"""
        print("[System] Stopping all devices...")
        
        # 停止心跳检测
        self.heartbeat_detector.stop()
        
        # 发送关闭命令
        for device_id, msg_queue in self.message_queues.items():
            msg_queue.put({"type": "shutdown"})
        
        # 等待设备停止
        for device in self.devices.values():
            device.join(timeout=2.0)
        
        print("[System] All devices stopped")
    
    def send_message(self, device_id: str, message: Dict) -> Optional[Dict]:
        """发送消息给设备并等待响应"""
        if device_id not in self.message_queues:
            return None
        
        self.message_queues[device_id].put(message)
        
        # 等待响应
        try:
            result = self.result_queue.get(timeout=5.0)
            return result
        except queue.Empty:
            return None
    
    def broadcast_message(self, message: Dict) -> List[Dict]:
        """广播消息给所有设备"""
        for msg_queue in self.message_queues.values():
            msg_queue.put(message.copy())
        
        results = []
        expected = len(self.message_queues)
        
        while len(results) < expected:
            try:
                result = self.result_queue.get(timeout=5.0)
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def compute_initial_kv_cache(self, input_ids: torch.Tensor) -> Dict:
        """
        计算初始KV-Cache
        
        所有设备并行计算各自分配的heads的KV-Cache
        """
        print("\n[System] Computing initial KV-Cache for all devices...")
        
        # 创建输入张量
        batch_size, seq_len = input_ids.shape
        input_tensor = torch.randn(batch_size, seq_len, self.dim)
        
        results = {}
        total_time = 0.0
        total_heads = 0
        
        for device_id, device in self.devices.items():
            if device.is_responsive:
                assigned_heads = device.kv_manager.get_assigned_heads()
                comp_time = device.compute_kv_cache(input_tensor, assigned_heads)
                
                results[device_id] = {
                    "heads_computed": assigned_heads,
                    "computation_time": comp_time
                }
                
                total_time += comp_time
                total_heads += len(assigned_heads)
                
                print(f"  {device_id}: computed {len(assigned_heads)} heads in {comp_time*1000:.2f}ms")
        
        print(f"\n[System] Initial KV-Cache computation complete")
        print(f"  Total heads: {total_heads}")
        print(f"  Total time: {total_time*1000:.2f}ms")
        
        return {
            "per_device": results,
            "total_time": total_time,
            "total_heads": total_heads
        }
    
    def simulate_device_offline(self, device_id: str):
        """模拟设备离线"""
        if device_id in self.devices:
            self.devices[device_id].go_offline()
            self.heartbeat_detector.simulate_device_offline(device_id)
            self.task_manager.mark_device_offline(device_id)
    
    def handle_device_offline_with_reuse(self, 
                                          offline_device_id: str,
                                          input_ids: torch.Tensor) -> Dict:
        """
        使用KV-Cache复用策略处理设备离线
        
        Args:
            offline_device_id: 离线设备ID
            input_ids: 输入token IDs
            
        Returns:
            处理结果
        """
        print(f"\n{'='*70}")
        print(f"Handling Device Offline: {offline_device_id}")
        print(f"Strategy: KV-Cache Reuse")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # 1. 重新分配任务
        target_device_id, reassigned_heads = self.task_manager.reassign_tasks(offline_device_id)
        
        print(f"\n[Task Reassignment]")
        print(f"  Offline device: {offline_device_id}")
        print(f"  Reassigned heads {reassigned_heads} to {target_device_id}")
        
        # 2. 更新目标设备的heads
        if target_device_id in self.devices:
            target_device = self.devices[target_device_id]
            original_heads = target_device.kv_manager.get_assigned_heads()
            cached_heads = target_device.kv_manager.get_heads_with_cache()
            
            target_device.add_heads(reassigned_heads)
            
            print(f"\n[{target_device_id}] Before reassignment:")
            print(f"  Original heads: {original_heads}")
            print(f"  Cached heads: {cached_heads}")
            print(f"  After adding: {target_device.kv_manager.get_assigned_heads()}")
        
        # 3. 只计算新分配的heads（复用已有缓存）
        batch_size, seq_len = input_ids.shape
        input_tensor = torch.randn(batch_size, seq_len, self.dim)
        
        # 只计算新分配的heads
        print(f"\n[KV-Cache Reuse] Only recomputing heads: {reassigned_heads}")
        recompute_time = target_device.compute_kv_cache(input_tensor, reassigned_heads)
        
        total_time = time.time() - start_time
        
        # 统计
        result = {
            "strategy": "kv_cache_reuse",
            "offline_device": offline_device_id,
            "target_device": target_device_id,
            "reassigned_heads": reassigned_heads,
            "heads_reused": cached_heads,
            "heads_recomputed": reassigned_heads,
            "num_heads_reused": len(cached_heads),
            "num_heads_recomputed": len(reassigned_heads),
            "recompute_time": recompute_time,
            "total_time": total_time
        }
        
        self.reuse_results.append(result)
        
        print(f"\n[Result]")
        print(f"  Heads reused: {len(cached_heads)}")
        print(f"  Heads recomputed: {len(reassigned_heads)}")
        print(f"  Recompute time: {recompute_time*1000:.2f}ms")
        print(f"  Total time: {total_time*1000:.2f}ms")
        
        return result
    
    def handle_device_offline_full_recompute(self, 
                                              offline_device_id: str,
                                              input_ids: torch.Tensor) -> Dict:
        """
        使用传统全量重计算策略处理设备离线
        
        Args:
            offline_device_id: 离线设备ID
            input_ids: 输入token IDs
            
        Returns:
            处理结果
        """
        print(f"\n{'='*70}")
        print(f"Handling Device Offline: {offline_device_id}")
        print(f"Strategy: Full Recompute (Traditional)")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # 创建输入张量
        batch_size, seq_len = input_ids.shape
        input_tensor = torch.randn(batch_size, seq_len, self.dim)
        
        # 清除所有在线设备的缓存
        print("\n[Full Recompute] Clearing all caches...")
        for device_id, device in self.devices.items():
            if device.is_responsive and device_id != offline_device_id:
                device.kv_manager.clear_all_cache()
        
        # 重新计算所有heads
        print("[Full Recompute] Recomputing all heads...")
        total_compute_time = 0.0
        total_heads = 0
        
        for device_id, device in self.devices.items():
            if device.is_responsive and device_id != offline_device_id:
                heads = device.kv_manager.get_assigned_heads()
                comp_time = device.compute_kv_cache(input_tensor, heads)
                
                total_compute_time += comp_time
                total_heads += len(heads)
                
                print(f"  {device_id}: recomputed {len(heads)} heads in {comp_time*1000:.2f}ms")
        
        total_time = time.time() - start_time
        
        result = {
            "strategy": "full_recompute",
            "offline_device": offline_device_id,
            "total_heads_computed": total_heads,
            "total_compute_time": total_compute_time,
            "total_time": total_time
        }
        
        self.full_recompute_results.append(result)
        
        print(f"\n[Result]")
        print(f"  Total heads recomputed: {total_heads}")
        print(f"  Total compute time: {total_compute_time*1000:.2f}ms")
        print(f"  Total time: {total_time*1000:.2f}ms")
        
        return result
    
    def run_comparison_demo(self, input_ids: torch.Tensor) -> Dict:
        """
        运行对比Demo
        
        对比KV-Cache复用策略和全量重计算策略的性能
        """
        print("\n" + "#"*70)
        print("# PERFORMANCE COMPARISON DEMO")
        print("#"*70)
        
        # 首先计算初始KV-Cache
        initial_result = self.compute_initial_kv_cache(input_ids)
        
        self.print_system_status()
        
        # 选择一个设备模拟离线
        offline_device_id = "Device_1"  # 选择第一个worker
        
        # ============ 测试1: KV-Cache复用策略 ============
        print("\n" + "="*70)
        print("TEST 1: KV-Cache Reuse Strategy")
        print("="*70)
        
        # 模拟设备离线
        self.simulate_device_offline(offline_device_id)
        time.sleep(0.5)  # 等待心跳检测
        
        # 使用复用策略处理
        reuse_result = self.handle_device_offline_with_reuse(offline_device_id, input_ids)
        
        self.print_system_status()
        
        # 恢复状态用于对比测试
        self._reset_for_comparison(offline_device_id, initial_result)
        
        # ============ 测试2: 全量重计算策略 ============
        print("\n" + "="*70)
        print("TEST 2: Full Recompute Strategy (Traditional)")
        print("="*70)
        
        # 再次模拟设备离线
        self.simulate_device_offline(offline_device_id)
        
        # 使用全量重计算策略处理
        full_result = self.handle_device_offline_full_recompute(offline_device_id, input_ids)
        
        # ============ 性能对比 ============
        comparison = self._compare_results(reuse_result, full_result)
        
        return comparison
    
    def _reset_for_comparison(self, offline_device_id: str, initial_result: Dict):
        """重置系统状态以进行公平对比"""
        print("\n[System] Resetting system state for comparison...")
        
        # 恢复离线设备
        if offline_device_id in self.devices:
            self.devices[offline_device_id].go_online()
            self.heartbeat_detector.simulate_device_online(offline_device_id)
            self.task_manager.mark_device_online(offline_device_id)
        
        # 重新初始化任务分配
        device_ids = list(self.devices.keys())
        self.task_manager = TaskReassignManager(
            total_heads=self.n_kv_heads,
            device_ids=device_ids
        )
        
        # 重新设置每个设备的heads和缓存
        for device_id, device in self.devices.items():
            assigned_heads = self.task_manager.get_device_heads(device_id)
            device.kv_manager = KVCacheManager(
                device_id=device_id,
                n_layers=self.n_layers,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim
            )
            device.kv_manager.set_assigned_heads(assigned_heads)
        
        # 重新计算初始KV-Cache
        batch_size = 1
        seq_len = 64
        input_tensor = torch.randn(batch_size, seq_len, self.dim)
        
        for device_id, device in self.devices.items():
            if device.is_responsive:
                assigned_heads = device.kv_manager.get_assigned_heads()
                device.compute_kv_cache(input_tensor, assigned_heads)
        
        print("[System] Reset complete")
    
    def _compare_results(self, reuse_result: Dict, full_result: Dict) -> Dict:
        """对比两种策略的结果"""
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON RESULTS")
        print("="*70)
        
        reuse_time = reuse_result['total_time']
        full_time = full_result['total_time']
        
        speedup = full_time / reuse_time if reuse_time > 0 else float('inf')
        time_saved = full_time - reuse_time
        time_saved_percent = (time_saved / full_time * 100) if full_time > 0 else 0
        
        heads_reused = reuse_result['num_heads_reused']
        heads_recomputed = reuse_result['num_heads_recomputed']
        total_heads_full = full_result['total_heads_computed']
        
        computation_saved = total_heads_full - heads_recomputed
        computation_saved_percent = (computation_saved / total_heads_full * 100) if total_heads_full > 0 else 0
        
        print(f"\n{'Strategy':<30} {'Time (ms)':<15} {'Heads Computed':<20}")
        print("-"*65)
        print(f"{'KV-Cache Reuse':<30} {reuse_time*1000:<15.2f} {heads_recomputed:<20}")
        print(f"{'Full Recompute':<30} {full_time*1000:<15.2f} {total_heads_full:<20}")
        print("-"*65)
        
        print(f"\n{'Metric':<35} {'Value':<20}")
        print("-"*55)
        print(f"{'Speedup':<35} {speedup:.2f}x")
        print(f"{'Time Saved':<35} {time_saved*1000:.2f}ms ({time_saved_percent:.1f}%)")
        print(f"{'Heads Reused (Cache Hit)':<35} {heads_reused}")
        print(f"{'Computation Saved':<35} {computation_saved} heads ({computation_saved_percent:.1f}%)")
        print("-"*55)
        
        print(f"\n✓ KV-Cache Reuse strategy is {speedup:.2f}x faster than Full Recompute!")
        print(f"✓ Saved {computation_saved_percent:.1f}% of computation by reusing cached values.")
        print("="*70)
        
        return {
            "reuse_result": reuse_result,
            "full_result": full_result,
            "speedup": speedup,
            "time_saved_ms": time_saved * 1000,
            "time_saved_percent": time_saved_percent,
            "computation_saved_percent": computation_saved_percent
        }
    
    def print_system_status(self):
        """打印系统状态"""
        print("\n" + "-"*70)
        print("System Status")
        print("-"*70)
        
        self.task_manager.print_status()
        
        print("Device KV-Cache Status:")
        for device_id, device in self.devices.items():
            status = "ONLINE" if device.is_responsive else "OFFLINE"
            assigned = device.kv_manager.get_assigned_heads()
            cached = device.kv_manager.get_heads_with_cache()
            needs_comp = device.kv_manager.get_heads_needing_computation()
            
            print(f"  {device_id}: [{status}]")
            print(f"    Assigned: {assigned}")
            print(f"    Cached: {cached}")
            print(f"    Needs Computation: {needs_comp}")
        
        print("-"*70 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Distributed Inference Device Offline Optimization Demo')
    
    parser.add_argument('--model-params', type=str,
                        default='/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/params.json',
                        help='Path to model params.json')
    parser.add_argument('--num-devices', type=int, default=4,
                        help='Number of simulated devices')
    parser.add_argument('--seq-length', type=int, default=64,
                        help='Sequence length for test input')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for test input')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Distributed Inference Device Offline Optimization Demo")
    print("="*70)
    print(f"Model params: {args.model_params}")
    print(f"Devices: {args.num_devices}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Batch size: {args.batch_size}")
    print("="*70 + "\n")
    
    # 创建测试输入
    input_ids = torch.randint(0, 1000, (args.batch_size, args.seq_length))
    
    # 创建分布式推理系统
    system = DistributedInferenceSystem(
        model_params_path=args.model_params,
        num_devices=args.num_devices
    )
    
    try:
        # 启动系统
        system.start()
        
        # 运行对比Demo
        comparison = system.run_comparison_demo(input_ids)
        
        # 等待一下让输出完整
        time.sleep(0.5)
        
        print("\n" + "="*70)
        print("Demo Completed Successfully!")
        print("="*70)
        print("\nKey Findings:")
        print(f"  - KV-Cache Reuse Strategy achieved {comparison['speedup']:.2f}x speedup")
        print(f"  - Saved {comparison['computation_saved_percent']:.1f}% of computation")
        print(f"  - Time saved: {comparison['time_saved_ms']:.2f}ms")
        print("\nThis demonstrates the effectiveness of the KV-Cache reuse optimization")
        print("for handling device offline scenarios in distributed inference systems.")
        print("="*70)
        
    finally:
        # 停止系统
        system.stop()


if __name__ == "__main__":
    main()
