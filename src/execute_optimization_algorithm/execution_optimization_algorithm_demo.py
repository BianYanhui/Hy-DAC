"""
分布式推理系统执行优化算法Demo
模拟多设备分布式推理场景，演示设备下线后的KV-Cache复用优化策略,
使用真实的Llama-3.2-1B模型进行推理
"""
import threading
import time
import random
from typing import Dict, List
import sys
import os

# 导入自定义模块
from heartbeat_detection import HeartbeatDetector
from task_reassign import TaskReassigner
from kv_cache_reused import KVCacheManager
from llama_model_loader import LlamaModel


class Worker:
    """Worker节点"""
    
    def __init__(self, worker_id: str, assigned_heads: List[int], leader):
        """
        初始化Worker
        
        Args:
            worker_id: Worker的唯一标识
            assigned_heads: 分配给该Worker的头部列表
            leader: Leader节点引用
        """
        self.worker_id = worker_id
        self.assigned_heads = assigned_heads.copy()
        self.leader = leader
        self.is_running = True
        self.heartbeat_thread = None
        self.is_alive = True
        
    def start(self):
        """启动Worker"""
        self.is_running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        print(f"[Worker-{self.worker_id}] 已启动，负责 Heads: {self.assigned_heads}")
    
    def _heartbeat_loop(self):
        """心跳发送循环"""
        while self.is_running and self.is_alive:
            # 向Leader发送心跳
            self.leader.receive_heartbeat(self.worker_id)
            time.sleep(1.0)  # 每秒发送一次心跳
    
    def simulate_failure(self):
        """模拟Worker失败（停止发送心跳）"""
        print(f"[Worker-{self.worker_id}] 💥 模拟设备下线...")
        self.is_alive = False
        self.is_running = False
    
    def update_heads(self, new_heads: List[int]):
        """更新Worker负责的头部"""
        self.assigned_heads = new_heads.copy()
        print(f"[Worker-{self.worker_id}] 更新任务，现在负责 Heads: {self.assigned_heads}")
    
    def stop(self):
        """停止Worker"""
        self.is_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2.0)


class Leader:
    """Leader节点（同时也是Worker）"""
    
    def __init__(self, leader_id: str, assigned_heads: List[int], 
                 kv_cache_manager: KVCacheManager, task_reassigner: TaskReassigner,
                 heartbeat_detector: HeartbeatDetector):
        """
        初始化Leader
        
        Args:
            leader_id: Leader的唯一标识
            assigned_heads: 分配给Leader的头部列表
            kv_cache_manager: KV-Cache管理器
            task_reassigner: 任务重分配器
            heartbeat_detector: 心跳检测器
        """
        self.leader_id = leader_id
        self.assigned_heads = assigned_heads.copy()
        self.kv_cache_manager = kv_cache_manager
        self.task_reassigner = task_reassigner
        self.heartbeat_detector = heartbeat_detector
        self.workers: Dict[str, Worker] = {}
        self.lock = threading.Lock()
        
        # 设置失败回调
        self.heartbeat_detector.set_failure_callback(self._handle_worker_failure)
    
    def receive_heartbeat(self, worker_id: str):
        """接收Worker的心跳"""
        self.heartbeat_detector.receive_heartbeat(worker_id)
    
    def register_worker(self, worker: Worker):
        """注册Worker"""
        with self.lock:
            self.workers[worker.worker_id] = worker
            self.heartbeat_detector.register_worker(worker.worker_id)
    
    def _handle_worker_failure(self, failed_worker_id: str):
        """处理Worker失败的回调"""
        print(f"\n{'='*60}")
        print(f"[Leader-{self.leader_id}] 🚨 检测到 Worker {failed_worker_id} 下线!")
        print(f"{'='*60}\n")
        
        # 获取所有存活的Worker（包括Leader自己）
        alive_workers = [self.leader_id]
        with self.lock:
            for wid, worker in self.workers.items():
                if self.heartbeat_detector.is_worker_alive(wid):
                    alive_workers.append(wid)
        
        print(f"[Leader-{self.leader_id}] 当前存活的节点: {alive_workers}")
        
        # 执行任务重分配
        print(f"\n[Leader-{self.leader_id}] 开始执行任务重分配...")
        new_assignments = self.task_reassigner.reassign_failed_worker(
            failed_worker_id, alive_workers
        )
        
        if not new_assignments:
            print(f"[Leader-{self.leader_id}] 任务重分配失败或无需重分配")
            return
        
        # 移除失败Worker的KV-Cache
        self.kv_cache_manager.remove_worker_cache(failed_worker_id)
        
        # 执行KV-Cache复用和重计算
        print(f"\n[Leader-{self.leader_id}] 开始执行 KV-Cache 复用和重计算...")
        self._perform_cache_reuse_and_recompute(new_assignments)
        
        print(f"\n{'='*60}")
        print(f"[Leader-{self.leader_id}] ✅ 故障恢复完成!")
        print(f"{'='*60}\n")
    
    def _perform_cache_reuse_and_recompute(self, new_assignments: Dict[str, List[int]]):
        """执行KV-Cache复用和重计算"""
        total_reused = 0
        total_recomputed = 0
        total_time = 0.0
        
        for worker_id, new_heads in new_assignments.items():
            print(f"\n[处理 {worker_id}]")
            
            # 获取该Worker原有的头部
            old_heads = self.task_reassigner.get_worker_heads(worker_id)
            
            # 执行复用和重计算
            compute_time, reused_count, recomputed_count = \
                self.kv_cache_manager.reuse_cache_and_compute_new(
                    worker_id, old_heads, new_heads
                )
            
            total_reused += reused_count
            total_recomputed += recomputed_count
            total_time += compute_time
            
            # 更新Worker的任务（如果是其他Worker）
            with self.lock:
                if worker_id in self.workers:
                    updated_heads = self.task_reassigner.get_worker_heads(worker_id)
                    self.workers[worker_id].update_heads(updated_heads)
                elif worker_id == self.leader_id:
                    # 更新Leader自己的任务
                    self.assigned_heads = self.task_reassigner.get_worker_heads(worker_id)
                    print(f"[Leader-{self.leader_id}] 更新自己的任务，现在负责 Heads: {self.assigned_heads}")
        
        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"KV-Cache 复用统计:")
        print(f"  复用的头部数量: {total_reused}")
        print(f"  重新计算的头部数量: {total_recomputed}")
        print(f"  重计算总耗时: {total_time:.3f} 秒")
        if total_reused + total_recomputed > 0:
            reuse_ratio = total_reused / (total_reused + total_recomputed) * 100
            print(f"  复用率: {reuse_ratio:.1f}%")
        print(f"{'='*60}")


class DistributedInferenceSystem:
    """分布式推理系统"""
    
    def __init__(self, num_heads: int = 16, num_workers: int = 4, 
                 model_path: str = None, use_real_model: bool = True):
        """
        初始化分布式推理系统
        
        Args:
            num_heads: 总头部数量
            num_workers: Worker数量（包括Leader）
            model_path: 模型路径（如果使用真实模型）
            use_real_model: 是否使用真实模型
        """
        self.num_heads = num_heads
        self.num_workers = num_workers
        self.use_real_model = use_real_model
        
        # 加载真实模型（如果需要）
        self.llama_model = None
        if use_real_model and model_path:
            params_path = os.path.join(os.path.dirname(model_path), "params.json")
            self.llama_model = LlamaModel(model_path, params_path)
            # 使用模型的实际head数量
            self.num_heads = self.llama_model.get_num_heads()
            print(f"[System] 使用真实模型，head数量: {self.num_heads}")
        
        # 初始化各个组件
        self.kv_cache_manager = KVCacheManager(
            llama_model=self.llama_model,
            num_layers=self.llama_model.get_num_layers() if self.llama_model else 16,
            hidden_size=self.llama_model.get_head_dim() if self.llama_model else 64
        )
        self.task_reassigner = TaskReassigner()
        self.heartbeat_detector = HeartbeatDetector(check_interval=2.0, timeout=5.0)
        
        # 初始化任务分配
        self.initial_assignments = self._create_initial_assignments()
        self.task_reassigner.initialize_assignments(self.initial_assignments)
        
        # 创建Leader和Workers
        leader_id = "Device-0"
        self.leader = Leader(
            leader_id,
            self.initial_assignments[leader_id],
            self.kv_cache_manager,
            self.task_reassigner,
            self.heartbeat_detector
        )
        
        # 初始化Leader的KV-Cache
        self.kv_cache_manager.initialize_worker_cache(
            leader_id, self.initial_assignments[leader_id]
        )
        
        # 创建其他Workers
        self.workers: List[Worker] = []
        for i in range(1, num_workers):
            worker_id = f"Device-{i}"
            worker = Worker(
                worker_id,
                self.initial_assignments[worker_id],
                self.leader
            )
            self.workers.append(worker)
            self.leader.register_worker(worker)
            
            # 初始化Worker的KV-Cache
            self.kv_cache_manager.initialize_worker_cache(
                worker_id, self.initial_assignments[worker_id]
            )
    
    def _create_initial_assignments(self) -> Dict[str, List[int]]:
        """创建初始的头部分配"""
        assignments = {}
        heads_per_worker = self.num_heads // self.num_workers
        remainder = self.num_heads % self.num_workers
        
        current_head = 1
        for i in range(self.num_workers):
            worker_id = f"Device-{i}"
            # 前面的Worker多分配一个头（如果有余数）
            num_heads_for_worker = heads_per_worker + (1 if i < remainder else 0)
            assignments[worker_id] = list(range(current_head, current_head + num_heads_for_worker))
            current_head += num_heads_for_worker
        
        return assignments
    
    def start(self):
        """启动系统"""
        print(f"\n{'='*60}")
        print(f"分布式推理系统启动")
        print(f"总头部数量: {self.num_heads}")
        print(f"设备数量: {self.num_workers}")
        print(f"{'='*60}\n")
        
        # 启动心跳检测
        self.heartbeat_detector.start_detection()
        
        # 启动所有Workers
        for worker in self.workers:
            worker.start()
        
        print(f"[Leader-{self.leader.leader_id}] 系统启动完成，负责 Heads: {self.leader.assigned_heads}\n")
    
    def simulate_worker_failure(self, worker_index: int, delay: float = 5.0):
        """
        模拟Worker失败
        
        Args:
            worker_index: Worker索引（从1开始，0是Leader）
            delay: 失败前的延迟时间（秒）
        """
        if worker_index < 1 or worker_index >= self.num_workers:
            print(f"⚠️ Worker索引无效: {worker_index}")
            return
        
        def delayed_failure():
            time.sleep(delay)
            self.workers[worker_index - 1].simulate_failure()
        
        failure_thread = threading.Thread(target=delayed_failure, daemon=True)
        failure_thread.start()
    
    def stop(self):
        """停止系统"""
        print(f"\n[System] 正在关闭系统...")
        
        # 停止心跳检测
        self.heartbeat_detector.stop_detection()
        
        # 停止所有Workers
        for worker in self.workers:
            worker.stop()
        
        print(f"[System] 系统已关闭")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("分布式推理系统 - 设备离线优化Demo")
    print("使用真实 Llama-3.2-1B 模型")
    print("="*60 + "\n")
    
    # 模型路径
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/model.safetensors"
    
    # 创建系统：使用真实模型，4个设备
    system = DistributedInferenceSystem(
        num_heads=32,  # Llama-3.2-1B有32个注意力头
        num_workers=4,
        model_path=model_path,
        use_real_model=True
    )
    
    # 启动系统
    system.start()
    
    # 让系统运行一段时间
    print("[Demo] 系统正常运行中...\n")
    time.sleep(3)
    
    # 模拟Device-1下线（5秒后）
    print("[Demo] 将在5秒后模拟 Device-1 下线...\n")
    system.simulate_worker_failure(worker_index=1, delay=5.0)
    
    # 等待故障检测和恢复完成
    time.sleep(15)
    
    # 再次展示当前状态
    print("\n" + "="*60)
    print("最终状态:")
    print("="*60)
    current_assignments = system.task_reassigner.get_current_assignments()
    for device_id, heads in current_assignments.items():
        alive_status = "✓ 在线" if system.heartbeat_detector.is_worker_alive(device_id) or device_id == "Device-0" else "✗ 离线"
        print(f"{device_id}: Heads {heads} - {alive_status}")
    print("="*60 + "\n")
    
    # 停止系统
    system.stop()
    
    print("\n[Demo] Demo运行完成!\n")


if __name__ == "__main__":
    main()
