"""
任务重分配模块
负责在Worker下线时重新分配计算任务（Heads）
"""
import random
from typing import Dict, List, Tuple


class TaskReassigner:
    """任务重分配器"""
    
    def __init__(self):
        """初始化任务重分配器"""
        self.head_assignments: Dict[str, List[int]] = {}  # worker_id -> [heads]
        
    def initialize_assignments(self, assignments: Dict[str, List[int]]):
        """
        初始化头部分配
        
        Args:
            assignments: 初始的头部分配，格式为 {worker_id: [head1, head2, ...]}
        """
        self.head_assignments = {k: v.copy() for k, v in assignments.items()}
        print("[TaskReassigner] 初始任务分配:")
        for worker_id, heads in self.head_assignments.items():
            print(f"  {worker_id}: Heads {heads}")
    
    def reassign_failed_worker(self, failed_worker_id: str, alive_workers: List[str]) -> Dict[str, List[int]]:
        """
        重新分配失败Worker的任务
        
        Args:
            failed_worker_id: 失败的Worker ID
            alive_workers: 存活的Worker ID列表
            
        Returns:
            新分配的任务字典，格式为 {worker_id: [new_heads]}
        """
        if failed_worker_id not in self.head_assignments:
            print(f"[TaskReassigner] Worker {failed_worker_id} 不存在于任务列表中")
            return {}
        
        # 获取失败Worker的所有头部
        failed_heads = self.head_assignments[failed_worker_id]
        if not failed_heads:
            print(f"[TaskReassigner] Worker {failed_worker_id} 没有分配任何头部")
            return {}
        
        print(f"[TaskReassigner] Worker {failed_worker_id} 下线，需要重新分配 Heads: {failed_heads}")
        
        # 移除失败Worker
        del self.head_assignments[failed_worker_id]
        
        # 过滤出仍然存活的Worker
        valid_workers = [w for w in alive_workers if w in self.head_assignments]
        
        if not valid_workers:
            print(f"[TaskReassigner] ⚠️ 没有存活的Worker可以接收任务!")
            return {}
        
        # 简单策略：随机选择一个Worker来接收所有失败的头部
        # （实际应用中可以使用更复杂的负载均衡策略）
        target_worker = random.choice(valid_workers)
        
        print(f"[TaskReassigner] 将 Heads {failed_heads} 重新分配给 {target_worker}")
        
        # 分配头部到目标Worker
        self.head_assignments[target_worker].extend(failed_heads)
        
        # 返回新分配的任务
        new_assignments = {target_worker: failed_heads}
        
        print("[TaskReassigner] 重分配后的任务分配:")
        for worker_id, heads in self.head_assignments.items():
            print(f"  {worker_id}: Heads {heads}")
        
        return new_assignments
    
    def get_current_assignments(self) -> Dict[str, List[int]]:
        """获取当前的任务分配"""
        return {k: v.copy() for k, v in self.head_assignments.items()}
    
    def get_worker_heads(self, worker_id: str) -> List[int]:
        """获取指定Worker的头部列表"""
        return self.head_assignments.get(worker_id, []).copy()
