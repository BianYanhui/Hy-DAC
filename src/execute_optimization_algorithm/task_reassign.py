"""
任务重分配模块 (Task Reassignment Module)

该模块负责在设备离线时，将离线设备的计算任务（Heads）重新分配给其他在线设备。
当前实现采用随机分配策略，后续可以根据需要修改为更复杂的分配算法。
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import threading


@dataclass
class DeviceInfo:
    """设备信息类"""
    device_id: str
    is_leader: bool = False
    is_online: bool = True
    assigned_heads: List[int] = field(default_factory=list)  # 分配给该设备的Head列表
    kv_cache: Dict[int, any] = field(default_factory=dict)  # 该设备持有的KV-Cache {head_id: cache}


class TaskReassignManager:
    """
    任务重分配管理器
    
    负责管理所有设备的任务分配，并在设备离线时进行任务重分配。
    """
    
    def __init__(self, total_heads: int, device_ids: List[str], 
                 head_distribution: Optional[Dict[str, List[int]]] = None):
        """
        初始化任务重分配管理器
        
        Args:
            total_heads: 总的Head数量
            device_ids: 设备ID列表，第一个设备默认为Leader
            head_distribution: 可选的初始Head分配方案 {device_id: [head_ids]}
                             如果不提供，则均匀分配
        """
        self.total_heads = total_heads
        self.devices: Dict[str, DeviceInfo] = {}
        self.lock = threading.Lock()
        
        # 初始化设备
        for i, device_id in enumerate(device_ids):
            self.devices[device_id] = DeviceInfo(
                device_id=device_id,
                is_leader=(i == 0),  # 第一个设备是Leader
                is_online=True,
                assigned_heads=[],
                kv_cache={}
            )
        
        # 分配Heads
        if head_distribution:
            self._apply_distribution(head_distribution)
        else:
            self._distribute_heads_evenly()
    
    def _distribute_heads_evenly(self):
        """均匀分配Heads给所有设备"""
        device_ids = list(self.devices.keys())
        num_devices = len(device_ids)
        
        # 均匀分配
        for head_id in range(self.total_heads):
            device_idx = head_id % num_devices
            self.devices[device_ids[device_idx]].assigned_heads.append(head_id)
        
        # 排序每个设备的heads
        for device in self.devices.values():
            device.assigned_heads.sort()
    
    def _apply_distribution(self, distribution: Dict[str, List[int]]):
        """应用指定的Head分配方案"""
        for device_id, heads in distribution.items():
            if device_id in self.devices:
                self.devices[device_id].assigned_heads = sorted(heads)
    
    def set_head_distribution(self, distribution: Dict[str, List[int]]):
        """
        设置Head分配比例
        
        Args:
            distribution: Head分配方案 {device_id: [head_ids]}
        """
        with self.lock:
            self._apply_distribution(distribution)
    
    def get_device_heads(self, device_id: str) -> List[int]:
        """获取设备分配的Heads"""
        if device_id in self.devices:
            return self.devices[device_id].assigned_heads.copy()
        return []
    
    def get_all_distributions(self) -> Dict[str, List[int]]:
        """获取所有设备的Head分配情况"""
        return {
            device_id: device.assigned_heads.copy()
            for device_id, device in self.devices.items()
        }
    
    def get_online_devices(self) -> List[str]:
        """获取所有在线设备"""
        return [
            device_id for device_id, device in self.devices.items()
            if device.is_online
        ]
    
    def get_offline_devices(self) -> List[str]:
        """获取所有离线设备"""
        return [
            device_id for device_id, device in self.devices.items()
            if not device.is_online
        ]
    
    def mark_device_offline(self, device_id: str) -> bool:
        """
        标记设备为离线状态
        
        Args:
            device_id: 设备ID
            
        Returns:
            是否成功标记
        """
        with self.lock:
            if device_id in self.devices:
                self.devices[device_id].is_online = False
                return True
            return False
    
    def mark_device_online(self, device_id: str) -> bool:
        """标记设备为在线状态"""
        with self.lock:
            if device_id in self.devices:
                self.devices[device_id].is_online = True
                return True
            return False
    
    def reassign_tasks(self, offline_device_id: str, 
                       target_device_id: Optional[str] = None) -> Tuple[str, List[int]]:
        """
        重新分配离线设备的任务
        
        当设备离线时，将其任务分配给其他在线设备。
        当前采用随机分配策略，将所有任务分配给一个随机选择的在线设备。
        
        Args:
            offline_device_id: 离线设备的ID
            target_device_id: 可选的目标设备ID，如果不指定则随机选择
            
        Returns:
            (target_device_id, reassigned_heads): 接收任务的设备ID和被重分配的Heads列表
        """
        with self.lock:
            if offline_device_id not in self.devices:
                raise ValueError(f"Unknown device: {offline_device_id}")
            
            offline_device = self.devices[offline_device_id]
            heads_to_reassign = offline_device.assigned_heads.copy()
            
            if not heads_to_reassign:
                return ("", [])
            
            # 获取在线设备列表（排除离线设备）
            online_devices = [
                device_id for device_id, device in self.devices.items()
                if device.is_online and device_id != offline_device_id
            ]
            
            if not online_devices:
                raise RuntimeError("No online devices available for task reassignment")
            
            # 选择目标设备
            if target_device_id and target_device_id in online_devices:
                selected_device = target_device_id
            else:
                # 随机选择一个在线设备
                selected_device = random.choice(online_devices)
            
            # 执行任务重分配
            self.devices[selected_device].assigned_heads.extend(heads_to_reassign)
            self.devices[selected_device].assigned_heads.sort()
            offline_device.assigned_heads = []
            
            return (selected_device, heads_to_reassign)
    
    def reassign_tasks_distributed(self, offline_device_id: str) -> Dict[str, List[int]]:
        """
        分布式重分配离线设备的任务
        
        将离线设备的任务均匀分配给所有其他在线设备。
        
        Args:
            offline_device_id: 离线设备的ID
            
        Returns:
            {device_id: [newly_assigned_heads]}: 每个设备新分配的Heads
        """
        with self.lock:
            if offline_device_id not in self.devices:
                raise ValueError(f"Unknown device: {offline_device_id}")
            
            offline_device = self.devices[offline_device_id]
            heads_to_reassign = offline_device.assigned_heads.copy()
            
            if not heads_to_reassign:
                return {}
            
            # 获取在线设备列表
            online_devices = [
                device_id for device_id, device in self.devices.items()
                if device.is_online and device_id != offline_device_id
            ]
            
            if not online_devices:
                raise RuntimeError("No online devices available for task reassignment")
            
            # 均匀分配给在线设备
            reassignment_result = {device_id: [] for device_id in online_devices}
            
            for i, head_id in enumerate(heads_to_reassign):
                target_device = online_devices[i % len(online_devices)]
                reassignment_result[target_device].append(head_id)
                self.devices[target_device].assigned_heads.append(head_id)
            
            # 清空离线设备的任务
            offline_device.assigned_heads = []
            
            # 排序
            for device_id in online_devices:
                self.devices[device_id].assigned_heads.sort()
            
            # 只返回有新任务的设备
            return {k: v for k, v in reassignment_result.items() if v}
    
    def get_reassignment_info(self) -> Dict:
        """获取当前分配状态的详细信息"""
        return {
            "total_heads": self.total_heads,
            "devices": {
                device_id: {
                    "is_leader": device.is_leader,
                    "is_online": device.is_online,
                    "assigned_heads": device.assigned_heads,
                    "head_count": len(device.assigned_heads)
                }
                for device_id, device in self.devices.items()
            }
        }
    
    def print_status(self):
        """打印当前分配状态"""
        print("\n" + "="*60)
        print("Task Distribution Status")
        print("="*60)
        for device_id, device in self.devices.items():
            status = "ONLINE" if device.is_online else "OFFLINE"
            role = "LEADER" if device.is_leader else "WORKER"
            heads_str = f"Heads {device.assigned_heads}" if device.assigned_heads else "No heads"
            print(f"  {device_id}: [{status}] [{role}] {heads_str}")
        print("="*60 + "\n")


# 便捷函数
def create_task_manager(total_heads: int, num_devices: int, 
                        device_prefix: str = "Device") -> TaskReassignManager:
    """
    创建任务管理器的便捷函数
    
    Args:
        total_heads: 总Head数量
        num_devices: 设备数量
        device_prefix: 设备名称前缀
        
    Returns:
        TaskReassignManager实例
    """
    device_ids = [f"{device_prefix}_{i}" for i in range(num_devices)]
    return TaskReassignManager(total_heads, device_ids)


if __name__ == "__main__":
    # 测试代码
    print("Testing Task Reassignment Module")
    
    # 创建4个设备，8个Heads
    manager = create_task_manager(total_heads=8, num_devices=4)
    
    print("\nInitial distribution:")
    manager.print_status()
    
    # 模拟设备1离线
    print("Simulating Device_1 going offline...")
    manager.mark_device_offline("Device_1")
    target, reassigned = manager.reassign_tasks("Device_1")
    
    print(f"Heads {reassigned} reassigned to {target}")
    manager.print_status()
