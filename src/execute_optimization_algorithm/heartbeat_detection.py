""",
心跳检测模块 (Heartbeat Detection Module)

该模块实现了分布式系统中的心跳检测机制。
Leader定期向所有Worker发送心跳请求，以检测Worker是否在线。
如果Worker在指定时间内没有响应，则认为该Worker已离线。
"""

import time
import threading
from typing import Dict, List, Callable, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import queue


class DeviceStatus(Enum):
    """设备状态枚举"""
    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass
class HeartbeatRecord:
    """心跳记录"""
    device_id: str
    last_heartbeat_time: float = 0.0
    status: DeviceStatus = DeviceStatus.UNKNOWN
    consecutive_failures: int = 0
    total_heartbeats: int = 0
    total_failures: int = 0


class HeartbeatDetector:
    """
    心跳检测器
    
    Leader使用该类来检测所有Worker的存活状态。
    采用线程模拟实际的网络心跳检测。
    """
    
    def __init__(self, 
                 heartbeat_interval: float = 1.0,
                 timeout: float = 0.5,
                 max_failures: int = 3,
                 on_device_offline: Optional[Callable[[str], None]] = None,
                 on_device_online: Optional[Callable[[str], None]] = None):
        """
        初始化心跳检测器
        
        Args:
            heartbeat_interval: 心跳检测间隔（秒）
            timeout: 单次心跳响应超时时间（秒）
            max_failures: 连续失败多少次后认为设备离线
            on_device_offline: 设备离线时的回调函数
            on_device_online: 设备上线时的回调函数
        """
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self.max_failures = max_failures
        self.on_device_offline = on_device_offline
        self.on_device_online = on_device_online
        
        # 设备记录
        self.devices: Dict[str, HeartbeatRecord] = {}
        
        # 设备响应队列 - 用于模拟心跳响应
        self.response_queues: Dict[str, queue.Queue] = {}
        
        # 控制标志
        self.running = False
        self.detector_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # 手动设置的离线设备（用于模拟）
        self.simulated_offline_devices: Set[str] = set()
    
    def register_device(self, device_id: str):
        """
        注册设备到心跳检测
        
        Args:
            device_id: 设备ID
        """
        with self.lock:
            if device_id not in self.devices:
                self.devices[device_id] = HeartbeatRecord(
                    device_id=device_id,
                    last_heartbeat_time=time.time(),
                    status=DeviceStatus.ONLINE
                )
                self.response_queues[device_id] = queue.Queue()
                print(f"[Heartbeat] Device {device_id} registered")
    
    def unregister_device(self, device_id: str):
        """注销设备"""
        with self.lock:
            if device_id in self.devices:
                del self.devices[device_id]
            if device_id in self.response_queues:
                del self.response_queues[device_id]
    
    def simulate_device_offline(self, device_id: str):
        """
        模拟设备离线（用于测试）
        
        Args:
            device_id: 要模拟离线的设备ID
        """
        with self.lock:
            self.simulated_offline_devices.add(device_id)
            print(f"[Heartbeat] Simulating {device_id} going offline...")
    
    def simulate_device_online(self, device_id: str):
        """模拟设备恢复在线"""
        with self.lock:
            self.simulated_offline_devices.discard(device_id)
            if device_id in self.devices:
                self.devices[device_id].consecutive_failures = 0
    
    def _send_heartbeat(self, device_id: str) -> bool:
        """
        向设备发送心跳请求
        
        Returns:
            是否收到响应
        """
        # 检查是否被模拟为离线
        if device_id in self.simulated_offline_devices:
            return False
        
        # 模拟网络延迟
        time.sleep(0.01)  # 10ms的模拟延迟
        
        # 正常情况下设备会响应
        return True
    
    def _check_device(self, device_id: str):
        """检查单个设备的心跳"""
        record = self.devices.get(device_id)
        if not record:
            return
        
        record.total_heartbeats += 1
        
        # 发送心跳并等待响应
        response = self._send_heartbeat(device_id)
        
        with self.lock:
            if response:
                # 收到响应
                old_status = record.status
                record.last_heartbeat_time = time.time()
                record.consecutive_failures = 0
                record.status = DeviceStatus.ONLINE
                
                # 如果之前是离线状态，触发上线回调
                if old_status == DeviceStatus.OFFLINE:
                    if self.on_device_online:
                        self.on_device_online(device_id)
                    print(f"[Heartbeat] Device {device_id} is back ONLINE")
            else:
                # 没有响应
                record.consecutive_failures += 1
                record.total_failures += 1
                
                if record.consecutive_failures >= self.max_failures:
                    if record.status != DeviceStatus.OFFLINE:
                        record.status = DeviceStatus.OFFLINE
                        print(f"[Heartbeat] Device {device_id} detected as OFFLINE "
                              f"(no response for {self.max_failures} consecutive checks)")
                        
                        # 触发离线回调
                        if self.on_device_offline:
                            # 在新线程中执行回调，避免阻塞心跳检测
                            threading.Thread(
                                target=self.on_device_offline,
                                args=(device_id,),
                                daemon=True
                            ).start()
    
    def _heartbeat_loop(self):
        """心跳检测主循环"""
        print("[Heartbeat] Heartbeat detection started")
        
        while self.running:
            start_time = time.time()
            
            # 获取当前所有设备ID的快照
            with self.lock:
                device_ids = list(self.devices.keys())
            
            # 检查每个设备
            for device_id in device_ids:
                if not self.running:
                    break
                self._check_device(device_id)
            
            # 计算需要等待的时间
            elapsed = time.time() - start_time
            sleep_time = max(0, self.heartbeat_interval - elapsed)
            
            if self.running and sleep_time > 0:
                time.sleep(sleep_time)
        
        print("[Heartbeat] Heartbeat detection stopped")
    
    def start(self):
        """启动心跳检测"""
        if self.running:
            return
        
        self.running = True
        self.detector_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="HeartbeatDetector"
        )
        self.detector_thread.start()
    
    def stop(self):
        """停止心跳检测"""
        self.running = False
        if self.detector_thread:
            self.detector_thread.join(timeout=2.0)
            self.detector_thread = None
    
    def get_device_status(self, device_id: str) -> DeviceStatus:
        """获取设备状态"""
        with self.lock:
            if device_id in self.devices:
                return self.devices[device_id].status
            return DeviceStatus.UNKNOWN
    
    def get_all_status(self) -> Dict[str, DeviceStatus]:
        """获取所有设备状态"""
        with self.lock:
            return {
                device_id: record.status
                for device_id, record in self.devices.items()
            }
    
    def get_online_devices(self) -> List[str]:
        """获取所有在线设备"""
        with self.lock:
            return [
                device_id for device_id, record in self.devices.items()
                if record.status == DeviceStatus.ONLINE
            ]
    
    def get_offline_devices(self) -> List[str]:
        """获取所有离线设备"""
        with self.lock:
            return [
                device_id for device_id, record in self.devices.items()
                if record.status == DeviceStatus.OFFLINE
            ]
    
    def get_statistics(self) -> Dict:
        """获取心跳检测统计信息"""
        with self.lock:
            return {
                device_id: {
                    "status": record.status.value,
                    "last_heartbeat": record.last_heartbeat_time,
                    "consecutive_failures": record.consecutive_failures,
                    "total_heartbeats": record.total_heartbeats,
                    "total_failures": record.total_failures,
                    "success_rate": (
                        (record.total_heartbeats - record.total_failures) / 
                        record.total_heartbeats * 100
                        if record.total_heartbeats > 0 else 0
                    )
                }
                for device_id, record in self.devices.items()
            }
    
    def print_status(self):
        """打印当前状态"""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("Heartbeat Detection Status")
        print("="*60)
        for device_id, info in stats.items():
            status = info["status"].upper()
            success_rate = info["success_rate"]
            print(f"  {device_id}: [{status}] Success Rate: {success_rate:.1f}%")
        print("="*60 + "\n")


class WorkerHeartbeatResponder:
    """
    Worker端心跳响应器
    
    Worker使用该类来响应Leader的心跳请求。
    """
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.running = False
        self.is_responsive = True  # 用于模拟离线
    
    def set_responsive(self, responsive: bool):
        """设置是否响应心跳"""
        self.is_responsive = responsive
    
    def respond_to_heartbeat(self) -> bool:
        """响应心跳请求"""
        if not self.is_responsive:
            return False
        return True


if __name__ == "__main__":
    # 测试代码
    print("Testing Heartbeat Detection Module")
    
    offline_events = []
    
    def on_offline(device_id):
        print(f"[Callback] Device {device_id} went offline!")
        offline_events.append(device_id)
    
    # 创建心跳检测器
    detector = HeartbeatDetector(
        heartbeat_interval=0.5,
        timeout=0.2,
        max_failures=2,
        on_device_offline=on_offline
    )
    
    # 注册设备
    for i in range(4):
        detector.register_device(f"Device_{i}")
    
    # 启动心跳检测
    detector.start()
    
    print("Waiting for initial heartbeats...")
    time.sleep(1)
    detector.print_status()
    
    # 模拟设备1离线
    print("\nSimulating Device_1 going offline...")
    detector.simulate_device_offline("Device_1")
    
    # 等待检测到离线
    time.sleep(2)
    detector.print_status()
    
    print(f"\nOffline events detected: {offline_events}")
    
    # 停止检测
    detector.stop()
    print("Test completed.")
