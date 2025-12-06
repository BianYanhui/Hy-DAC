"""
心跳检测模块
负责Leader定期检测Worker的存活状态
"""
import threading
import time
from typing import Dict, Set, Callable


class HeartbeatDetector:
    """心跳检测器"""
    
    def __init__(self, check_interval: float = 2.0, timeout: float = 5.0):
        """
        初始化心跳检测器
        
        Args:
            check_interval: 心跳检测间隔（秒）
            timeout: 心跳超时时间（秒）
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self.worker_last_heartbeat: Dict[str, float] = {}
        self.failed_workers: Set[str] = set()
        self.running = False
        self.detection_thread = None
        self.failure_callback = None
        self.lock = threading.Lock()
        
    def register_worker(self, worker_id: str):
        """注册一个Worker"""
        with self.lock:
            self.worker_last_heartbeat[worker_id] = time.time()
            print(f"[HeartbeatDetector] Worker {worker_id} 已注册")
    
    def receive_heartbeat(self, worker_id: str):
        """接收Worker的心跳"""
        with self.lock:
            if worker_id in self.worker_last_heartbeat:
                self.worker_last_heartbeat[worker_id] = time.time()
                # print(f"[HeartbeatDetector] 收到 Worker {worker_id} 的心跳")
    
    def set_failure_callback(self, callback: Callable[[str], None]):
        """设置Worker失败时的回调函数"""
        self.failure_callback = callback
    
    def start_detection(self):
        """启动心跳检测"""
        if self.running:
            print("[HeartbeatDetector] 心跳检测已在运行")
            return
        
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        print("[HeartbeatDetector] 心跳检测已启动")
    
    def stop_detection(self):
        """停止心跳检测"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join()
        print("[HeartbeatDetector] 心跳检测已停止")
    
    def _detection_loop(self):
        """心跳检测循环"""
        while self.running:
            time.sleep(self.check_interval)
            self._check_workers()
    
    def _check_workers(self):
        """检查所有Worker的心跳状态"""
        current_time = time.time()
        newly_failed_workers = []
        
        with self.lock:
            for worker_id, last_heartbeat in list(self.worker_last_heartbeat.items()):
                # 如果已经标记为失败，跳过
                if worker_id in self.failed_workers:
                    continue
                
                # 检查是否超时
                if current_time - last_heartbeat > self.timeout:
                    print(f"[HeartbeatDetector] ⚠️ Worker {worker_id} 心跳超时!")
                    self.failed_workers.add(worker_id)
                    newly_failed_workers.append(worker_id)
        
        # 在锁外调用回调，避免死锁
        if self.failure_callback:
            for worker_id in newly_failed_workers:
                self.failure_callback(worker_id)
    
    def get_failed_workers(self) -> Set[str]:
        """获取已失败的Worker列表"""
        with self.lock:
            return self.failed_workers.copy()
    
    def is_worker_alive(self, worker_id: str) -> bool:
        """检查Worker是否存活"""
        with self.lock:
            return worker_id not in self.failed_workers
