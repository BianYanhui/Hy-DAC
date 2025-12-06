"""
性能对比模块 (Performance Comparator Module)

该模块用于对比两种设备离线恢复策略的性能：
1. KV-Cache复用策略（优化算法）：复用已有缓存，只重计算新分配的Heads
2. 全量重计算策略（传统算法）：清除所有缓存，重新计算所有Heads

通过对比实验，验证KV-Cache复用策略的高效性。
"""

import time
import torch
import copy
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PerformanceMetrics:
    """性能指标"""
    strategy_name: str
    total_time: float  # 总时间（秒）
    computation_time: float  # 计算时间（秒）
    heads_computed: int  # 计算的Head数量
    heads_reused: int  # 复用的Head数量
    memory_saved: float  # 节省的内存（MB）
    speedup_ratio: float = 0.0  # 加速比
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy_name,
            "total_time_ms": self.total_time * 1000,
            "computation_time_ms": self.computation_time * 1000,
            "heads_computed": self.heads_computed,
            "heads_reused": self.heads_reused,
            "memory_saved_mb": self.memory_saved,
            "speedup_ratio": self.speedup_ratio
        }


@dataclass 
class ComparisonResult:
    """对比结果"""
    reuse_metrics: PerformanceMetrics
    full_recompute_metrics: PerformanceMetrics
    speedup: float  # 加速倍数
    computation_saved_percent: float  # 计算节省百分比
    
    def to_dict(self) -> Dict:
        return {
            "kv_cache_reuse": self.reuse_metrics.to_dict(),
            "full_recompute": self.full_recompute_metrics.to_dict(),
            "speedup": self.speedup,
            "computation_saved_percent": self.computation_saved_percent
        }


class PerformanceComparator:
    """
    性能对比器
    
    用于对比KV-Cache复用策略和全量重计算策略的性能。
    """
    
    def __init__(self, 
                 n_kv_heads: int = 8,
                 n_layers: int = 16,
                 head_dim: int = 64,
                 seq_length: int = 512):
        """
        初始化性能对比器
        
        Args:
            n_kv_heads: KV Head数量
            n_layers: Transformer层数
            head_dim: Head维度
            seq_length: 序列长度
        """
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.seq_length = seq_length
        
        # 计算单个Head的KV-Cache大小（字节）
        # K和V各有 [seq_length, head_dim] 的float32
        self.cache_per_head_per_layer = 2 * seq_length * head_dim * 4  # bytes
        self.cache_per_head = self.cache_per_head_per_layer * n_layers
        
        # 存储比较结果
        self.comparison_results: List[ComparisonResult] = []
    
    def simulate_kv_computation(self, num_heads: int, 
                                 num_layers: int = None,
                                 batch_size: int = 1) -> Tuple[float, torch.Tensor]:
        """
        模拟KV-Cache计算
        
        Args:
            num_heads: 要计算的Head数量
            num_layers: 层数（默认使用配置的层数）
            batch_size: 批次大小
            
        Returns:
            (computation_time, dummy_cache): 计算时间和模拟的缓存
        """
        if num_layers is None:
            num_layers = self.n_layers
        
        start_time = time.time()
        
        # 模拟计算
        for layer in range(num_layers):
            for head in range(num_heads):
                # 模拟K和V的计算
                k = torch.randn(batch_size, self.seq_length, self.head_dim)
                v = torch.randn(batch_size, self.seq_length, self.head_dim)
                
                # 模拟一些矩阵运算
                _ = torch.matmul(k, k.transpose(-2, -1))
                _ = torch.matmul(v, v.transpose(-2, -1))
            
            # 模拟层间延迟
            time.sleep(0.001)
        
        computation_time = time.time() - start_time
        
        # 创建dummy cache用于内存计算
        dummy_cache = torch.randn(batch_size, num_heads, self.seq_length, self.head_dim)
        
        return computation_time, dummy_cache
    
    def run_kv_cache_reuse_strategy(self,
                                     total_heads: int,
                                     heads_already_cached: List[int],
                                     heads_to_recompute: List[int]) -> PerformanceMetrics:
        """
        运行KV-Cache复用策略
        
        Args:
            total_heads: 总Head数量
            heads_already_cached: 已有缓存的Heads
            heads_to_recompute: 需要重新计算的Heads
            
        Returns:
            性能指标
        """
        print(f"\n[KV-Cache Reuse Strategy]")
        print(f"  Total heads: {total_heads}")
        print(f"  Heads with cache (reused): {len(heads_already_cached)}")
        print(f"  Heads to recompute: {len(heads_to_recompute)}")
        
        start_time = time.time()
        
        # 只计算需要重新计算的heads
        if heads_to_recompute:
            comp_time, _ = self.simulate_kv_computation(len(heads_to_recompute))
        else:
            comp_time = 0.0
        
        total_time = time.time() - start_time
        
        # 计算节省的内存（通过复用缓存）
        memory_saved = len(heads_already_cached) * self.cache_per_head / (1024 * 1024)
        
        metrics = PerformanceMetrics(
            strategy_name="KV-Cache Reuse",
            total_time=total_time,
            computation_time=comp_time,
            heads_computed=len(heads_to_recompute),
            heads_reused=len(heads_already_cached),
            memory_saved=memory_saved
        )
        
        print(f"  Computation time: {comp_time*1000:.2f}ms")
        print(f"  Total time: {total_time*1000:.2f}ms")
        
        return metrics
    
    def run_full_recompute_strategy(self, total_heads: int) -> PerformanceMetrics:
        """
        运行全量重计算策略
        
        Args:
            total_heads: 总Head数量
            
        Returns:
            性能指标
        """
        print(f"\n[Full Recompute Strategy]")
        print(f"  Total heads to recompute: {total_heads}")
        
        start_time = time.time()
        
        # 计算所有heads
        comp_time, _ = self.simulate_kv_computation(total_heads)
        
        total_time = time.time() - start_time
        
        metrics = PerformanceMetrics(
            strategy_name="Full Recompute",
            total_time=total_time,
            computation_time=comp_time,
            heads_computed=total_heads,
            heads_reused=0,
            memory_saved=0.0
        )
        
        print(f"  Computation time: {comp_time*1000:.2f}ms")
        print(f"  Total time: {total_time*1000:.2f}ms")
        
        return metrics
    
    def compare_strategies(self,
                           heads_per_device: Dict[str, List[int]],
                           offline_device: str,
                           target_device: str) -> ComparisonResult:
        """
        对比两种策略
        
        Args:
            heads_per_device: 每个设备的Head分配 {device_id: [head_ids]}
            offline_device: 离线设备ID
            target_device: 接收任务的目标设备ID
            
        Returns:
            对比结果
        """
        print("\n" + "="*70)
        print("Performance Comparison: KV-Cache Reuse vs Full Recompute")
        print("="*70)
        
        # 获取离线设备的heads
        offline_heads = heads_per_device.get(offline_device, [])
        # 获取目标设备原有的heads
        target_original_heads = heads_per_device.get(target_device, [])
        
        # 计算在线设备的总heads（排除离线设备）
        online_devices = {k: v for k, v in heads_per_device.items() if k != offline_device}
        total_online_heads = sum(len(v) for v in online_devices.values())
        
        print(f"\nScenario:")
        print(f"  Offline device: {offline_device} with heads {offline_heads}")
        print(f"  Target device: {target_device} with original heads {target_original_heads}")
        print(f"  Reassigning heads {offline_heads} to {target_device}")
        
        # 运行KV-Cache复用策略
        # 目标设备可以复用自己原有的缓存，只需要计算新分配的heads
        reuse_metrics = self.run_kv_cache_reuse_strategy(
            total_heads=total_online_heads + len(offline_heads),
            heads_already_cached=target_original_heads,
            heads_to_recompute=offline_heads
        )
        
        # 运行全量重计算策略
        # 所有在线设备的heads都需要重新计算
        full_metrics = self.run_full_recompute_strategy(
            total_heads=total_online_heads + len(offline_heads)
        )
        
        # 计算加速比
        if reuse_metrics.total_time > 0:
            speedup = full_metrics.total_time / reuse_metrics.total_time
        else:
            speedup = float('inf')
        
        # 计算节省的计算百分比
        if full_metrics.heads_computed > 0:
            saved_percent = (1 - reuse_metrics.heads_computed / full_metrics.heads_computed) * 100
        else:
            saved_percent = 0.0
        
        reuse_metrics.speedup_ratio = speedup
        
        result = ComparisonResult(
            reuse_metrics=reuse_metrics,
            full_recompute_metrics=full_metrics,
            speedup=speedup,
            computation_saved_percent=saved_percent
        )
        
        self.comparison_results.append(result)
        
        # 打印对比结果
        print("\n" + "-"*50)
        print("Comparison Results:")
        print("-"*50)
        print(f"  Speedup: {speedup:.2f}x faster")
        print(f"  Computation saved: {saved_percent:.1f}%")
        print(f"  KV-Cache Reuse time: {reuse_metrics.total_time*1000:.2f}ms")
        print(f"  Full Recompute time: {full_metrics.total_time*1000:.2f}ms")
        print(f"  Time saved: {(full_metrics.total_time - reuse_metrics.total_time)*1000:.2f}ms")
        
        return result
    
    def run_multiple_scenarios(self, 
                                num_devices: int = 4,
                                heads_per_scenario: List[int] = None) -> List[ComparisonResult]:
        """
        运行多个场景的对比
        
        Args:
            num_devices: 设备数量
            heads_per_scenario: 每个场景的head数量列表
            
        Returns:
            所有场景的对比结果
        """
        if heads_per_scenario is None:
            heads_per_scenario = [4, 8, 16, 32]
        
        results = []
        
        for n_heads in heads_per_scenario:
            print(f"\n{'#'*70}")
            print(f"Scenario: {n_heads} KV-Heads, {num_devices} devices")
            print(f"{'#'*70}")
            
            # 更新head配置
            self.n_kv_heads = n_heads
            
            # 创建设备分配
            heads_per_device = {}
            heads_per_worker = n_heads // num_devices
            remainder = n_heads % num_devices
            
            current_head = 0
            for i in range(num_devices):
                device_id = f"Device_{i}"
                num_heads_for_device = heads_per_worker + (1 if i < remainder else 0)
                heads_per_device[device_id] = list(range(current_head, current_head + num_heads_for_device))
                current_head += num_heads_for_device
            
            # 选择一个设备离线（随机选择一个非leader）
            offline_device = f"Device_{num_devices - 2}"  # 倒数第二个设备
            target_device = f"Device_{num_devices - 1}"   # 最后一个设备
            
            result = self.compare_strategies(
                heads_per_device=heads_per_device,
                offline_device=offline_device,
                target_device=target_device
            )
            results.append(result)
        
        return results
    
    def plot_comparison_results(self, 
                                 results: List[ComparisonResult] = None,
                                 save_path: str = None):
        """
        绘制对比结果图表
        
        Args:
            results: 对比结果列表
            save_path: 保存路径
        """
        if results is None:
            results = self.comparison_results
        
        if not results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 数据准备
        scenarios = [f"Scenario {i+1}" for i in range(len(results))]
        reuse_times = [r.reuse_metrics.total_time * 1000 for r in results]
        full_times = [r.full_recompute_metrics.total_time * 1000 for r in results]
        speedups = [r.speedup for r in results]
        saved_percents = [r.computation_saved_percent for r in results]
        
        # 图1: 时间对比（柱状图）
        ax1 = axes[0, 0]
        x = np.arange(len(scenarios))
        width = 0.35
        bars1 = ax1.bar(x - width/2, reuse_times, width, label='KV-Cache Reuse', color='#2ecc71')
        bars2 = ax1.bar(x + width/2, full_times, width, label='Full Recompute', color='#e74c3c')
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 图2: 加速比（柱状图）
        ax2 = axes[0, 1]
        bars = ax2.bar(scenarios, speedups, color='#3498db')
        ax2.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Speedup Ratio')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 在柱子上标注数值
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax2.annotate(f'{speedup:.2f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 图3: 计算节省百分比（饼图）
        ax3 = axes[1, 0]
        if len(results) > 0:
            avg_saved = np.mean(saved_percents)
            sizes = [avg_saved, 100 - avg_saved]
            labels = ['Computation Saved', 'Computation Required']
            colors = ['#2ecc71', '#e74c3c']
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'Average Computation Savings')
        
        # 图4: Heads计算对比
        ax4 = axes[1, 1]
        reuse_heads = [r.reuse_metrics.heads_computed for r in results]
        full_heads = [r.full_recompute_metrics.heads_computed for r in results]
        
        x = np.arange(len(scenarios))
        bars1 = ax4.bar(x - width/2, reuse_heads, width, label='KV-Cache Reuse', color='#2ecc71')
        bars2 = ax4.bar(x + width/2, full_heads, width, label='Full Recompute', color='#e74c3c')
        ax4.set_xlabel('Scenario')
        ax4.set_ylabel('Number of Heads Computed')
        ax4.set_title('Heads Computation Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(scenarios)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, 
                        results: List[ComparisonResult] = None,
                        save_path: str = None) -> str:
        """
        生成性能对比报告
        
        Args:
            results: 对比结果列表
            save_path: 保存路径
            
        Returns:
            报告文本
        """
        if results is None:
            results = self.comparison_results
        
        report_lines = [
            "="*70,
            "Performance Comparison Report",
            "KV-Cache Reuse Strategy vs Full Recompute Strategy",
            "="*70,
            "",
            f"Configuration:",
            f"  - Number of Layers: {self.n_layers}",
            f"  - Head Dimension: {self.head_dim}",
            f"  - Sequence Length: {self.seq_length}",
            "",
            "-"*70,
            "Results Summary:",
            "-"*70,
            ""
        ]
        
        for i, result in enumerate(results):
            report_lines.extend([
                f"Scenario {i+1}:",
                f"  KV-Cache Reuse Strategy:",
                f"    - Total Time: {result.reuse_metrics.total_time*1000:.2f}ms",
                f"    - Heads Computed: {result.reuse_metrics.heads_computed}",
                f"    - Heads Reused: {result.reuse_metrics.heads_reused}",
                f"  Full Recompute Strategy:",
                f"    - Total Time: {result.full_recompute_metrics.total_time*1000:.2f}ms",
                f"    - Heads Computed: {result.full_recompute_metrics.heads_computed}",
                f"  Comparison:",
                f"    - Speedup: {result.speedup:.2f}x",
                f"    - Computation Saved: {result.computation_saved_percent:.1f}%",
                ""
            ])
        
        # 统计汇总
        if results:
            avg_speedup = np.mean([r.speedup for r in results])
            avg_saved = np.mean([r.computation_saved_percent for r in results])
            max_speedup = max([r.speedup for r in results])
            
            report_lines.extend([
                "="*70,
                "Overall Statistics:",
                "="*70,
                f"  Average Speedup: {avg_speedup:.2f}x",
                f"  Maximum Speedup: {max_speedup:.2f}x",
                f"  Average Computation Saved: {avg_saved:.1f}%",
                "",
                "Conclusion:",
                f"  The KV-Cache Reuse strategy achieves an average speedup of {avg_speedup:.2f}x",
                f"  compared to the Full Recompute strategy, saving {avg_saved:.1f}% of",
                "  computation by reusing cached KV values.",
                "="*70
            ])
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report


if __name__ == "__main__":
    # 测试代码
    print("Testing Performance Comparator Module")
    
    # 创建性能对比器
    comparator = PerformanceComparator(
        n_kv_heads=8,
        n_layers=4,  # 使用较少的层进行测试
        head_dim=64,
        seq_length=128  # 较短的序列加速测试
    )
    
    # 单个场景测试
    heads_per_device = {
        "Device_0": [0, 1],
        "Device_1": [2, 3],
        "Device_2": [4, 5],
        "Device_3": [6, 7]
    }
    
    result = comparator.compare_strategies(
        heads_per_device=heads_per_device,
        offline_device="Device_1",
        target_device="Device_3"
    )
    
    # 打印详细结果
    print("\n" + "="*70)
    print("Detailed Result:")
    print(json.dumps(result.to_dict(), indent=2))
    
    # 生成报告
    report = comparator.generate_report()
    print("\n" + report)
