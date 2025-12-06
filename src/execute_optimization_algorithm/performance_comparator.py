"""
性能对比分析模块
用于对比KV-Cache复用方法与传统完全重计算方法的性能差异
"""
from typing import Dict, List, Tuple
import time


class PerformanceComparator:
    """性能对比器"""
    
    def __init__(self):
        """初始化性能对比器"""
        self.traditional_results = []
        self.optimized_results = []
        
    def add_comparison(self, scenario: str, traditional_time: float, optimized_time: float,
                      total_heads: int, reused_heads: int, recomputed_heads: int):
        """
        添加一次对比结果
        
        Args:
            scenario: 场景描述
            traditional_time: 传统方法耗时
            optimized_time: 优化方法耗时
            total_heads: 总头部数量
            reused_heads: 复用的头部数量
            recomputed_heads: 重计算的头部数量
        """
        self.traditional_results.append({
            'scenario': scenario,
            'time': traditional_time,
            'heads': total_heads
        })
        
        self.optimized_results.append({
            'scenario': scenario,
            'time': optimized_time,
            'total_heads': total_heads,
            'reused_heads': reused_heads,
            'recomputed_heads': recomputed_heads
        })
    
    def generate_report(self) -> str:
        """
        生成详细的对比报告
        
        Returns:
            报告字符串
        """
        if not self.traditional_results or not self.optimized_results:
            return "没有对比数据"
        
        report_lines = []
        report_lines.append("\n" + "="*80)
        report_lines.append("KV-Cache复用优化 vs 传统完全重计算 - 性能对比报告")
        report_lines.append("="*80)
        
        total_traditional_time = 0.0
        total_optimized_time = 0.0
        total_heads_processed = 0
        total_reused = 0
        total_recomputed = 0
        
        for i, (trad, opt) in enumerate(zip(self.traditional_results, self.optimized_results), 1):
            report_lines.append(f"\n场景 {i}: {trad['scenario']}")
            report_lines.append("-" * 80)
            
            report_lines.append(f"\n传统方法（完全重计算）:")
            report_lines.append(f"  需要计算的头部: {trad['heads']} 个")
            report_lines.append(f"  计算耗时: {trad['time']:.3f} 秒")
            
            report_lines.append(f"\n优化方法（KV-Cache复用）:")
            report_lines.append(f"  复用的头部: {opt['reused_heads']} 个")
            report_lines.append(f"  重新计算的头部: {opt['recomputed_heads']} 个")
            report_lines.append(f"  计算耗时: {opt['time']:.3f} 秒")
            
            reuse_ratio = (opt['reused_heads'] / opt['total_heads'] * 100) if opt['total_heads'] > 0 else 0
            report_lines.append(f"  复用率: {reuse_ratio:.1f}%")
            
            time_saved = trad['time'] - opt['time']
            speedup = trad['time'] / opt['time'] if opt['time'] > 0 else float('inf')
            improvement = (time_saved / trad['time'] * 100) if trad['time'] > 0 else 0
            
            report_lines.append(f"\n性能提升:")
            report_lines.append(f"  节省时间: {time_saved:.3f} 秒")
            report_lines.append(f"  加速比: {speedup:.2f}x")
            report_lines.append(f"  性能提升: {improvement:.1f}%")
            
            total_traditional_time += trad['time']
            total_optimized_time += opt['time']
            total_heads_processed += opt['total_heads']
            total_reused += opt['reused_heads']
            total_recomputed += opt['recomputed_heads']
        
        # 总体统计
        report_lines.append("\n" + "="*80)
        report_lines.append("总体统计:")
        report_lines.append("="*80)
        
        report_lines.append(f"\n总共处理的头部数量: {total_heads_processed} 个")
        report_lines.append(f"  复用的头部: {total_reused} 个")
        report_lines.append(f"  重新计算的头部: {total_recomputed} 个")
        
        overall_reuse_ratio = (total_reused / total_heads_processed * 100) if total_heads_processed > 0 else 0
        report_lines.append(f"  总体复用率: {overall_reuse_ratio:.1f}%")
        
        report_lines.append(f"\n总耗时对比:")
        report_lines.append(f"  传统方法总耗时: {total_traditional_time:.3f} 秒")
        report_lines.append(f"  优化方法总耗时: {total_optimized_time:.3f} 秒")
        
        total_time_saved = total_traditional_time - total_optimized_time
        overall_speedup = total_traditional_time / total_optimized_time if total_optimized_time > 0 else float('inf')
        overall_improvement = (total_time_saved / total_traditional_time * 100) if total_traditional_time > 0 else 0
        
        report_lines.append(f"\n总体性能提升:")
        report_lines.append(f"  总节省时间: {total_time_saved:.3f} 秒")
        report_lines.append(f"  总体加速比: {overall_speedup:.2f}x")
        report_lines.append(f"  总体性能提升: {overall_improvement:.1f}%")
        
        # 结论
        report_lines.append("\n" + "="*80)
        report_lines.append("结论:")
        report_lines.append("="*80)
        
        if overall_improvement > 0:
            report_lines.append(f"\n✅ KV-Cache复用优化方法显著优于传统完全重计算方法!")
            report_lines.append(f"   通过复用已有的KV-Cache，减少了 {overall_improvement:.1f}% 的计算时间。")
            report_lines.append(f"   在设备下线场景中，{overall_reuse_ratio:.1f}% 的KV-Cache可以被复用，")
            report_lines.append(f"   只需重新计算 {total_recomputed} 个头部，而不是全部 {total_heads_processed} 个。")
        else:
            report_lines.append(f"\n⚠️ 优化效果不明显，可能需要调整策略。")
        
        report_lines.append("\n" + "="*80 + "\n")
        
        return "\n".join(report_lines)
    
    def print_report(self):
        """打印对比报告"""
        print(self.generate_report())
    
    def save_report(self, filepath: str):
        """
        保存对比报告到文件
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        print(f"\n[PerformanceComparator] 性能对比报告已保存到: {filepath}")
