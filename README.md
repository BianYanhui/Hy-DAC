# Hy-DAC

面向分布式张量并行推理的设备离线优化工程实现，核心思路是：当某个计算设备离线（掉线/故障）后，通过任务重分配与 KV-Cache 复用，仅对新增任务进行增量重计算，从而避免“全量清空+重算”的高开销；与传统策略相比，可显著降低恢复延迟与计算量。


## 核心能力

- 分布式推理环境模拟：Leader + 多个 Worker 的线程化模拟，含心跳检测与设备状态管理。
- 任务重分配策略：设备离线后，将其负责的 KV-Heads 重新分配给在线设备（支持集中式/均匀分配）。
- KV-Cache 复用：对已有缓存的 Heads 直接复用，仅为新分配的 Heads 计算 KV。
- 与传统策略对比：提供性能对比器，评估“复用策略 vs 全量重计算”的时间与计算节省效果。
- 真实模型示例：提供基于 Llama-3.2-1B 的真实 KV 计算 Demo（需本地模型权重）。


## 目录结构

```
Hy-DAC/
├─ LICENSE
├─ README.md            # 项目说明（本文件）
└─ src/
	 ├─ adaptive_algorithms/       # 预留/扩展：自适应算法（如调度/分配策略）
	 └─ execute_optimization_algorithm/
			├─ execution_optimization_algorithm_demo.py       # 模拟环境下的离线优化 Demo
			├─ execution_optimization_algorithm_real_demo.py  # 真实 Llama 模型下的优化 Demo
			├─ heartbeat_detection.py                         # 心跳检测（Leader 监控 Worker 存活）
			├─ kv_cache_reused.py                             # KV-Cache 管理与复用引擎
			├─ llama_model.py                                 # Llama-3.2-1B 轻量实现与加载
			├─ performance_comparator.py                      # 性能对比（复用 vs 全量重算）
			└─ task_reassign.py                               # 任务重分配管理器
```


## 环境依赖

- Python 3.9+
- PyTorch（CPU 或 GPU 均可；真实模型 Demo 在 CPU 上亦可运行但较慢）
- 其他：numpy、safetensors、matplotlib（用于可视化，若仅运行模拟 Demo 可选装）

建议使用虚拟环境进行隔离：

```bash
# 可选：创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装常用依赖（按需）
pip install torch numpy safetensors matplotlib
```


## 快速开始

以下命令默认在 macOS 的 zsh 中执行，工作目录为仓库根目录。

### 1) 运行模拟环境 Demo（推荐先体验）

基于简化的注意力模块与线程化的分布式环境模拟，演示心跳检测、设备离线、任务重分配与 KV-Cache 复用。

```bash
cd src/execute_optimization_algorithm
python execution_optimization_algorithm_demo.py
```

运行后可观察：
- 系统初始化与设备分配（KV-Heads）
- 初始 KV-Cache 计算耗时
- 模拟某设备离线后的“复用策略”和“全量重算策略”对比（打印加速比、节省比例等）


### 2) 运行真实模型 Demo（需本地权重）

该 Demo 使用真实的 Llama-3.2-1B 模型进行 KV 计算，验证复用策略在真实模型上的效果。

文件 `src/execute_optimization_algorithm/execution_optimization_algorithm_real_demo.py` 中包含两处本地路径：

```python
model_path = "/path/to/Llama-3.2-1B/model.safetensors"
params_path = "/path/to/Llama-3.2-1B/params.json"
```

请将以上路径替换为你本地模型权重与配置的位置，然后执行：

```bash
cd src/execute_optimization_algorithm
python execution_optimization_algorithm_real_demo.py
```

提示：真实模型 Demo 在 CPU 上可运行但速度较慢；若有 GPU 环境，建议在代码中将 `device="cpu"` 改为合适的 CUDA 设备。


## 功能模块概览

- `heartbeat_detection.py`
	- 提供心跳检测器（Leader 周期性检查 Worker）；支持模拟离线/恢复与事件回调。
- `task_reassign.py`
	- 设备任务（KV-Heads）管理与重分配；支持集中式/均匀分配，易于替换为更先进策略。
- `kv_cache_reused.py`
	- KV-Cache 管理器与复用引擎：跟踪每层/每个 Head 的缓存命中与缺失，只对缺失部分增量计算。
- `performance_comparator.py`
	- 对比“KV-Cache 复用”与“全量重算”的总时延/计算量/加速比等指标。
- `llama_model.py`
	- 轻量级 Llama-3.2-1B 组件与加载器：支持按 Head 计算 KV，便于分布式拆分与复用。
- `execution_optimization_algorithm_demo.py`
	- 线程化模拟分布式推理的端到端示例：心跳检测 → 离线 → 重分配 → 复用策略对比。
- `execution_optimization_algorithm_real_demo.py`
	- 基于真实 Llama 模型的端到端示例：与模拟版流程一致，但计算来自真实模型。


## 结果与可视化

运行脚本后，终端将打印关键指标：
- 初始 KV 计算耗时
- 设备离线后复用策略的增量计算耗时
- 全量重算策略的总计算耗时
- 加速比（speedup）与节省的计算比例（computation saved %）

如需进一步可视化，可在 `performance_comparator.py` 中使用 matplotlib 将多场景结果绘图（脚本内已引入 `matplotlib.pyplot`）。


## 常见问题（FAQ）

1. 没有 requirements.txt？
	 - 本仓库依赖较少且可选，建议按“环境依赖”章节手动安装；如需，我们也可以补充 `requirements.txt`（欢迎提 Issue）。
2. 真实模型路径从哪里来？
	 - 请将你本地的 Llama-3.2-1B `model.safetensors` 与 `params.json` 的路径填写到代码中对应变量。
3. 一直在 CPU 上跑很慢？
	 - 可将 Demo 中引擎/模型的 `device` 参数改为 CUDA 设备（如 `cuda:0`），前提是环境安装了支持的 PyTorch。
4. 我能替换任务重分配策略吗？
	 - 可以。`task_reassign.py` 目前实现了简单策略，你可以在其上实现基于负载/拓扑/带宽感知的高级策略。


## 许可

本项目遵循仓库中的 `LICENSE` 许可文件。


## 引用

如果本项目或相关论文对你的研究/产品有帮助，欢迎在文献中引用（BibTeX 占位，待论文公开版本更新）：

```bibtex
@misc{hy_dac_2025,
	title  = {Hy-DAC: Hybrid Device-Aware Cache Reuse for Distributed Inference},
	author = {Bian, Yanhui and collaborators},
	year   = {2025},
	note   = {Engineering repository: https://github.com/BianYanhui/Hy-DAC}
}
```

—— 欢迎提交 Issue/PR 共同完善：例如更丰富的调度策略、更多模型适配与评测脚本等。

