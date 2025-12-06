#!/bin/bash

# 激活conda环境py310并运行Demo
echo "激活 conda 环境 py310..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py310

echo "检查Python版本和必要的包..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import safetensors; print('safetensors已安装')"

echo ""
echo "开始运行分布式推理系统Demo..."
echo "========================================"
echo ""

python execution_optimization_algorithm_demo.py

echo ""
echo "Demo运行完成!"
