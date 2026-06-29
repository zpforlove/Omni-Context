#!/usr/bin/env bash
# ============================================================================
# Omni-Context 统一评测环境搭建脚本
# 环境名: omni-context  (基于 /cpfs_speech3/yulian.zpf/anaconda3)
# 目标: 一个环境跑通 Qwen3-Omni / MiniCPM-o-4.5 / Ming-flash-omni
# 策略: 以 Qwen3-Omni 所需的较新 transformers 为主线先跑通；其余自定义建模代码
#       的模型若出现版本冲突，再在本脚本末尾按需 pin（见 NOTES）。
# ============================================================================
set -e
ENV_NAME=omni-context
PY_VER=3.11
CONDA_BASE=/cpfs_speech3/yulian.zpf/anaconda3

source ${CONDA_BASE}/etc/profile.d/conda.sh

if conda env list | grep -qE "^${ENV_NAME}\s|/${ENV_NAME}$"; then
  echo "[setup] env ${ENV_NAME} already exists, skip create"
else
  echo "[setup] creating conda env ${ENV_NAME} (python ${PY_VER})"
  conda create -y -n ${ENV_NAME} python=${PY_VER}
fi

conda activate ${ENV_NAME}
python -V

PIP="python -m pip"
$PIP install --upgrade pip setuptools wheel

# ---- 核心推理栈 ----------------------------------------------------------
# !! 本机驱动 470.199.02 仅支持 CUDA 11.4 → 必须用 cu118 的 torch !!
# 已验证可用环境(ming) 用 torch 2.1.2+cu118；这里选更新的 2.4.1+cu118 以兼容
# transformers 4.57，失败回退 2.1.2+cu118。wheel 走 aliyun pytorch-wheels 镜像。
# 用 2.1.2+cu118：该 wheel 自带 CUDA 库(libcudart .so.11.0)、无 nvidia-* 外部依赖，
# 已在本节点 ming 环境验证 cuda 可用。直链安装(aliyun pytorch-wheels 扁平目录)。
# (注：2.4.1+cu118 体积小、会错误拉取 cu12 的 nvidia 依赖导致 CUDA 不可用，故不用。)
TB="https://mirrors.aliyun.com/pytorch-wheels/cu118"
$PIP install \
  "${TB}/torch-2.1.2%2Bcu118-cp311-cp311-linux_x86_64.whl" \
  "${TB}/torchaudio-2.1.2%2Bcu118-cp311-cp311-linux_x86_64.whl"
# 清除可能被误装的 cu12 nvidia 运行库，避免与 cu118 自带库冲突
$PIP uninstall -y nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 \
  nvidia-cudnn-cu12 nvidia-cublas-cu12 2>/dev/null || true
python -c "import torch;print('[setup] torch',torch.__version__,'cuda_avail',torch.cuda.is_available())"

# transformers: Qwen3-Omni (qwen3_omni_moe) 需要 >=4.57
$PIP install "transformers==4.57.1" "accelerate>=1.0.0" "tokenizers>=0.21"
$PIP install "qwen-omni-utils" "qwen-vl-utils"

# ---- 音频 / 评测工具 ------------------------------------------------------
$PIP install "numpy<2.0" librosa soundfile scipy   # torch 2.1.2 需 numpy 1.x ABI
$PIP install jiwer rapidfuzz rouge-score sacrebleu jieba regex
$PIP install pyyaml tqdm pandas tabulate

# ---- flash-attn: 不在此安装（cu118+源码编译耗时且易冲突），统一用 sdpa ----
echo "[setup] 跳过 flash-attn，注意力实现统一回退 sdpa"

echo "[setup] done. transformers / torch versions:"
python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('transformers',transformers.__version__)"

# ============================================================================
# NOTES (按需在跑对应模型前执行，避免污染 Qwen3-Omni 主线):
#  - MiniCPM-o-4.5  : 自带 trust_remote_code 建模代码，声明 transformers 4.51。
#                     若 4.57 加载报错，临时: pip install vector-quantize-pytorch vocos
#                     一般 4.57 可加载（remote code 内做了兼容）；冲突时记录在报告。
#  - Ming-flash-omni: 参考 /cpfs_speech3/yulian.zpf/Ming/requirements.txt 与
#                     已有 conda env `ming`；若强冲突，按用户既定方案“单独处理”。
# ============================================================================
