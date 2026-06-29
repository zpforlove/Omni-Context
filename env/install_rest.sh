#!/usr/bin/env bash
# 等本地 wheel 下好后，离线安装 torch(2.1.2+cu118) + 其余依赖。
set -e
source /cpfs_speech3/yulian.zpf/anaconda3/etc/profile.d/conda.sh
conda activate omni-context
cd /cpfs_speech3/yulian.zpf/Omni-Context/env/wheels
PIP="python -m pip"

echo "[rest] installing torch/torchaudio from local wheels"
$PIP install ./torch-2.1.2+cu118-cp311-cp311-linux_x86_64.whl \
            ./torchaudio-2.1.2+cu118-cp311-cp311-linux_x86_64.whl

echo "[rest] core libs"
$PIP install "numpy<2.0"
$PIP install "transformers==4.57.1" "accelerate>=1.0.0" "tokenizers>=0.21"
$PIP install "qwen-omni-utils" "qwen-vl-utils"
$PIP install librosa soundfile scipy
$PIP install jiwer rapidfuzz rouge-score sacrebleu jieba regex
$PIP install pyyaml tqdm pandas tabulate

# 清除可能误装的 cu12 nvidia 运行库（与 cu118 自带库冲突）
$PIP uninstall -y nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 \
  nvidia-cudnn-cu12 nvidia-cublas-cu12 2>/dev/null || true

echo "[rest] verify:"
python -c "import torch,transformers;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'tf',transformers.__version__)"
python -c "from transformers import Qwen3OmniMoeForConditionalGeneration,Qwen3OmniMoeProcessor;print('Qwen3Omni classes OK')"
echo "[rest] DONE"
