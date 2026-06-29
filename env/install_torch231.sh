#!/usr/bin/env bash
# 升级 torch 到 2.3.1+cu118（满足 transformers 4.57 的 register_pytree_node）。
# nvidia-cu11 依赖从官方 download.pytorch.org（快, ~5MB/s）拉取；其余从 aliyun pypi。
source /cpfs_speech3/yulian.zpf/anaconda3/etc/profile.d/conda.sh
conda activate omni-context
cd /cpfs_speech3/yulian.zpf/Omni-Context/env/wheels

python -m pip install \
  ./torch-2.3.1+cu118-cp311-cp311-linux_x86_64.whl \
  ./torchaudio-2.3.1+cu118-cp311-cp311-linux_x86_64.whl \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/

echo "===VERIFY==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
try:
    import transformers
    print("transformers", transformers.__version__, "OK")
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    print("Qwen3Omni classes OK")
except Exception as e:
    print("TRANSFORMERS_ERR", repr(e)[:160])
PY
echo "===VDONE==="
