#!/usr/bin/env bash
ROOT="${OMNI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
# 全量 600 条 Qwen3-Omni 评测：E0/E1/E2 三条件分到 GPU 0/1/2 并行跑（各自加载一份模型）。
# 断点续跑：已完成的 sample_id 自动跳过（含冒烟的前 6 条）。
source /cpfs_speech3/yulian.zpf/anaconda3/etc/profile.d/conda.sh
conda activate omni-context
cd $ROOT/code
LOG=$ROOT/logs

CUDA_VISIBLE_DEVICES=0 nohup python run_eval.py --model qwen3_omni --conditions E0 > $LOG/full_qwen3_E0.log 2>&1 &
echo "E0 on GPU0 pid $!"
CUDA_VISIBLE_DEVICES=1 nohup python run_eval.py --model qwen3_omni --conditions E1 > $LOG/full_qwen3_E1.log 2>&1 &
echo "E1 on GPU1 pid $!"
CUDA_VISIBLE_DEVICES=2 nohup python run_eval.py --model qwen3_omni --conditions E2 > $LOG/full_qwen3_E2.log 2>&1 &
echo "E2 on GPU2 pid $!"
wait
echo "ALL_QWEN3_CONDITIONS_DONE"
