#!/usr/bin/env bash
ROOT="${OMNI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
# 一键评测：在 omni-context 环境内按模型顺序跑评测并生成报告。
# 用法: bash run_all.sh [model1 model2 ...]   默认: qwen3_omni
set -e
source /cpfs_speech3/yulian.zpf/anaconda3/etc/profile.d/conda.sh
conda activate omni-context
cd $ROOT/code

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(qwen3_omni)
LOGDIR=$ROOT/logs

for m in "${MODELS[@]}"; do
  echo "==================== EVAL $m ===================="
  python run_eval.py --model "$m" 2>&1 | tee "$LOGDIR/eval_${m}.log"
done

echo "==================== BUILD REPORT ===================="
python build_report.py 2>&1 | tee "$LOGDIR/build_report.log"
echo "done. see reports/RESULTS_TABLES.md"
