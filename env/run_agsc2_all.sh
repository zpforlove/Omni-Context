#!/usr/bin/env bash
# AGSC-v2 并行编排（3 条件 NoCtx_real/AGSC_real/AGSC_silent，404 硬样本）：
#   阶段1 Qwen3   : 3 条件 -> GPU0/1/2 (omni-context)
#   阶段2 MiniCPM : 3 条件 -> GPU0/1/2 (omni-context-mcpm)
#   阶段3 Ming    : 3 条件串行, 4 卡 (ming)
#   阶段4 报告
CB=/cpfs_speech3/yulian.zpf/anaconda3/etc/profile.d/conda.sh
ROOT="${OMNI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
LOG=$ROOT/logs
CONDS="NoCtx_real AGSC_real AGSC_silent"
E="PYTHONUTF8=1 LC_ALL=C.UTF-8"

echo "===== $(date) 阶段1 Qwen3 ====="
i=0
for c in $CONDS; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c "source $CB; conda activate omni-context; cd $ROOT/code; $E python run_eval_agsc2.py --model qwen3_omni --cond $c" > $LOG/agsc2_qwen3_$c.log 2>&1 &
  i=$((i+1))
done
wait
echo "===== $(date) 阶段2 MiniCPM ====="
i=0
for c in $CONDS; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c "source $CB; conda activate omni-context-mcpm; cd $ROOT/code; $E python run_eval_agsc2.py --model minicpm_o --cond $c" > $LOG/agsc2_minicpm_$c.log 2>&1 &
  i=$((i+1))
done
wait
echo "===== $(date) 阶段3 Ming ====="
for c in $CONDS; do
  bash -c "source $CB; conda activate ming; cd $ROOT/code; $E python run_eval_agsc2.py --model ming --cond $c" > $LOG/agsc2_ming_$c.log 2>&1
done
echo "===== $(date) 阶段4 报告 ====="
bash -c "source $CB; conda activate omni-context; cd $ROOT/code; $E python build_report_agsc2.py" > $LOG/agsc2_report.log 2>&1
echo "ALL_AGSC2_DONE $(date)"
