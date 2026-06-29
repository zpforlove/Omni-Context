#!/usr/bin/env bash
# AGSC 全量编排（3 条件 NoCtx_real/AGSC_real/AGSC_silent，全量子集）：
#   阶段1 Qwen3  : 3 条件 -> GPU0/1/2 (omni-context)
#   阶段2 MiniCPM: 3 条件 -> GPU0/1/2 (omni-context-mcpm)
#   阶段3 Ming   : 3 条件串行, 4 卡 (ming)
#   阶段4 报告
CB=/cpfs_speech3/yulian.zpf/anaconda3/etc/profile.d/conda.sh
ROOT=/cpfs_speech3/yulian.zpf/Omni-Context
LOG=$ROOT/logs
CONDS="NoCtx_real AGSC_real AGSC_silent"

echo "===== $(date) 阶段1 Qwen3 ====="
i=0
for c in $CONDS; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c \
    "source $CB; conda activate omni-context; cd $ROOT/code; python run_eval_agsc.py --model qwen3_omni --cond $c" \
    > $LOG/agsc_qwen3_$c.log 2>&1 &
  i=$((i+1))
done
wait
echo "===== $(date) 阶段2 MiniCPM ====="
i=0
for c in $CONDS; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c \
    "source $CB; conda activate omni-context-mcpm; cd $ROOT/code; python run_eval_agsc.py --model minicpm_o --cond $c" \
    > $LOG/agsc_minicpm_$c.log 2>&1 &
  i=$((i+1))
done
wait
echo "===== $(date) 阶段3 Ming ====="
for c in $CONDS; do
  bash -c "source $CB; conda activate ming; cd $ROOT/code; python run_eval_agsc.py --model ming --cond $c" \
    > $LOG/agsc_ming_$c.log 2>&1
done
echo "===== $(date) 阶段4 报告 ====="
bash -c "source $CB; conda activate omni-context; cd $ROOT/code; python build_report_agsc.py" > $LOG/agsc_report.log 2>&1
echo "ALL_AGSC_DONE $(date)"
