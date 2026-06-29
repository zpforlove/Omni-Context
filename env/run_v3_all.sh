#!/usr/bin/env bash
# V3 AC-Gated 全量编排（4 泄漏任务 × cap150，4 条件 Raw/Gated × 真实/静音）：
#   阶段1 Qwen3  : 4 条件 -> GPU0/1/2/3 并行 (omni-context)
#   阶段2 MiniCPM: 4 条件 -> GPU0/1/2/3 并行 (omni-context-mcpm)
#   阶段3 Ming   : 4 条件串行，4 卡 device_map (ming)
#   阶段4 报告   : build_report_v3
CB=/cpfs_speech3/yulian.zpf/anaconda3/etc/profile.d/conda.sh
ROOT=/cpfs_speech3/yulian.zpf/Omni-Context
LOG=$ROOT/logs
CONDS="Raw_real Raw_silent Gated_real Gated_silent"
CAP=150

echo "===== $(date) 阶段1 Qwen3 (4卡并行) ====="
i=0
for c in $CONDS; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c \
    "source $CB; conda activate omni-context; cd $ROOT/code; python run_eval_v3.py --model qwen3_omni --cond $c --cap $CAP" \
    > $LOG/v3_qwen3_$c.log 2>&1 &
  i=$((i+1))
done
wait
echo "===== $(date) 阶段2 MiniCPM (4卡并行) ====="
i=0
for c in $CONDS; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c \
    "source $CB; conda activate omni-context-mcpm; cd $ROOT/code; python run_eval_v3.py --model minicpm_o --cond $c --cap $CAP" \
    > $LOG/v3_minicpm_$c.log 2>&1 &
  i=$((i+1))
done
wait
echo "===== $(date) 阶段3 Ming (4卡, 条件串行) ====="
for c in $CONDS; do
  bash -c "source $CB; conda activate ming; cd $ROOT/code; python run_eval_v3.py --model ming --cond $c --cap $CAP" \
    > $LOG/v3_ming_$c.log 2>&1
done
echo "===== $(date) 阶段4 报告 ====="
bash -c "source $CB; conda activate omni-context; cd $ROOT/code; python build_report_v3.py" > $LOG/v3_report.log 2>&1
echo "ALL_V3_DONE $(date)"
