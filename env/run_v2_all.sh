#!/usr/bin/env bash
# V2 全量评测编排（600 子集，四条件 A0/A2/B2/B4）：
#   阶段1 Qwen3  : 4 条件 -> GPU0/1/2/3 并行 (omni-context)
#   阶段2 MiniCPM: 4 条件 -> GPU0/1/2/3 并行 (omni-context-mcpm)
#   阶段3 Ming   : 4 条件串行，每条件用全部 4 卡 device_map (ming)
#   阶段4 报告   : build_report_v2
CB=/cpfs_speech3/yulian.zpf/anaconda3/etc/profile.d/conda.sh
ROOT=/cpfs_speech3/yulian.zpf/Omni-Context
LOG=$ROOT/logs
CONDS="A0 A2 B2 B4"

echo "===== $(date) 阶段1 Qwen3 (4卡并行) ====="
i=0
for c in $CONDS; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c \
    "source $CB; conda activate omni-context; cd $ROOT/code; python run_eval_v2.py --model qwen3_omni --cond $c" \
    > $LOG/v2_qwen3_$c.log 2>&1 &
  i=$((i+1))
done
wait
echo "===== $(date) 阶段2 MiniCPM (4卡并行) ====="
i=0
for c in $CONDS; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c \
    "source $CB; conda activate omni-context-mcpm; cd $ROOT/code; python run_eval_v2.py --model minicpm_o --cond $c" \
    > $LOG/v2_minicpm_$c.log 2>&1 &
  i=$((i+1))
done
wait
echo "===== $(date) 阶段3 Ming (4卡, 条件串行) ====="
for c in $CONDS; do
  bash -c "source $CB; conda activate ming; cd $ROOT/code; python run_eval_v2.py --model ming --cond $c" \
    > $LOG/v2_ming_$c.log 2>&1
done
echo "===== $(date) 阶段4 报告 ====="
bash -c "source $CB; conda activate omni-context; cd $ROOT/code; python build_report_v2.py" > $LOG/v2_report.log 2>&1
echo "ALL_V2_DONE $(date)"
