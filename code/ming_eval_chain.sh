#!/bin/bash
# S4->S5 自动接链：等 Ming 训练结束 -> base 评测 -> csb 评测（串行，104B 占满4卡）
cd /cpfs_speech3/yulian.zpf/Omni-Context/code
P=/cpfs_speech3/anaconda3/envs/ming/bin/python
LOGDIR=/cpfs_speech3/yulian.zpf/Omni-Context/logs
ulimit -n 65535
while pgrep -f "gdpo_chain_train.py --model ming" > /dev/null; do sleep 180; done
sleep 60
if [ ! -d /cpfs_speech3/yulian.zpf/Omni-Context/checkpoints/ming_csb_lora ]; then
  echo "TRAIN FAILED: no ming_csb_lora checkpoint, abort eval $(date)" >> $LOGDIR/ming_eval_chain.log
  exit 1
fi
echo "train done $(date), start base eval" >> $LOGDIR/ming_eval_chain.log
$P csb_eval_run.py --model ming --tag csb_ming_base > $LOGDIR/ce_csb_ming_base.log 2>&1
echo "base eval done $(date), start csb eval" >> $LOGDIR/ming_eval_chain.log
$P csb_eval_run.py --model ming --tag csb_ming_csb --lora /cpfs_speech3/yulian.zpf/Omni-Context/checkpoints/ming_csb_lora > $LOGDIR/ce_csb_ming_csb.log 2>&1
echo "all done $(date)" >> $LOGDIR/ming_eval_chain.log
