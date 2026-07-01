import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Stage C：Qwen3-Omni LoRA SFT —— DeepSpeed ZeRO-2 数据并行（4 卡各持全模型，并行不同 batch，~4x 加速）。
启动：deepspeed --num_gpus 4 train_stage_c_ds.py --epochs 1 [--smoke 6]
数据：datasets/stage_c_train_v2.jsonl（门控 AGSC + gold + 扩充高质量合成，零泄漏）。
"""
import argparse
import json
import os
import sys
import torch
import deepspeed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_stage_c import build_inputs, _enable_input_grads  # 复用输入构造与梯度 hook

ROOT = OMNI_ROOT
MODEL = "/cpfs_speech3/yulian.zpf/Qwen3-Omni-30B-A3B-Instruct"
DATA = ROOT + "/datasets/stage_c_train_v2.jsonl"
ADAPTER_OUT = ROOT + "/checkpoints/qwen3_agsc_lora"


def ds_config(accum, lr):
    # 不在 config 里指定 optimizer → 改为传入 torch.optim.AdamW 实例，
    # 避免 DeepSpeed 默认 FusedAdam(CUDA 算子按系统 CUDA12.8 编译，驱动470 跑不了)。
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": accum,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True,
                              "reduce_bucket_size": 5e7},
        "gradient_clipping": 1.0,
        "zero_force_ds_cpu_optimizer": False,
        "steps_per_print": 1000000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=60)
    ap.add_argument("--local_rank", type=int, default=0)
    args = ap.parse_args()

    deepspeed.init_distributed()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    is_main = rank == 0

    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    attn = "sdpa"
    try:
        import flash_attn  # noqa
        attn = "flash_attention_2"
    except Exception:
        pass
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    from qwen_omni_utils import process_mm_info
    if is_main:
        print(f"[ds] world={world} 加载模型到各卡 ...", flush=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation=attn, low_cpu_mem_usage=True)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    for attr in ["talker", "code2wav", "token2wav"]:  # 训练只需 thinker，删冗余省显存
        if getattr(model, attr, None) is not None:
            setattr(model, attr, None)
    thinker = model.thinker.to(f"cuda:{local}")
    del model

    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    thinker = get_peft_model(thinker, cfg)
    if is_main:
        thinker.print_trainable_parameters()
    thinker.config.use_cache = False
    thinker.gradient_checkpointing_enable()
    _enable_input_grads(thinker)
    thinker.train()

    params = [p for p in thinker.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)  # torch 优化器，避免 DeepSpeed FusedAdam CUDA 算子
    engine, _, _, _ = deepspeed.initialize(model=thinker, optimizer=opt,
                                           config=ds_config(args.accum, args.lr))
    dev = engine.device

    data = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    import random
    random.Random(20260607).shuffle(data)
    if args.smoke:
        data = data[:args.smoke * world]
    shard = data[rank::world]  # 数据并行：每卡不同子集
    if is_main:
        print(f"[ds] total={len(data)} per-rank={len(shard)} accum={args.accum}", flush=True)

    micro = 0
    for epoch in range(args.epochs):
        for ex in shard:
            try:
                inputs = build_inputs(ex, proc, process_mm_info)
                inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
                if inputs.get("input_features") is not None:
                    inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
                loss = engine(**inputs).loss
                engine.backward(loss)
                engine.step()
            except Exception as e:
                if is_main:
                    print(f"  [skip {ex['id']}] {repr(e)[:140]}", flush=True)
                continue
            micro += 1
            if is_main and micro % args.accum == 0:
                print(f"  epoch{epoch} optstep{micro//args.accum} (micro{micro}) loss={loss.item():.4f}", flush=True)
            if args.save_every and micro % (args.save_every * args.accum) == 0:
                if is_main:
                    os.makedirs(os.path.dirname(ADAPTER_OUT), exist_ok=True)
                    thinker.save_pretrained(ADAPTER_OUT)
                    print(f"  [ckpt] optstep{micro//args.accum} -> {ADAPTER_OUT}", flush=True)

    if is_main and not args.smoke:
        os.makedirs(os.path.dirname(ADAPTER_OUT), exist_ok=True)
        thinker.save_pretrained(ADAPTER_OUT)
        print(f"[ds] saved LoRA -> {ADAPTER_OUT}", flush=True)
    if is_main:
        print("[ds] DONE" if not args.smoke else "[smoke] OK", flush=True)


if __name__ == "__main__":
    main()
