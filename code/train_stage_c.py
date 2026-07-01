import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Stage C：Qwen3-Omni LoRA SFT —— 让模型学会用门控 AGSC 时间窗在重叠/会议中归属说话人。
数据：datasets/stage_c_train.jsonl（门控 AGSC + gold 目标，零泄漏）。
标签：仅监督 assistant(target) 部分，prompt 与音频占位 mask 为 -100。
LoRA：thinker 注意力 q/k/v/o_proj（轻量稳定，MoE 友好）。bf16 + 梯度检查点 + device_map=auto(多卡)。
用法：
  python train_stage_c.py --smoke 3                 # 冒烟：3 步，验证 loss/backward/显存
  python train_stage_c.py --epochs 1 --max_steps 0  # 正式训练
"""
import argparse
import json
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = OMNI_ROOT
MODEL = "/cpfs_speech3/yulian.zpf/Qwen3-Omni-30B-A3B-Instruct"
DATA = ROOT + "/datasets/stage_c_train.jsonl"
ADAPTER_OUT = ROOT + "/checkpoints/qwen3_agsc_lora"


def load_model():
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    attn = "sdpa"
    try:
        import flash_attn  # noqa
        attn = "flash_attention_2"
    except Exception:
        pass
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation=attn)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    from qwen_omni_utils import process_mm_info
    return model, proc, process_mm_info


def add_lora(model):
    """对 thinker（真正的 LM）应用 LoRA，并装回 model，使训练 forward 与 generate 都走带 LoRA 的 thinker。"""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model.thinker = get_peft_model(model.thinker, cfg)
    model.thinker.print_trainable_parameters()
    return model


def _enable_input_grads(model):
    """梯度检查点需输入嵌入输出 requires_grad。顶层 Qwen3Omni 未实现 get_input_embeddings，手动挂 hook 到 thinker 文本嵌入。"""
    try:
        model.enable_input_require_grads()
        print("[grad] enable_input_require_grads OK")
        return
    except Exception:
        pass
    emb = None
    for name, mod in model.named_modules():
        if name.endswith("thinker.model.embed_tokens") and isinstance(mod, torch.nn.Embedding):
            emb = mod; break
    if emb is None:  # 退化：任意第一个文本 embed_tokens
        for name, mod in model.named_modules():
            if name.endswith("embed_tokens") and isinstance(mod, torch.nn.Embedding):
                emb = mod; print(f"[grad] fallback embed: {name}"); break
    if emb is None:
        raise RuntimeError("找不到 embed_tokens 挂 grad hook")
    emb.register_forward_hook(lambda m, inp, out: out.requires_grad_(True))
    print("[grad] hooked input embedding for checkpointing")


def build_inputs(ex, proc, process_mm_info):
    from common.prompts import SYS_PROMPT
    paths = [ex["audio_path"]] + ([ex["audio2_path"]] if ex.get("two_audio") else [])
    user = [{"type": "audio", "audio": p} for p in paths] + [{"type": "text", "text": ex["prompt"]}]
    conv_p = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]},
              {"role": "user", "content": user}]
    conv_f = conv_p + [{"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]}]
    text_f = proc.apply_chat_template(conv_f, add_generation_prompt=False, tokenize=False)
    text_p = proc.apply_chat_template(conv_p, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv_f, use_audio_in_video=False)
    full = proc(text=text_f, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False)
    prm = proc(text=text_p, audio=audios, images=images, videos=videos,
               return_tensors="pt", padding=True, use_audio_in_video=False)
    plen = prm["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    full["labels"] = labels
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=300)
    args = ap.parse_args()

    data = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    import random
    random.Random(20260606).shuffle(data)
    if args.smoke:
        data = data[:args.smoke]
    print(f"[train] {len(data)} examples")

    model, proc, pmm = load_model()
    model = add_lora(model)
    thinker = model.thinker  # 带 LoRA 的 LM，训练对象
    if hasattr(thinker, "config"):
        thinker.config.use_cache = False
    thinker.gradient_checkpointing_enable()
    _enable_input_grads(thinker)
    thinker.train()

    params = [p for p in thinker.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    dev = model.device if hasattr(model, "device") else torch.device("cuda:0")

    step = 0
    opt.zero_grad()
    for epoch in range(args.epochs):
        for i, ex in enumerate(data):
            try:
                inputs = build_inputs(ex, proc, pmm)
                inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
                if "input_features" in inputs and inputs["input_features"] is not None:
                    inputs["input_features"] = inputs["input_features"].to(model.dtype)
                out = thinker(**inputs)  # 直接 forward thinker(LM)，返回 loss
                loss = out.loss
                if loss is None:
                    raise RuntimeError("loss is None")
                (loss / args.accum).backward()
            except Exception as e:
                print(f"  [skip {ex['id']}] {repr(e)[:160]}")
                continue
            if (i + 1) % args.accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                print(f"  epoch{epoch} step{step} (ex{i+1}) loss={loss.item():.4f} gradnorm={float(gnorm):.4f}", flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
                if args.save_every and step % args.save_every == 0:
                    thinker.save_pretrained(ADAPTER_OUT); print(f"  [ckpt] -> {ADAPTER_OUT}")
            if args.smoke and i + 1 >= args.smoke:
                print(f"  [smoke] last loss={loss.item():.4f}")
                break
        if args.max_steps and step >= args.max_steps:
            break

    if not args.smoke:
        os.makedirs(os.path.dirname(ADAPTER_OUT), exist_ok=True)
        thinker.save_pretrained(ADAPTER_OUT)
        print(f"[train] saved LoRA -> {ADAPTER_OUT}")
    else:
        print("[smoke] OK (forward+backward 正常)")


if __name__ == "__main__":
    main()
