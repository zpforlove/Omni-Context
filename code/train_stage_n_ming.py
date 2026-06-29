"""Stage C：Ming-flash-omni-2.0 (BailingMM2) LoRA SFT —— 4 卡 naive-MP（222G），env=ming。
复用 Ming 适配器的 load（device_map/_split_model + eager + DynamicCache 垫片）。
输入：复刻 adapter.infer 的 processor 构造（+assistant target）；
forward：extract_audio_feature → self.model.model(query_embeds_audio=..., placeholder_audio_loc_lens=..., labels=) 出 loss。
LoRA：moe 注意力 query_key_value/dense。
用法：python train_stage_c_ming.py --smoke 4 | --epochs 1
"""
import argparse
import json
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa
from run_eval import get_adapter

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
DATA = ROOT + "/datasets/stage_n_train.jsonl"
ADAPTER_OUT = ROOT + "/checkpoints/ming_noise_lora"


def build_train(adapter, audio_path, prompt, target):
    """Ming 模板只接受 HUMAN 轮：用 HUMAN 构造 prompt(含音频展开)，再手动拼 target token+eos，labels 掩码 prompt。"""
    proc, tok = adapter.processor, adapter.tokenizer
    messages = [{"role": "HUMAN", "content": [{"type": "text", "text": prompt}, {"type": "audio", "audio": audio_path}]}]
    text = proc.apply_chat_template(messages, sys_prompt_exp=None, use_cot_system_prompt=False)
    img, vid, aud = proc.process_vision_info(messages)
    inp = proc(text=[text], images=img, videos=vid, audios=aud,
               audio_kwargs={"use_whisper_encoder": True}, return_tensors="pt")
    plen = inp["input_ids"].shape[1]
    tgt = tok(target, add_special_tokens=False, return_tensors="pt")["input_ids"]
    eos = proc.gen_terminator
    eos = eos[0] if isinstance(eos, (list, tuple)) else int(eos)
    eos_t = torch.tensor([[eos]], dtype=tgt.dtype)
    full_ids = torch.cat([inp["input_ids"], tgt, eos_t], dim=1)
    tail = tgt.shape[1] + 1
    inp["input_ids"] = full_ids
    if inp.get("attention_mask") is not None:
        inp["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones(1, tail, dtype=inp["attention_mask"].dtype)], dim=1)
    labels = full_ids.clone().long()
    labels[:, :plen] = -100
    return inp, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=120)
    args = ap.parse_args()

    data = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    import random
    random.Random(20260607).shuffle(data)
    data = [d for d in data if not d.get("two_audio")]  # 单音频任务
    if args.smoke:
        data = data[:args.smoke]
    print(f"[ming] {len(data)} examples", flush=True)

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval_config.yaml")))
    adapter = get_adapter("ming", cfg["models"]["ming"])
    adapter.load()
    model = adapter.model  # BailingMM2Native (device_map across GPUs)

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=None,
                      target_modules=r".*layers\.\d+\.attention\.(query_key_value|dense)")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    for c in (getattr(adapter.model, "config", None), getattr(adapter.model.model, "config", None)):
        if c is not None and hasattr(c, "use_cache"):
            c.use_cache = False
    model.train()
    dev0 = torch.device("cuda:0")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    step = 0; opt.zero_grad()
    for epoch in range(args.epochs):
        for i, ex in enumerate(data):
            try:
                full, labels = build_train(adapter, ex["audio_path"], ex["prompt"], ex["target"])
                full = full.to(dev0); labels = labels.to(dev0)
                for k in list(full.keys()):
                    if k in ("audio_feats", "pixel_values"):
                        full[k] = full[k].to(torch.bfloat16)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ae, ael = adapter.model.extract_audio_feature(full["audio_feats"], full["audio_feats_lengths"], use_whisper_encoder=True)
                    out = adapter.model.model(  # labels=None：取 logits 自己算 loss(避开 Ming 的 fused CE inplace_backward)
                        input_ids=full["input_ids"], attention_mask=full.get("attention_mask"),
                        query_embeds_audio=ae, query_embeds_audio_lengths=ael,
                        placeholder_audio_loc_lens=full["audio_placeholder_loc_lens"],
                        use_cache=False)
                logits = (out.logits if hasattr(out, "logits") else out[0]).float()
                lab = labels.to(logits.device)
                sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                st = lab[:, 1:].reshape(-1)
                loss = torch.nn.functional.cross_entropy(sl, st, ignore_index=-100)
                (loss / args.accum).backward()
            except Exception as e:
                print(f"  [skip {ex['id']}] {repr(e)[:120]}", flush=True); continue
            if (i + 1) % args.accum == 0:
                g = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                print(f"  epoch{epoch} step{step} loss={loss.item():.4f} gnorm={float(g):.3f}", flush=True)
                if args.save_every and step % args.save_every == 0:
                    os.makedirs(ADAPTER_OUT, exist_ok=True); model.save_pretrained(ADAPTER_OUT)
                    print(f"  [ckpt] step{step} -> {ADAPTER_OUT}", flush=True)
            if args.smoke and i + 1 >= args.smoke:
                print(f"  [smoke] last loss={loss.item():.4f}"); break
    if not args.smoke:
        os.makedirs(ADAPTER_OUT, exist_ok=True); model.save_pretrained(ADAPTER_OUT)
        print(f"[ming] saved LoRA -> {ADAPTER_OUT}")
    print("[ming] DONE" if not args.smoke else "[smoke] OK")


if __name__ == "__main__":
    main()
