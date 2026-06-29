"""Stage C：MiniCPM-o-4.5 LoRA SFT（单卡，env=omni-context-mcpm）。
复刻 chat 的输入构造（teacher_forcing），forward(data, labels=) 透传到底层 LLM 出 loss。
LoRA 加在 model.llm 的 q/k/v/o_proj。8B 单卡可容，不用梯度检查点。
用法：python train_stage_c_minicpm.py --smoke 4 | --epochs 1
"""
import argparse
import json
import os
import sys
import numpy as np
import librosa
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
MODEL = "/cpfs_speech3/yulian.zpf/MiniCPM-o-4_5"
DATA = ROOT + "/datasets/stage_n_train.jsonl"
ADAPTER_OUT = ROOT + "/checkpoints/minicpm_noise_lora"


def load():
    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, attn_implementation="sdpa",
                                      torch_dtype=torch.bfloat16, init_vision=True, init_audio=True, init_tts=False)
    model = model.cuda()
    model.prepare_processor(processor=proc, tokenizer=tok)
    return model, proc, tok


def build_one(model, proc, audio, prompt, target=None):
    """复刻 chat 的输入构造；target=None 时只到 generation prompt(算 prompt 长度)。"""
    teacher = target is not None
    msgs = [{"role": "user", "content": [prompt, audio]}]
    if teacher:
        msgs.append({"role": "assistant", "content": [target]})
    audios, audio_parts = [], []
    copy = []
    for i, m in enumerate(msgs):
        cur = []
        for c in m["content"]:
            if isinstance(c, np.ndarray):
                audios.append(c); audio_parts.append(i); cur.append("<audio>./</audio>")
            else:
                cur.append(str(c))
        copy.append({"role": m["role"], "content": "".join(cur)})
    text = proc.tokenizer.apply_chat_template(copy, tokenize=False,
                                              add_generation_prompt=not teacher,
                                              use_tts_template=True, enable_thinking=False)
    inputs = proc([text], [[]], [audios], [audio_parts], return_tensors="pt").to(model.device)
    return inputs


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
    data = [d for d in data if not d.get("two_audio")]  # MiniCPM 单音频任务(AMI 双音频跳过)
    if args.smoke:
        data = data[:args.smoke]
    print(f"[mcpm] {len(data)} examples")

    model, proc, tok = load()
    from peft import LoraConfig, get_peft_model
    # 正则只命中 llm 的注意力投影：原地注入 LoRA，不改 MiniCPM 内部模块路径(避免 embed_tokens 访问失败)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=r".*llm\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.train()
    base = model.base_model.model  # 内层 MiniCPMO(LoRA 已原地注入)，绕开 PeftCausalLM 的 input_ids 签名
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    step = 0; opt.zero_grad()
    for epoch in range(args.epochs):
        for i, ex in enumerate(data):
            try:
                audio, _ = librosa.load(ex["audio_path"], sr=16000, mono=True)
                full = build_one(model, proc, audio, ex["prompt"], ex["target"])
                prm = build_one(model, proc, audio, ex["prompt"], None)
                plen = prm["input_ids"].shape[1]
                labels = full["input_ids"].clone().long()
                labels[:, :plen] = -100
                L = full["input_ids"].shape[1]
                full["position_ids"] = torch.arange(L, device=full["input_ids"].device).unsqueeze(0)
                out = base(full, labels=labels, attention_mask=full.get("attention_mask"))
                loss = out.loss if hasattr(out, "loss") else out[0]
                (loss / args.accum).backward()
            except Exception as e:
                print(f"  [skip {ex['id']}] {repr(e)[:160]}"); continue
            if (i + 1) % args.accum == 0:
                g = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                print(f"  epoch{epoch} step{step} loss={loss.item():.4f} gnorm={float(g):.3f}", flush=True)
                if args.save_every and step % args.save_every == 0:
                    os.makedirs(ADAPTER_OUT, exist_ok=True); model.save_pretrained(ADAPTER_OUT)
                    print(f"  [ckpt] step{step} -> {ADAPTER_OUT}")
            if args.smoke and i + 1 >= args.smoke:
                print(f"  [smoke] last loss={loss.item():.4f}"); break
    if not args.smoke:
        os.makedirs(ADAPTER_OUT, exist_ok=True); model.save_pretrained(ADAPTER_OUT)
        print(f"[mcpm] saved LoRA -> {ADAPTER_OUT}")
    print("[mcpm] DONE" if not args.smoke else "[smoke] OK")


if __name__ == "__main__":
    main()
