"""GDPO/GRPO 门控强化学习训练器 v2（MiniCPM-o 8B + LoRA，单卡，omni-context-mcpm env）。

实现 arXiv:2601.05242 GDPO，针对二元门控动作空间的高效精确版：
  - 动作空间 {COMPLEX, CLEAN}，策略 π_θ(a) = softmax(seq_logprob(a))（两次教师强制前向，精确）
  - 行为策略 μ = (1-ε)π + ε·uniform（ε-探索保证组内多样性——chat 采样接口失效 & 模型过自信会
    导致组内零方差、优势恒 0，这是 GRPO/GDPO 共同的退化点）
  - G 个动作从 μ 采样；PPO 比率 s = π_θ(a)/μ(a)，clip(1±0.2)（Eq.3 形式，μ 为 detach 常数）
  - GDPO: 每奖励组内 z-score(Eq.4) → 加权和(Eq.7) → batch 级归一化(Eq.6)
    GRPO 对照臂: 奖励先加权求和 → 组内 z-score
  - 双奖励（r_down 以 r_gate 部分条件化思想保留在奖励定义内）:
      r_gate: 判定与窗口真值一致 → 1
      r_down: 复杂窗判 COMPLEX → tanh(max(Δ_s,0)/0.2)（实测注入收益查表）
              干净窗判 COMPLEX → −0.5（实测注入伤干净段）；判 CLEAN → 0
"""
import argparse, json, os, random, sys
sys.path.insert(0, "/cpfs_speech3/yulian.zpf/Omni-Context/code")
import numpy as np
import torch
import librosa
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
PROMPT = ("Listen to this short audio clip. Decide if it is acoustically COMPLEX "
          "(two or more people talking at the same time, or strong background noise) "
          "or CLEAN (one clear speaker, little noise). "
          "Answer with exactly one word: COMPLEX or CLEAN.")
ACTIONS = ["COMPLEX", "CLEAN"]


def load_model():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tsm", ROOT + "/code/train_stage_n_minicpm.py")
    tsm = importlib.util.module_from_spec(spec)
    sys.modules["tsm"] = tsm
    spec.loader.exec_module(tsm)
    model, proc, tok = tsm.load()

    def build_one_aligned(model_, proc_, audio, prompt, target=None):
        """与部署 chat 路径同模板（use_tts_template=False）。"""
        import numpy as _np
        teacher = target is not None
        msgs = [{"role": "user", "content": [prompt, audio]}]
        if teacher:
            msgs.append({"role": "assistant", "content": [target]})
        audios, audio_parts, copy = [], [], []
        for i, m in enumerate(msgs):
            cur = []
            for c in m["content"]:
                if isinstance(c, _np.ndarray):
                    audios.append(c); audio_parts.append(i); cur.append("<audio>./</audio>")
                else:
                    cur.append(str(c))
            copy.append({"role": m["role"], "content": "".join(cur)})
        text = proc_.tokenizer.apply_chat_template(copy, tokenize=False,
                                                   add_generation_prompt=not teacher,
                                                   use_tts_template=False, enable_thinking=False)
        return proc_([text], [[]], [audios], [audio_parts], return_tensors="pt").to(model_.device)

    return model, proc, tok, build_one_aligned


def rewards_for(action, gt_label, delta_s):
    r_gate = 1.0 if action == gt_label else 0.0
    if action == "COMPLEX":
        r_down = float(np.tanh(max(delta_s, 0.0) / 0.2)) if gt_label == "COMPLEX" else -0.5
    else:
        r_down = 0.0
    return np.array([r_gate, r_down], dtype=np.float64)


def znorm(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["gdpo", "grpo"], default="gdpo")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eps", type=float, default=0.25)   # 探索率
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--batch_windows", type=int, default=8)
    ap.add_argument("--weights", default="1.0,1.0")      # w_gate,w_down
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    W = np.array([float(x) for x in a.weights.split(",")])

    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/gdpo_train.jsonl") if l.strip()]
    rtab = json.load(open(ROOT + "/results/gdpo_reward_table.json"))
    wins = []
    for r in rows:
        d = rtab.get(r["id"], {}).get("delta", 0.0)
        wins.append({"wav": r["win_complex"], "label": "COMPLEX", "delta": d})
        wins.append({"wav": r["win_clean"], "label": "CLEAN", "delta": d})
    rng = random.Random(20260610); rng.shuffle(wins)
    np_rng = np.random.RandomState(20260610)
    if a.smoke:
        wins = wins[: a.smoke]
    print(f"[{a.algo}] {len(wins)} windows G={a.G} eps={a.eps} W={W.tolist()}", flush=True)

    model, proc, tok, build_one = load_model()
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=r".*llm\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    base = model.base_model.model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)

    def action_logprobs(audio):
        """两个动作的序列 logprob（带梯度，sum 而非 mean → 真序列概率）。"""
        lps = []
        for act in ACTIONS:
            full = build_one(model, proc, audio, PROMPT, act)
            prm = build_one(model, proc, audio, PROMPT, None)
            plen = prm["input_ids"].shape[1]
            L = full["input_ids"].shape[1]
            full["position_ids"] = torch.arange(L, device=full["input_ids"].device).unsqueeze(0)
            labels = full["input_ids"].clone().long()
            labels[:, :plen] = -100
            out = base(full, labels=labels, attention_mask=full.get("attention_mask"))
            ntok = int((labels != -100).sum())
            lps.append(-(out.loss if hasattr(out, "loss") else out[0]) * ntok)  # mean CE → sum logprob
        return torch.stack(lps)  # [2]

    logf = open(ROOT + f"/results/gdpo_train_{a.algo}.log.jsonl", "a", encoding="utf-8")
    gstep = 0
    model.train()
    for epoch in range(a.epochs):
        rng.shuffle(wins)
        for bi in range(0, len(wins), a.batch_windows):
            batch = wins[bi: bi + a.batch_windows]
            opt.zero_grad()
            entries = []   # (lp_tensor[2], mu[2], acts[G], A[G]) 先收集再 batch 归一化
            stats = {"acc": 0.0, "piC": [], "n": 0}
            for w in batch:
                audio, _ = librosa.load(w["wav"], sr=16000, mono=True)
                lp = action_logprobs(audio)                     # 带梯度
                with torch.no_grad():
                    pi = torch.softmax(lp.float(), dim=0).cpu().numpy()
                mu = (1 - a.eps) * pi + a.eps * 0.5
                acts = np_rng.choice(2, size=a.G, p=mu / mu.sum())
                Rm = np.stack([rewards_for(ACTIONS[k], w["label"], w["delta"]) for k in acts])  # [G,2]
                if a.algo == "gdpo":
                    A = sum(W[j] * znorm(Rm[:, j]) for j in range(Rm.shape[1]))   # Eq.4+7
                else:
                    A = znorm((Rm * W).sum(axis=1))                                # GRPO
                entries.append((lp, mu, acts, A))
                gt_idx = ACTIONS.index(w["label"])
                stats["acc"] += float((acts == gt_idx).mean()); stats["piC"].append(float(pi[0])); stats["n"] += 1
            # Eq.6 batch 级归一化（GDPO）
            allA = np.concatenate([e[3] for e in entries])
            if a.algo == "gdpo":
                m, s = allA.mean(), allA.std() + 1e-6
            loss_total = torch.zeros((), device="cuda")
            n_terms = sum(len(e[2]) for e in entries)
            for lp, mu, acts, A in entries:
                logpi = torch.log_softmax(lp.float(), dim=0)
                for k, adv in zip(acts, A):
                    if a.algo == "gdpo":
                        adv = (adv - m) / s
                    if abs(adv) < 1e-9:
                        continue
                    ratio = torch.exp(logpi[k] - float(np.log(mu[k])))
                    un = ratio * float(adv)
                    cl = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * float(adv)
                    loss_total = loss_total - torch.min(un, cl) / n_terms
            loss_total.backward()
            g = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            gstep += 1
            rec = {"step": gstep, "epoch": epoch, "loss": float(loss_total.detach()),
                   "samp_acc": stats["acc"] / stats["n"], "pi_complex_mean": float(np.mean(stats["piC"])),
                   "gnorm": float(g)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if gstep % 5 == 0:
                print(f"[{a.algo}] step{gstep} loss={rec['loss']:.4f} sampacc={rec['samp_acc']:.2f} "
                      f"piC={rec['pi_complex_mean']:.3f} gnorm={rec['gnorm']:.2f}", flush=True)
        outdir = ROOT + f"/checkpoints/minicpm_{a.algo}_gate_lora"
        model.save_pretrained(outdir)
        print(f"[{a.algo}] epoch{epoch} saved -> {outdir}", flush=True)
    print(f"DONE {a.algo}")


if __name__ == "__main__":
    main()
