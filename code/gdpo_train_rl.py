import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""GDPO 门控 RL 统一训练器：--model qwen3_omni | ming（MiniCPM 用 gdpo_train_minicpm.py）。

同 v2 方法：二元动作解析策略 π=softmax(seq_logprob) + ε-探索行为策略 + PPO clip 比率（Eq.3）+
GDPO 解耦归一化（Eq.4/6/7）。每窗仅 2 次教师强制前向。
  qwen3_omni: train_stage_c.build_inputs + thinker(**inputs) → loss；LoRA 在 thinker attn（env omni-context）
  ming      : train_stage_n_ming.build_train + extract_audio_feature + 手动 CE；LoRA 在 attention.qkv/dense
              （env ming，4 卡 device_map）
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.join(OMNI_ROOT, "code"))
import numpy as np
import torch
ROOT = OMNI_ROOT
PROMPT = ("Listen to this short audio clip. Decide if it is acoustically COMPLEX "
          "(two or more people talking at the same time, or strong background noise) "
          "or CLEAN (one clear speaker, little noise). "
          "Answer with exactly one word: COMPLEX or CLEAN.")
ACTIONS = ["COMPLEX", "CLEAN"]


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


class Qwen3Policy:
    def __init__(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("tsc", ROOT + "/code/train_stage_c.py")
        tsc = importlib.util.module_from_spec(spec); sys.modules["tsc"] = tsc
        spec.loader.exec_module(tsc)
        self.tsc = tsc
        model, proc, pmm = tsc.load_model()
        model = tsc.add_lora(model)
        self.model, self.proc, self.pmm = model, proc, pmm
        self.thinker = model.thinker
        if hasattr(self.thinker, "config"):
            self.thinker.config.use_cache = False
        self.thinker.train()
        self.dev = model.device if hasattr(model, "device") else torch.device("cuda:0")
        self.params = [p for p in self.thinker.parameters() if p.requires_grad]

    def seq_logprob(self, wav, action):
        ex = {"audio_path": wav, "prompt": PROMPT, "target": action}
        inputs = self.tsc.build_inputs(ex, self.proc, self.pmm)
        ntok = int((inputs["labels"] != -100).sum())
        inputs = {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if inputs.get("input_features") is not None:
            inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
        out = self.thinker(**inputs)
        return -out.loss * ntok

    def save(self, outdir):
        self.thinker.save_pretrained(outdir)


class MingPolicy:
    def __init__(self):
        import yaml
        from run_eval import get_adapter
        import importlib.util
        spec = importlib.util.spec_from_file_location("tnm", ROOT + "/code/train_stage_n_ming.py")
        tnm = importlib.util.module_from_spec(spec); sys.modules["tnm"] = tnm
        spec.loader.exec_module(tnm)
        self.tnm = tnm
        cfg = yaml.safe_load(open(ROOT + "/configs/eval_config.yaml"))
        adapter = get_adapter("ming", cfg["models"]["ming"]); adapter.load()
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type=None,
                          target_modules=r".*layers\.\d+\.attention\.(query_key_value|dense)")
        self.peft = get_peft_model(adapter.model, lcfg)
        self.peft.print_trainable_parameters()
        for c in (getattr(adapter.model, "config", None), getattr(adapter.model.model, "config", None)):
            if c is not None and hasattr(c, "use_cache"):
                c.use_cache = False
        self.peft.train()
        self.adapter = adapter
        self.dev = torch.device("cuda:0")
        self.params = [p for p in self.peft.parameters() if p.requires_grad]

    def seq_logprob(self, wav, action):
        full, labels = self.tnm.build_train(self.adapter, wav, PROMPT, action)
        full = full.to(self.dev); labels = labels.to(self.dev)
        for k in list(full.keys()):
            if k in ("audio_feats", "pixel_values"):
                full[k] = full[k].to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ae, ael = self.adapter.model.extract_audio_feature(
                full["audio_feats"], full["audio_feats_lengths"], use_whisper_encoder=True)
            out = self.adapter.model.model(
                input_ids=full["input_ids"], attention_mask=full.get("attention_mask"),
                query_embeds_audio=ae, query_embeds_audio_lengths=ael,
                placeholder_audio_loc_lens=full["audio_placeholder_loc_lens"], use_cache=False)
        logits = (out.logits if hasattr(out, "logits") else out[0]).float()
        lab = labels.to(logits.device)
        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        st = lab[:, 1:].reshape(-1)
        ce = torch.nn.functional.cross_entropy(sl, st, ignore_index=-100, reduction="sum")
        return -ce

    def save(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        self.peft.save_pretrained(outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3_omni", "ming"])
    ap.add_argument("--algo", choices=["gdpo", "grpo"], default="gdpo")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--batch_windows", type=int, default=8)
    ap.add_argument("--weights", default="1.0,1.0")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    W = np.array([float(x) for x in a.weights.split(",")])

    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/gdpo_train.jsonl") if l.strip()]
    rtab = json.load(open(ROOT + f"/results/gdpo_reward_table_{a.model}.json"))
    wins = []
    for r in rows:
        d = rtab.get(r["id"], {}).get("delta", 0.0)
        wins.append({"wav": r["win_complex"], "label": "COMPLEX", "delta": d})
        wins.append({"wav": r["win_clean"], "label": "CLEAN", "delta": d})
    rng = random.Random(20260610); rng.shuffle(wins)
    np_rng = np.random.RandomState(20260610)
    if a.smoke:
        wins = wins[: a.smoke]
    print(f"[{a.model}/{a.algo}] {len(wins)} windows G={a.G} eps={a.eps}", flush=True)

    pol = Qwen3Policy() if a.model == "qwen3_omni" else MingPolicy()
    opt = torch.optim.AdamW(pol.params, lr=a.lr)
    logf = open(ROOT + f"/results/gdpo_train_{a.model}_{a.algo}.log.jsonl", "a", encoding="utf-8")
    gstep = 0
    for epoch in range(a.epochs):
        rng.shuffle(wins)
        for bi in range(0, len(wins), a.batch_windows):
            batch = wins[bi: bi + a.batch_windows]
            opt.zero_grad()
            entries, stats = [], {"acc": 0.0, "piC": [], "n": 0}
            ok = True
            for w in batch:
                try:
                    lp = torch.stack([pol.seq_logprob(w["wav"], act) for act in ACTIONS])
                except Exception as e:
                    print("  [skip]", repr(e)[:120], flush=True); continue
                with torch.no_grad():
                    pi = torch.softmax(lp.float(), dim=0).cpu().numpy()
                mu = (1 - a.eps) * pi + a.eps * 0.5
                acts = np_rng.choice(2, size=a.G, p=mu / mu.sum())
                Rm = np.stack([rewards_for(ACTIONS[k], w["label"], w["delta"]) for k in acts])
                if a.algo == "gdpo":
                    A = sum(W[j] * znorm(Rm[:, j]) for j in range(Rm.shape[1]))
                else:
                    A = znorm((Rm * W).sum(axis=1))
                entries.append((lp, mu, acts, A))
                stats["acc"] += float((acts == ACTIONS.index(w["label"])).mean())
                stats["piC"].append(float(pi[0])); stats["n"] += 1
            if not entries:
                continue
            allA = np.concatenate([e[3] for e in entries])
            if a.algo == "gdpo":
                m, s = allA.mean(), allA.std() + 1e-6
            n_terms = sum(len(e[2]) for e in entries)
            loss_total = torch.zeros((), device="cuda:0")
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
                    loss_total = loss_total - torch.min(un, cl).to(loss_total.device) / n_terms
            loss_total.backward()
            g = torch.nn.utils.clip_grad_norm_(pol.params, 1.0)
            opt.step()
            gstep += 1
            rec = {"step": gstep, "epoch": epoch, "loss": float(loss_total.detach()),
                   "samp_acc": stats["acc"] / max(stats["n"], 1),
                   "pi_complex_mean": float(np.mean(stats["piC"])) if stats["piC"] else None,
                   "gnorm": float(g)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if gstep % 5 == 0:
                print(f"[{a.model}/{a.algo}] step{gstep} loss={rec['loss']:.4f} "
                      f"sampacc={rec['samp_acc']:.2f} piC={rec['pi_complex_mean']:.3f} gnorm={rec['gnorm']:.2f}", flush=True)
        outdir = ROOT + f"/checkpoints/{('qwen3' if a.model=='qwen3_omni' else 'ming')}_{a.algo}_gate_lora"
        pol.save(outdir)
        print(f"[{a.model}/{a.algo}] epoch{epoch} saved -> {outdir}", flush=True)
    print(f"DONE {a.model} {a.algo}")


if __name__ == "__main__":
    main()
