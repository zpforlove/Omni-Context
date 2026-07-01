import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""门控塌缩 SFT 矫正。
RL 救不动饱和门控(logit差~20→softmax梯度≈0，GDPO组内advantage常抵消)；
SFT 交叉熵 -log p(CLEAN) 在 p→0 时梯度极大，直接破饱和。
只对 GATE 行 teacher-forcing(clean→'GATE: CLEAN' / mix→'GATE: COMPLEX')，纯矫正门控不动转写。
热启动自 csb_lora，存 _gatesft 新路径不覆盖原 lora。
"""
import argparse, json, os, sys, random
sys.path.insert(0, os.path.join(OMNI_ROOT, "code"))
import torch
import run_bench_eval as R
from gdpo_chain_train import POLICIES, build_prompt, CK_NAME
ROOT = R.ROOT


def load_resume(pol, model, path):
    import safetensors.torch as st_
    sd = st_.load_file(os.path.join(path, "adapter_model.safetensors"))
    sd = {(k.replace("lora_A.weight", "lora_A.default.weight")
            .replace("lora_B.weight", "lora_B.default.weight")): v for k, v in sd.items()}
    tgt = pol.thinker if model == "qwen3_omni" else (pol.peft if hasattr(pol, "peft") else None)
    if tgt is not None:
        missing, unexpected = tgt.load_state_dict(sd, strict=False)
        print(f"[resume] loaded {len(sd)-len(unexpected)}/{len(sd)} lora tensors from {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(POLICIES))
    ap.add_argument("--resume_lora", required=True)
    ap.add_argument("--clean_ratio", type=float, default=0.5)
    ap.add_argument("--eps_cap", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--ck_suffix", default="_gatesft")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/csb_train.jsonl") if l.strip()]
    eps = []
    rng = random.Random(20260613)
    for r in rows:
        if r["kind"] == "mix":
            mat = rng.choice(["none", "t2", "t4", "full"])
            clue = None if mat == "none" else (r.get("context", {}).get(f"clue_{mat}") or None)
            span = (int(r["complex_start"]), int(r["complex_end"])) if clue else None
            refs = [r["ref_clean_a"]] + r["ref_spk"] + [r["ref_clean_b"]]
            eps.append({"wav": r["wav"], "truth": "COMPLEX", "clue": clue, "span": span, "refs": refs})
        elif r["kind"] == "clean":
            rc = r["ref_clean"]
            refs = rc if isinstance(rc, list) else [rc]
            eps.append({"wav": r["wav"], "truth": "CLEAN", "clue": None, "span": None, "refs": refs})
    rng.shuffle(eps)
    if a.eps_cap:
        eps = eps[: a.eps_cap]
    if 0 < a.clean_ratio < 1:
        mix_e = [e for e in eps if e["truth"] == "COMPLEX"]
        clean_e = [e for e in eps if e["truth"] == "CLEAN"]
        if clean_e and mix_e:
            tc = int(round(len(mix_e) * a.clean_ratio / (1 - a.clean_ratio)))
            if tc > len(clean_e):
                clean_e = (clean_e * (-(-tc // len(clean_e))))[:tc]
            eps = mix_e + clean_e
            rng.shuffle(eps)
            print(f"[upsample] mix={len(mix_e)} clean={len(clean_e)} total={len(eps)}", flush=True)
    if a.smoke:
        eps = eps[: a.smoke]
    print(f"[gate-sft/{a.model}] {len(eps)} episodes lr={a.lr} accum={a.accum}", flush=True)

    pol = POLICIES[a.model]()
    load_resume(pol, a.model, a.resume_lora)
    opt = torch.optim.AdamW(pol.params, lr=a.lr)
    logf = open(ROOT + f"/results/gatesft_{a.model}.log.jsonl", "a", encoding="utf-8")
    opt.zero_grad(); step = 0; acc = 0; lsum = 0.0
    for e in eps:
        prompt = build_prompt(e["clue"], e["span"])
        body = "\n".join(str(x) for x in e["refs"] if x)
        target = f"GATE: {e['truth']}\n" + body
        try:
            loss = -pol.mean_logprob(e["wav"], prompt, target) / a.accum
            loss.backward(); lsum += float(loss.detach()) * a.accum; acc += 1
        except Exception as ex:
            print("[skip]", repr(ex)[:100], flush=True); continue
        if acc >= a.accum:
            g = torch.nn.utils.clip_grad_norm_(pol.params, 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            rec = {"step": step, "loss": lsum / a.accum, "gnorm": float(g)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if step % 5 == 0:
                print(f"[gate-sft] step{step} loss={lsum/a.accum:.4f} gnorm={float(g):.2f}", flush=True)
            acc = 0; lsum = 0.0
    pol.save(ROOT + f"/checkpoints/{CK_NAME[a.model]}_csb_lora{a.ck_suffix}")
    print(f"[gate-sft] saved {CK_NAME[a.model]}_csb_lora{a.ck_suffix}", flush=True)
    print("DONE gate-sft", a.model)


if __name__ == "__main__":
    main()
