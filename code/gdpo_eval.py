"""GDPO R3 评测（omni-context-mcpm env）。三个部分：
A) 探针：120 个 held-out 窗（30 评测流），base / +LoRA 的 CLEAN/COMPLEX 准确率与延迟
B) 端到端：训好的模型当检测器，对 30 评测流逐 2s 窗(步长1s)滑动判定 → 门控时间线 P/R/F1、
   触发延迟；按其判定区间注入(gated_v2 措辞) → 4 参考 cpWER + 干净/复杂段词召回
C) 推理智商：SpeakerCounting / MultiSpeakerDetection（acc，run_bench_eval 任务）
用法: python gdpo_eval.py --part probe|e2e|reason [--lora checkpoints/minicpm_gdpo_gate_lora --tag gdpo]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/cpfs_speech3/yulian.zpf/Omni-Context/code")
import numpy as np
import yaml
import run_bench_eval as R
from stream_gate_eval import perm_cpwer, render_clue, recall, CONTRACT, INSTR, CONSTRAINT
ROOT = R.ROOT
PROBE = ("Listen to this short audio clip. Decide if it is acoustically COMPLEX "
         "(two or more people talking at the same time, or strong background noise) "
         "or CLEAN (one clear speaker, little noise). "
         "Answer with exactly one word: COMPLEX or CLEAN.")


def load_adapter(model_name, lora):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval_config.yaml")))
    adapter = R.get_adapter(model_name, cfg["models"][model_name]); adapter.load()
    if lora:
        from peft import PeftModel
        if model_name == "qwen3_omni":
            adapter.model.thinker = PeftModel.from_pretrained(adapter.model.thinker, lora)
            adapter.model.thinker.eval()
        else:
            adapter.model = PeftModel.from_pretrained(adapter.model, lora)
            adapter.model.eval()
        print(f"[lora] {lora}")
    return adapter


def parse_pred(t):
    t = (t or "").strip().lower()
    if t.startswith("complex"): return "COMPLEX"
    if t.startswith("clean"): return "CLEAN"
    return "?"


def part_probe(adapter, tag):
    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/stream_gate_eval.jsonl") if l.strip()]
    import soundfile as sf
    outp = ROOT + f"/results/gdpo_probe__{tag}.jsonl"
    fout = open(outp, "w", encoding="utf-8")
    n = c = 0; lat = []
    accs = {"CLEAN": [0, 0], "COMPLEX": [0, 0]}
    for r in rows:
        for wlen in (2,):
            for (t0, lab) in ((1.0, "CLEAN"), (r["complex_start"] + 1.0, "COMPLEX")):
                wp = f"{ROOT}/benchmarks/_wav/_stream_probe/{r['id']}__w{wlen}_{lab}.wav"
                if not os.path.exists(wp):
                    continue
                t = time.perf_counter()
                raw = adapter.infer(wp, PROBE, max_new_tokens=4)
                lat.append(time.perf_counter() - t)
                pred = parse_pred(raw)
                ok = pred == lab
                accs[lab][0] += ok; accs[lab][1] += 1
                n += 1; c += ok
                fout.write(json.dumps({"id": r["id"], "label": lab, "pred": pred, "raw": raw[:40]}) + "\n")
    fout.close()
    res = {"tag": tag, "n": n, "acc": c / n,
           "clean_acc": accs["CLEAN"][0] / max(accs["CLEAN"][1], 1),
           "complex_acc": accs["COMPLEX"][0] / max(accs["COMPLEX"][1], 1),
           "lat_mean": float(np.mean(lat))}
    print("PROBE", json.dumps(res)); return res


def part_e2e(adapter, tag):
    import soundfile as sf, librosa
    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/stream_gate_eval.jsonl") if l.strip()]
    from stream_gate import hysteresis
    TP = FP = FN = TN = 0; lats = []
    recs = []
    for r in rows:
        wav, sr = librosa.load(r["wav"], sr=16000, mono=True)
        T = int(np.ceil(len(wav) / 16000))
        raw = []
        for s in range(T):
            seg = wav[max(0, (s - 1)) * 16000: (s + 1) * 16000]  # 2s 窗（含前1s）
            if len(seg) < 8000:
                raw.append(False); continue
            tmp = f"/tmp/gd_{tag}_{r['id']}_{s}.wav"; sf.write(tmp, seg, 16000)
            pred = parse_pred(adapter.infer(tmp, PROBE, max_new_tokens=4))
            os.unlink(tmp)
            raw.append(pred == "COMPLEX")
        gate = hysteresis(raw)
        gt = [r["complex_start"] <= s < r["complex_end"] for s in range(T)]
        for g_, t_ in zip(gate, gt):
            TP += g_ and t_; FP += g_ and not t_; FN += (not g_) and t_; TN += (not g_) and (not t_)
        cs = int(r["complex_start"])
        fire = [s for s, g_ in enumerate(gate) if g_ and s >= cs]
        if fire: lats.append(fire[0] - cs)
        # 注入推理（模型判定区间）
        spans = []
        cur = None
        for s, g_ in enumerate(gate):
            if g_ and cur is None: cur = s
            if not g_ and cur is not None: spans.append((cur, s)); cur = None
        if cur is not None: spans.append((cur, T))
        refs = [r["ref_clean_a"]] + r["ref_spk"] + [r["ref_clean_b"]]
        if spans:
            span = (min(s[0] for s in spans), max(s[1] for s in spans))
            clue = render_clue(r["clue_gt"], span) + ("\nIMPORTANT: still transcribe the ENTIRE audio "
                   "from beginning to end, including the clean speech before and after the complex segment.")
            prompt = "\n".join([CONTRACT, "\n" + clue + "\n", INSTR, CONSTRAINT])
        else:
            prompt = "\n".join([INSTR, CONSTRAINT])
        hyp = adapter.infer(r["wav"], prompt, max_new_tokens=300)
        recs.append({"id": r["id"], "injected": bool(spans),
                     "cpwer": perm_cpwer(refs, hyp),
                     "recall_clean": recall([r["ref_clean_a"], r["ref_clean_b"]], hyp),
                     "recall_complex": recall(r["ref_spk"], hyp)})
    P = TP / (TP + FP + 1e-9); Rc = TP / (TP + FN + 1e-9)
    import statistics as st
    res = {"tag": tag, "precision": round(P, 3), "recall": round(Rc, 3),
           "f1": round(2 * P * Rc / (P + Rc + 1e-9), 3),
           "trigger_latency": round(float(np.mean(lats)), 2) if lats else None,
           "clean_skip_rate": round(TN / (TN + FP + 1e-9), 3),
           "cpwer": round(100 * st.mean(x["cpwer"] for x in recs), 1),
           "recall_clean": round(100 * st.mean(x["recall_clean"] for x in recs if x["recall_clean"] is not None), 1),
           "recall_complex": round(100 * st.mean(x["recall_complex"] for x in recs if x["recall_complex"] is not None), 1)}
    json.dump({"summary": res, "rows": recs}, open(ROOT + f"/results/gdpo_e2e__{tag}.json", "w"))
    print("E2E", json.dumps(res)); return res


def part_reason(adapter, tag):
    out = {}
    for task in ("SpeakerCounting_LibriTTS-TestClean", "MultiSpeakerDetection_LibriSpeech-TestClean"):
        spec = R.TASKS[task]
        man = [json.loads(l) for l in open(ROOT + f"/benchmarks/_manifest/{task}.jsonl")]
        n = c = 0
        for r in man:
            prompt = R.build_prompt(task, r["instruction"], "baseline", None)
            try:
                raw = adapter.infer(r["audio_path"], prompt, max_new_tokens=32)
            except Exception as e:
                print("ERR", r["id"], repr(e)[:60]); continue
            if spec["kind"] == "count":
                ok = R.norm_count(raw) == R.norm_count(r["label"])
            else:
                ok = R.norm_bool(raw) == R.norm_bool(r["label"])
            n += 1; c += ok
        out[task] = {"n": n, "acc": round(c / n, 3)}
        print("REASON", task, out[task])
    json.dump(out, open(ROOT + f"/results/gdpo_reason__{tag}.json", "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["probe", "e2e", "reason", "all"])
    ap.add_argument("--model", default="minicpm_o")
    ap.add_argument("--lora", default="")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    adapter = load_adapter(a.model, a.lora or None)
    if a.part in ("probe", "all"): part_probe(adapter, a.tag)
    if a.part in ("e2e", "all"): part_e2e(adapter, a.tag)
    if a.part in ("reason", "all"): part_reason(adapter, a.tag)
    print("DONE", a.tag)


if __name__ == "__main__":
    main()
