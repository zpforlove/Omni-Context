import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Part D：一 token 实时门控探针——Omni 模型自身当注入检测器。
从 30 条门控流切窗：干净区(秒1-2/1-3) 与 复杂区(秒5-6/5-7)，窗长 1s/2s。
提示模型输出单词 COMPLEX / CLEAN，测准确率 + 单窗墙钟延迟。
这是"模型自带门控 token"的零样本可行性下界(GRPO 内化的起点)。
"""
import json, os, sys, argparse, time
sys.path.insert(0, os.path.join(OMNI_ROOT, "code"))
import yaml, soundfile as sf
import run_bench_eval as R
ROOT = R.ROOT
SR = 16000

PROMPT = ("Listen to this short audio clip. Decide if it is acoustically COMPLEX "
          "(two or more people talking at the same time, or strong background noise) "
          "or CLEAN (one clear speaker, little noise). "
          "Answer with exactly one word: COMPLEX or CLEAN.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/stream_gate_eval.jsonl", encoding="utf-8") if l.strip()]
    outdir = ROOT + "/benchmarks/_wav/_stream_probe"; os.makedirs(outdir, exist_ok=True)
    wins = []
    for r in rows:
        wav, sr = sf.read(r["wav"])
        for wlen in (1, 2):
            for (t0, lab) in ((1.0, "CLEAN"), (r["complex_start"] + 1.0, "COMPLEX")):
                seg = wav[int(t0 * sr): int((t0 + wlen) * sr)]
                wp = f"{outdir}/{r['id']}__w{wlen}_{lab}.wav"
                if not os.path.exists(wp):
                    sf.write(wp, seg, sr)
                wins.append({"wav": wp, "wlen": wlen, "label": lab, "id": r["id"]})

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval_config.yaml")))
    adapter = R.get_adapter(a.model, cfg["models"][a.model]); adapter.load()
    outp = ROOT + f"/results/stream_probe__{a.model}.jsonl"
    done = set()
    if os.path.exists(outp):
        done = {(json.loads(l)["wav"]) for l in open(outp, encoding="utf-8") if l.strip()}
    fout = open(outp, "a", encoding="utf-8")
    # warmup
    try:
        adapter.infer(wins[0]["wav"], PROMPT, max_new_tokens=4)
    except Exception:
        pass
    n = 0
    for w in wins:
        if w["wav"] in done:
            continue
        try:
            t0 = time.perf_counter()
            raw = adapter.infer(w["wav"], PROMPT, max_new_tokens=4)
            dt = time.perf_counter() - t0
        except Exception as e:
            print("ERR", w["wav"], repr(e)[:80]); continue
        pred = "COMPLEX" if "complex" in raw.lower() else ("CLEAN" if "clean" in raw.lower() else "?")
        fout.write(json.dumps({"wav": w["wav"], "id": w["id"], "wlen": w["wlen"],
                               "label": w["label"], "pred": pred, "raw": raw[:50],
                               "correct": pred == w["label"], "latency_s": round(dt, 3)},
                              ensure_ascii=False) + "\n"); fout.flush()
        n += 1
        if n % 20 == 0:
            print(f"[{a.model}] {n}/{len(wins)}", flush=True)
    fout.close()
    print("DONE", a.model)


if __name__ == "__main__":
    main()
