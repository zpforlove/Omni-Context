import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Part A 推理：对每个 (id, 前缀t) 在三条件下做 S2 双说话人 ASR，cpWER↓ 对完整 label 评分。
  baseline      : 截断音频 + 无线索
  agsc_stream   : 截断音频 + 【仅前缀算出的】流式线索
  agsc_offline  : 截断音频 + 【完整音频算出的】oracle 线索(现有全量 _agsc)
复用 run_bench_eval 的 build_prompt / cpwer。
"""
import json, os, sys, argparse, time
sys.path.insert(0, os.path.join(OMNI_ROOT, "code"))
import yaml
import run_bench_eval as R
ROOT = R.ROOT
task = "SparseLibriMix2_noisy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conds", default="baseline,agsc_stream,agsc_offline")
    a = ap.parse_args()
    conds = a.conds.split(",")

    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/stream_clues.jsonl", encoding="utf-8") if l.strip()]
    full = {json.loads(l)["id"]: json.loads(l).get("spk_keywords", {})
            for l in open(ROOT + f"/benchmarks/_agsc/{task}.jsonl", encoding="utf-8") if l.strip()}

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval_config.yaml")))
    adapter = R.get_adapter(a.model, cfg["models"][a.model])
    adapter.load()
    mnt = 200

    outp = ROOT + f"/results/stream_eval__{a.model}.jsonl"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    done = set()
    if os.path.exists(outp):
        for l in open(outp, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); done.add((d["id"], d["tag"], d["cond"]))
    fout = open(outp, "a", encoding="utf-8")

    def prompt_for(cond, r):
        if cond == "baseline":
            return R.build_prompt(task, r["instruction"], "baseline", None)
        if cond == "agsc_stream":
            return R.build_prompt(task, r["instruction"], "agsc", {"spk_keywords": r.get("spk_keywords", {})})
        if cond == "agsc_offline":
            return R.build_prompt(task, r["instruction"], "agsc", {"spk_keywords": full.get(r["id"], {})})

    n = 0
    for r in rows:
        ref = r["label"]
        for cond in conds:
            if (r["id"], r["tag"], cond) in done:
                continue
            pr = prompt_for(cond, r)
            try:
                t0 = time.perf_counter()
                raw = adapter.infer(r["wav"], pr, max_new_tokens=mnt)
                dt = time.perf_counter() - t0
            except Exception as e:
                print("ERR", r["id"], r["tag"], cond, repr(e)[:100]); continue
            sc = R.cpwer(ref, raw)
            rec = {"id": r["id"], "t": r["t"], "tag": r["tag"], "dur": r["dur"],
                   "cond": cond, "cpwer": sc, "infer_s": round(dt, 2)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            n += 1
            if n % 20 == 0:
                print(f"[{a.model}] {n} done", flush=True)
    fout.close()
    print(f"DONE {a.model} +{n} -> {outp}")


if __name__ == "__main__":
    main()
