import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Curve B（线索成熟度）：模型听【完整音频】(基线稳定、低方差)，线索由前缀[0:t]计算。
gain(t)=cpWER_base − cpWER_agsc(clue@t)。隔离"线索需要多少音频才有用"，去除截断音频的未听尾部稀释。
"""
import json, os, sys, argparse, time
sys.path.insert(0, os.path.join(OMNI_ROOT, "code"))
import yaml
import run_bench_eval as R
ROOT = R.ROOT
task = "SparseLibriMix2_noisy"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    clue_rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/stream_clues.jsonl", encoding="utf-8") if l.strip()]
    man = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT + f"/benchmarks/_manifest/{task}.jsonl", encoding="utf-8")}
    ids = sorted({r["id"] for r in clue_rows})

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval_config.yaml")))
    adapter = R.get_adapter(a.model, cfg["models"][a.model]); adapter.load()
    mnt = 200
    outp = ROOT + f"/results/stream_maturity__{a.model}.jsonl"
    done = set()
    if os.path.exists(outp):
        for l in open(outp, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); done.add((d["id"], d["tag"]))
    fout = open(outp, "a", encoding="utf-8")

    # baseline once per id (tag=base)
    for sid in ids:
        if (sid, "base") in done: continue
        m = man[sid]; full_wav = m["audio_path"]; ref = m["label"]
        pr = R.build_prompt(task, m["instruction"], "baseline", None)
        try:
            raw = adapter.infer(full_wav, pr, max_new_tokens=mnt)
        except Exception as e:
            print("ERR base", sid, repr(e)[:80]); continue
        fout.write(json.dumps({"id": sid, "tag": "base", "cond": "baseline", "cpwer": R.cpwer(ref, raw)}, ensure_ascii=False) + "\n"); fout.flush()
    # agsc with clue@prefix
    for r in clue_rows:
        sid = r["id"]; tag = r["tag"]
        if (sid, tag) in done: continue
        m = man[sid]; full_wav = m["audio_path"]; ref = m["label"]
        pr = R.build_prompt(task, m["instruction"], "agsc", {"spk_keywords": r.get("spk_keywords", {})})
        try:
            raw = adapter.infer(full_wav, pr, max_new_tokens=mnt)
        except Exception as e:
            print("ERR", sid, tag, repr(e)[:80]); continue
        fout.write(json.dumps({"id": sid, "tag": tag, "t": r["t"], "cond": "agsc",
                               "n_kw": sum(len(v) for v in r.get("spk_keywords", {}).values()),
                               "cpwer": R.cpwer(ref, raw)}, ensure_ascii=False) + "\n"); fout.flush()
    fout.close()
    print("DONE", a.model, "->", outp)


if __name__ == "__main__":
    main()
