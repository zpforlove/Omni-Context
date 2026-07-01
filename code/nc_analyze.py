import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Stage N-C 分析：S1/S2 训练前 baseline vs +AGSC，按 SNR/难度切片（验证 H1/H2）。"""
import json
import os
import sys
import statistics as st

ROOT = OMNI_ROOT
RAW = ROOT + "/results/bench_raw"
MANI = ROOT + "/benchmarks/_manifest"


def load(f):
    p = os.path.join(RAW, f)
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()} if os.path.exists(p) else {}


def manifest_snr(task):
    p = os.path.join(MANI, task + ".jsonl")
    return {json.loads(l)["id"]: json.loads(l).get("snr_db") for l in open(p, encoding="utf-8") if l.strip()}


def cap(x):
    return min(x, 1.0)


def report(model, task, snr_buckets=True):
    b, a = load(f"{model}__{task}__baseline.jsonl"), load(f"{model}__{task}__agsc.jsonl")
    ids = sorted(set(b) & set(a))
    if not ids:
        print(f"{model}/{task}: incomplete (b={len(b)} a={len(a)})"); return
    snr = manifest_snr(task)
    def m(d, g): return st.mean(cap(d[i]["score"]) for i in g) * 100

    print(f"\n### {model} | {task}  N={len(ids)}  (metric↓越低越好)")
    print(f"  全部:  base={m(b,ids):.1f}  agsc={m(a,ids):.1f}  Δ={m(b,ids)-m(a,ids):+.1f}")
    if snr_buckets:
        for lo, hi, name in [(-1e9, 0.5, "低SNR(≤0)"), (0.5, 7.5, "中SNR(5)"), (7.5, 1e9, "高SNR(≥10)")]:
            g = [i for i in ids if snr.get(i) is not None and lo <= snr[i] < hi]
            if g:
                print(f"  {name:10s} n={len(g):3d}  base={m(b,g):.1f}  agsc={m(a,g):.1f}  Δ={m(b,g)-m(a,g):+.1f}")
    # 难 1/3（按 baseline）
    srt = sorted(ids, key=lambda i: cap(b[i]["score"])); hard = srt[2 * len(srt) // 3:]
    print(f"  难1/3:  base={m(b,hard):.1f}  agsc={m(a,hard):.1f}  Δ={m(b,hard)-m(a,hard):+.1f}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3_omni"
    report(model, "speech_env_S1")
    report(model, "SparseLibriMix2_noisy")
