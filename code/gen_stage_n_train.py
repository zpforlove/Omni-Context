import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Stage N-B：合成统一 Stage-N 训练集 = S1(噪声) + S2(重叠+噪声) + 旧重叠数据(防遗忘)。
输出 datasets/stage_n_train.jsonl（门控 AGSC + gold + 零泄漏）。
"""
import json
import os
import random
import sys
ROOT = OMNI_ROOT
sys.path.insert(0, ROOT + "/code")
from run_bench_eval import build_prompt, TASKS  # 复用评测一致的 prompt 构造

rng = random.Random(20260608)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def main():
    rows = []
    # S1：直接用 s1_train.jsonl（已 stage_c 格式）
    s1 = load(ROOT + "/datasets/s1_train.jsonl")
    rows += s1

    # S2：SparseLibriMix2_noisy manifest + agsc → agsc-cond + baseline-cond
    task = "SparseLibriMix2_noisy"
    man = load(ROOT + "/benchmarks/_manifest/" + task + ".jsonl")
    ag = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/" + task + ".jsonl", encoding="utf-8")}
    n2 = 0
    for r in man:
        a = ag.get(r["id"])
        use = bool(a) and len(a.get("overlap_regions", [])) >= 1
        variants = [("agsc", a)] if use else []
        variants.append(("baseline", None))
        for cond, aa in variants:
            rows.append({"id": f"{r['id']}__{cond}", "source": "s2_noisy",
                         "audio_path": r["audio_path"], "two_audio": False,
                         "prompt": build_prompt(task, r["instruction"], cond, aa), "target": r["label"]})
            n2 += 1

    # 旧重叠数据（防遗忘）：从 stage_c_train_v2 抽样
    old = load(ROOT + "/datasets/stage_c_train_v2.jsonl")
    old = [d for d in old if not d.get("two_audio")]
    rng.shuffle(old)
    keep_old = old[:800]
    rows += keep_old

    rng.shuffle(rows)
    out = ROOT + "/datasets/stage_n_train.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[stage_n] total {len(rows)} -> {out}")
    print("  来源:", dict(Counter(r["source"] for r in rows)))


if __name__ == "__main__":
    main()
