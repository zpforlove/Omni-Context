"""分层抽样：从 test split 抽取约 N 条覆盖全部难度切片的评测子集。

分层维度:
  - type: multi_speaker / speech_env
  - multi_speaker -> 按 speaker_overlap_ratio 桶 (0.0/0.15/0.3/0.5)
  - speech_env   -> 按 (snr_db 桶) x (environment_type) 联合分层
保证每个切片都被覆盖，且整体接近均衡。
"""
import argparse, json, math, random
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def stratum_key(row):
    t = row["type"]
    d = row["difficulty"]
    if t == "multi_speaker":
        return ("multi_speaker", f"ovl={d.get('speaker_overlap_ratio')}",
                f"nspk={d.get('num_speakers')}")
    else:
        return ("speech_env", f"snr={d.get('snr_db')}",
                f"env={d.get('environment_type')}")


def stratified_sample(rows, total, seed):
    """两级分层：先按大类(type)均分配额，再在大类内按子层均匀抽样。"""
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    types = sorted(by_type.keys())
    quota_per_type = total // len(types)

    buckets = defaultdict(list)
    for r in rows:
        buckets[stratum_key(r)].append(r)

    selected = []
    for ty in types:
        sub_keys = sorted(k for k in buckets if k[0] == ty)
        per = max(1, quota_per_type // max(1, len(sub_keys)))
        picked = []
        for k in sub_keys:
            pool = list(buckets[k])
            rng.shuffle(pool)
            picked.extend(pool[:min(per, len(pool))])
        # 大类内补足/截断到配额
        if len(picked) < quota_per_type:
            ids = {r["sample_id"] for r in picked}
            rest = [r for r in by_type[ty] if r["sample_id"] not in ids]
            rng.shuffle(rest)
            picked.extend(rest[:quota_per_type - len(picked)])
        else:
            rng.shuffle(picked)
            picked = picked[:quota_per_type]
        selected.extend(picked)

    selected.sort(key=lambda r: r["sample_id"])
    return selected, buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="test.jsonl 路径")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows = [r for r in load_jsonl(args.manifest) if r.get("split") == args.split]
    sel, buckets = stratified_sample(rows, args.size, args.seed)

    with open(args.out, "w") as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印分层统计
    from collections import Counter
    by_type = Counter(r["type"] for r in sel)
    print(f"[sampling] total pool={len(rows)}  strata={len(buckets)}  selected={len(sel)}")
    print(f"[sampling] type dist: {dict(by_type)}")
    se = [r for r in sel if r["type"] == "speech_env"]
    ms = [r for r in sel if r["type"] == "multi_speaker"]
    print(f"[sampling] speech_env snr dist:",
          dict(Counter(r["difficulty"]["snr_db"] for r in se)))
    print(f"[sampling] speech_env env dist:",
          dict(Counter(r["difficulty"]["environment_type"] for r in se)))
    print(f"[sampling] multi_speaker overlap dist:",
          dict(Counter(r["difficulty"]["speaker_overlap_ratio"] for r in ms)))
    print(f"[sampling] written -> {args.out}")


if __name__ == "__main__":
    main()
