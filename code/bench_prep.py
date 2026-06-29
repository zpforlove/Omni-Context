"""Stage B-1：benchmark 预处理。
把 HF benchmark（Dynamic-SUPERB 说话人子集）抽样、导出音频文件、产统一 manifest。
manifest 行：{id, task, audio_path, instruction, label}
用法：python bench_prep.py --task SpeakerCounting_LibriTTS-TestClean --n 80
"""
import argparse
import json
import os

BENCH = "/cpfs_speech3/yulian.zpf/Omni-Context/benchmarks"
WAV = BENCH + "/_wav"
MAN = BENCH + "/_manifest"


def prep(task, n, seed=20260606):
    from datasets import load_dataset, Audio
    ds = load_dataset(BENCH + "/" + task)["test"].cast_column("audio", Audio(decode=False))
    idx = list(range(len(ds)))
    if n and n < len(ds):
        import random
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:n])
    outdir = os.path.join(WAV, task)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(MAN, exist_ok=True)
    rows = []
    for i in idx:
        r = ds[i]
        a = r["audio"]
        ext = os.path.splitext(a.get("path") or "x.wav")[1] or ".wav"
        ap = os.path.join(outdir, f"{i:06d}{ext}")
        if not os.path.exists(ap):
            with open(ap, "wb") as f:
                f.write(a["bytes"])
        rows.append({"id": f"{task}_{i:06d}", "task": task, "audio_path": ap,
                     "instruction": r["instruction"], "label": str(r["label"]).strip()})
    mpath = os.path.join(MAN, task + ".jsonl")
    with open(mpath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prep] {task}: {len(rows)} samples -> {mpath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    prep(a.task, a.n)
