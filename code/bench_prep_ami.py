import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Stage B Wave2-1：AMI TargetSpeaker-ASR 预处理（双音频）。
audio=混音(audio1)，audio2=目标说话人 enrollment，label=目标说话人转写。
manifest 行：{id, task, audio_path, audio2_path, instruction, label}
用法：python bench_prep_ami.py --n 100
"""
import argparse
import json
import os

BENCH = os.path.join(OMNI_ROOT, "benchmarks")
TASK = "TargetSpeaker-ASR_AMItest"


def prep(n, seed=20260606):
    from datasets import load_dataset, Audio
    ds = load_dataset(BENCH + "/" + TASK)["test"]
    for col in ["audio", "audio2"]:
        ds = ds.cast_column(col, Audio(decode=False))
    idx = list(range(len(ds)))
    if n and n < len(ds):
        import random
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:n])
    wdir = os.path.join(BENCH, "_wav", TASK)
    os.makedirs(wdir, exist_ok=True)
    os.makedirs(os.path.join(BENCH, "_manifest"), exist_ok=True)
    rows = []
    for i in idx:
        r = ds[i]
        paths = {}
        for col, tag in [("audio", "mix"), ("audio2", "enroll")]:
            a = r[col]
            ext = os.path.splitext(a.get("path") or "x.wav")[1] or ".wav"
            ap = os.path.join(wdir, f"{i:06d}_{tag}{ext}")
            if not os.path.exists(ap):
                with open(ap, "wb") as f:
                    f.write(a["bytes"])
            paths[tag] = ap
        rows.append({"id": f"{TASK}_{i:06d}", "task": TASK, "audio_path": paths["mix"],
                     "audio2_path": paths["enroll"], "instruction": r["instruction"],
                     "label": str(r["label"]).strip()})
    mp = os.path.join(BENCH, "_manifest", TASK + ".jsonl")
    with open(mp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prep] {TASK}: {len(rows)} -> {mp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    prep(a.n)
