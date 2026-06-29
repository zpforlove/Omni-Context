"""Stage N-B：S2（重叠+噪声）数据集构建 —— 给 SparseLibriMix2 注入 WHAM 噪声(多 SNR)，保留每说话人真值。
读 SparseLibriMix2 manifest(已有 wav + 2行label) → 叠 WHAM 噪声@{0,5,10}dB → 写新音频 + 新 manifest speech-task "SparseLibriMix2_noisy"。
用法：python noise_inject_s2.py --n 150
"""
import argparse
import io
import json
import os
import random

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
BENCH = ROOT + "/benchmarks"
WHAM = BENCH + "/_wham/data/train-00000-of-00010-ce580a0a440284e9.parquet"
rng = random.Random(20260608)
SNRS = [0, 5, 10]


def load_wham_noises(k=300):
    import pyarrow.parquet as pq
    import soundfile as sf
    import numpy as np
    t = pq.read_table(WHAM)
    col = None
    for name in t.column_names:
        if "audio" in name.lower():
            col = name; break
    arr = t.column(col).to_pylist()
    noises = []
    for a in arr[:k]:
        b = a["bytes"] if isinstance(a, dict) else a
        try:
            w, sr = sf.read(io.BytesIO(b))
            if w.ndim > 1:
                w = w.mean(1)
            noises.append((w.astype("float32"), sr))
        except Exception:
            pass
    return noises


def mix(speech, sr, noise, nsr, snr_db):
    import numpy as np
    import librosa
    if nsr != sr:
        noise = librosa.resample(noise, orig_sr=nsr, target_sr=sr)
    if len(noise) < len(speech):
        reps = int(np.ceil(len(speech) / max(1, len(noise))))
        noise = np.tile(noise, reps)
    st = rng.randint(0, max(0, len(noise) - len(speech)))
    noise = noise[st:st + len(speech)]
    ps = np.mean(speech ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    scale = (ps / (pn * (10 ** (snr_db / 10)))) ** 0.5
    out = speech + scale * noise
    m = np.max(np.abs(out)) + 1e-9
    if m > 1.0:
        out = out / m * 0.98
    return out.astype("float32")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150); a = ap.parse_args()
    import soundfile as sf
    man = [json.loads(l) for l in open(BENCH + "/_manifest/SparseLibriMix2.jsonl", encoding="utf-8")][:a.n]
    noises = load_wham_noises()
    print(f"[s2] {len(noises)} WHAM 噪声; {len(man)} 重叠样本")
    wdir = BENCH + "/_wav/SparseLibriMix2_noisy"; os.makedirs(wdir, exist_ok=True)
    os.makedirs(BENCH + "/_manifest", exist_ok=True)
    out = open(BENCH + "/_manifest/SparseLibriMix2_noisy.jsonl", "w", encoding="utf-8")
    n = 0
    for r in man:
        sp, sr = sf.read(r["audio_path"])
        if sp.ndim > 1:
            sp = sp.mean(1)
        snr = rng.choice(SNRS)
        nz, nsr = rng.choice(noises)
        noisy = mix(sp.astype("float32"), sr, nz, nsr, snr)
        ap_ = os.path.join(wdir, os.path.basename(r["audio_path"]))
        sf.write(ap_, noisy, sr)
        out.write(json.dumps({"id": r["id"].replace("SparseLibriMix2", "SparseLibriMix2_noisy"),
                              "task": "SparseLibriMix2_noisy", "audio_path": ap_,
                              "instruction": r["instruction"], "label": r["label"], "snr_db": snr},
                             ensure_ascii=False) + "\n")
        n += 1
        if n % 50 == 0:
            print(f"[s2] {n} done", flush=True)
    out.close()
    print(f"[s2] finished {n} -> _manifest/SparseLibriMix2_noisy.jsonl")


if __name__ == "__main__":
    main()
