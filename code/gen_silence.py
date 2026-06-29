"""为每条样本生成等长静音 wav（AC 探针用：抽掉音频、保留 context）。"""
import json
import os
import numpy as np
import soundfile as sf

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
DS = "/cpfs_speech3/yulian.zpf/Omni-Context/Omni-Context-DataSet"
OUT = os.path.join(ROOT, "contexts_v3", "silence")


def main():
    os.makedirs(OUT, exist_ok=True)
    subset = [json.loads(l) for l in open(os.path.join(ROOT, "subsets", "eval_subset_600.jsonl"))]
    n = 0
    for r in subset:
        sid = r["sample_id"]
        try:
            info = sf.info(os.path.join(DS, r["audio_path"]))
            dur, sr = info.duration, info.samplerate
        except Exception:
            dur, sr = float(r.get("duration_sec", 6.0)), 16000
        wav = np.zeros(int(dur * sr), dtype=np.float32)
        sf.write(os.path.join(OUT, f"{sid}.wav"), wav, sr)
        n += 1
    print(f"[silence] wrote {n} silent wavs -> {OUT}")


if __name__ == "__main__":
    main()
