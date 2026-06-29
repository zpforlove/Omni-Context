"""Stage B Wave2-2：AMI TargetSpeaker-ASR 的 predicted-AGSC（仅用日志器，零泄漏）。
难点：目标说话人由 enrollment(audio2) 指定。用"前置 enrollment 联合日志"定位目标段：
  concat = [enroll | 0.5s 静音 | mix] → pyannote 日志 → enroll 区间[0,d2] 内占主导的说话人标签 = 目标
  → 取该标签在 mix 区间的时间窗(回移 -(d2+gap)) 作为"目标说话人时间窗"线索（不含转写）。
输出 _agsc/<task>.jsonl：{id, target_windows, n_speakers_est, overlap_regions, confidence}
用法：python bench_build_agsc_ami.py
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
BENCH = ROOT + "/benchmarks"
TASK = "TargetSpeaker-ASR_AMItest"
GAP = 0.5
sys.path.insert(0, ROOT + "/code")


def build():
    import numpy as np
    import soundfile as sf
    import tempfile
    from context_synth_pipeline import PyannoteDiarizer, detect_overlap
    man = [json.loads(l) for l in open(os.path.join(BENCH, "_manifest", TASK + ".jsonl"), encoding="utf-8")]
    dia = PyannoteDiarizer()
    outp = os.path.join(BENCH, "_agsc", TASK + ".jsonl")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["id"] for l in open(outp, encoding="utf-8") if l.strip()}
    fout = open(outp, "a", encoding="utf-8")
    n = 0
    for r in man:
        if r["id"] in done:
            continue
        try:
            mix, sr = sf.read(r["audio_path"])
            enr, sr2 = sf.read(r["audio2_path"])
            if mix.ndim > 1:
                mix = mix.mean(1)
            if enr.ndim > 1:
                enr = enr.mean(1)
            if sr2 != sr:
                import librosa
                enr = librosa.resample(enr, orig_sr=sr2, target_sr=sr)
            d2 = len(enr) / sr
            gap = np.zeros(int(GAP * sr))
            cat = np.concatenate([enr, gap, mix]).astype("float32")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, cat, sr)
                segs = dia(f.name)
            os.unlink(f.name)
        except Exception as e:
            print("ERR", r["id"], repr(e)[:140]); continue
        off = d2 + GAP
        # enroll 区间[0,d2]内占主导的标签 = 目标
        dur = {}
        for s in segs:
            ov = max(0.0, min(s["end"], d2) - max(s["start"], 0.0))
            if ov > 0:
                dur[s["speaker"]] = dur.get(s["speaker"], 0.0) + ov
        target = max(dur, key=dur.get) if dur else None
        # 目标在 mix 区间的时间窗（回移）
        tw = []
        for s in segs:
            if s["speaker"] == target and s["end"] > off + 0.05:
                st = max(0.0, s["start"] - off); en = s["end"] - off
                if en - st >= 0.2:
                    tw.append({"start": round(st, 2), "end": round(en, 2)})
        # mix 区间的 overlap（整体）
        mix_segs = [{"start": max(0.0, s["start"] - off), "end": s["end"] - off, "speaker": s["speaker"]}
                    for s in segs if s["end"] > off + 0.05]
        rec = {"id": r["id"], "target_windows": tw,
               "n_speakers_est": len({s["speaker"] for s in mix_segs}),
               "overlap_regions": detect_overlap(mix_segs),
               "confidence": "unverified", "provenance": "predicted"}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        n += 1
        if n % 20 == 0:
            print(f"[agsc-ami] {n} done")
    fout.close()
    print(f"[agsc-ami] finished +{n} -> {outp}")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    build()
