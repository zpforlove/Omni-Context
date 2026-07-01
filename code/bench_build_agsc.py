import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Stage B-2：给 benchmark 音频产 predicted-AGSC（说话人计数/多说话人任务只需日志，不跑 ASR）。
用 pyannote-3.1 日志 → n_speakers_est + 话轮时间线 + overlap。绝不看 benchmark 标签 → 零泄漏。
输出 _agsc/<task>.jsonl：{id, n_speakers_est, n_turns, overlap_regions, timeline, confidence}
用法：python bench_build_agsc.py --task SpeakerCounting_LibriTTS-TestClean
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

ROOT = OMNI_ROOT
BENCH = ROOT + "/benchmarks"
sys.path.insert(0, ROOT + "/code")


def build(task):
    from context_synth_pipeline import PyannoteDiarizer, detect_overlap
    man = [json.loads(l) for l in open(os.path.join(BENCH, "_manifest", task + ".jsonl"), encoding="utf-8")]
    dia = PyannoteDiarizer()
    outdir = os.path.join(BENCH, "_agsc")
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, task + ".jsonl")
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["id"] for l in open(outp, encoding="utf-8") if l.strip()}
    fout = open(outp, "a", encoding="utf-8")
    n = 0
    for r in man:
        if r["id"] in done:
            continue
        try:
            segs = dia(r["audio_path"])
        except Exception as e:
            print("ERR", r["id"], repr(e)[:120]); continue
        ov = detect_overlap(segs)
        spk = sorted({s["speaker"] for s in segs})
        timeline = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "speaker": s["speaker"]} for s in segs]
        rec = {"id": r["id"], "n_speakers_est": len(spk), "n_turns": len(segs),
               "overlap_regions": ov, "timeline": timeline,
               "confidence": "unverified", "provenance": "predicted"}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        n += 1
        if n % 20 == 0:
            print(f"[agsc] {task} {n} done")
    fout.close()
    print(f"[agsc] {task} finished +{n} -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    build(a.task)
