"""Stage N-C 整改：给 S2(SparseLibriMix2_noisy) 产【富线索】predicted-AGSC。
denoise(DeepFilterNet) → pyannote 日志 → 每段 Mega-ASR 草稿 → 每说话人 时间窗+内容草稿。
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
sys.path.insert(0, ROOT + "/code")


def main():
    from context_synth_pipeline import build_s2_agsc, PyannoteDiarizer, MegaASRWrapper, DeepFilterDenoiser
    task = "SparseLibriMix2_noisy"
    man = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/" + task + ".jsonl", encoding="utf-8")]
    dia = PyannoteDiarizer(); asr = MegaASRWrapper()
    try:
        dn = DeepFilterDenoiser()
        print("[s2rich] denoiser ready")
    except Exception as e:
        dn = None; print("[s2rich] denoiser FAIL, no-denoise:", repr(e)[:120])
    outp = ROOT + "/benchmarks/_agsc/" + task + ".jsonl"
    f = open(outp, "w", encoding="utf-8")  # 覆盖重建
    n = 0
    for r in man:
        try:
            ag = build_s2_agsc(r["audio_path"], dia, asr, denoiser=dn)
        except Exception as e:
            print("ERR", r["id"], repr(e)[:120]); continue
        ag["id"] = r["id"]
        f.write(json.dumps(ag, ensure_ascii=False) + "\n"); f.flush()
        n += 1
        if n % 30 == 0:
            print(f"[s2rich] {n} done", flush=True)
    f.close()
    print(f"[s2rich] finished +{n} -> {outp}")


if __name__ == "__main__":
    main()
