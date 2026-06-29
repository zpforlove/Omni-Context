"""GDPO R1a（omni-pipeline env）：合成 120 条训练流（comp 30-149，与评测 0-29 不重叠）。
每流：[clean 4s][complex 4s][clean 3s] + 2s 复杂/干净窗 + 真值区间线索（SepFormer+ASR 门控关键词）。
输出 benchmarks/_wav/_gdpo_train/ + benchmarks/_agsc/gdpo_train.jsonl
"""
import json, os, sys, re, warnings, tempfile
warnings.filterwarnings("ignore")
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
sys.path.insert(0, ROOT + "/code")
import numpy as np, soundfile as sf, torch, torchaudio
import stream_gate as G
from speechbrain.inference.separation import SepformerSeparation
from context_synth_pipeline import MegaASRWrapper
from bench_build_agsc_s2sep import kw_gate
from stream_gate_prep2 import clue_for_span

SR = 16000
OFFSET, N = 30, 120

def main():
    clean = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/SpeakerCounting_LibriTTS-TestClean.jsonl") if json.loads(l)["label"] == "one"]
    comp = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/SparseLibriMix2_noisy.jsonl")]
    pool = []
    for r in comp:
        pool += re.findall(r"[a-z']+", r["label"].lower())
    pool = [w for w in set(pool) if len(w) >= 3]

    sep = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr",
          savedir=ROOT + "/checkpoints/_sepformer", run_opts={"device": "cuda"})
    asr = MegaASRWrapper()

    outdir = ROOT + "/benchmarks/_wav/_gdpo_train"; os.makedirs(outdir, exist_ok=True)
    outp = ROOT + "/benchmarks/_agsc/gdpo_train.jsonl"
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["id"] for l in open(outp) if l.strip()}
    f = open(outp, "a", encoding="utf-8")
    for i in range(N):
        sid = f"gtrain_{i:03d}"
        if sid in done:
            continue
        c = comp[OFFSET + i]
        a = G.load16(clean[i % len(clean)]["audio_path"], G.CLEAN_S)
        x = G.load16(c["audio_path"]); x = x[: int(G.COMPLEX_S * SR)]
        b = G.load16(clean[(i + 3) % len(clean)]["audio_path"], G.TAIL_S)
        stream = np.concatenate([a, x, b]).astype("float32")
        cs, ce = len(a) / SR, (len(a) + len(x)) / SR
        wp = f"{outdir}/{sid}.wav"; sf.write(wp, stream, SR)
        # 2s windows
        wc = stream[int((cs + 1.0) * SR): int((cs + 3.0) * SR)]
        wk = stream[int(1.0 * SR): int(3.0 * SR)]
        wcp, wkp = f"{outdir}/{sid}__wC.wav", f"{outdir}/{sid}__wK.wav"
        sf.write(wcp, wc, SR); sf.write(wkp, wk, SR)
        try:
            clue = clue_for_span(stream, cs, ce, sep, asr, pool)
        except Exception as e:
            print("ERR clue", sid, repr(e)[:80]); clue = {}
        f.write(json.dumps({"id": sid, "wav": wp, "complex_start": cs, "complex_end": ce,
                            "win_complex": wcp, "win_clean": wkp,
                            "clue_gt": clue, "ref_spk": c["label"].split("\n")},
                           ensure_ascii=False) + "\n"); f.flush()
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{N}]", flush=True)
    f.close(); print("DONE prep streams")

if __name__ == "__main__":
    main()
