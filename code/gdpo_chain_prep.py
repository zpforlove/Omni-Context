import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""C1（omni-pipeline env）：全链路 GDPO 训练数据扩备。
1) 120 条复合训练流：补 clue@2s / clue@4s（复杂段前缀线索）+ 干净段 Mega-ASR 伪参考。
2) 合成 40 条纯干净训练流（门控负样本，~11s，三段干净拼接）+ 伪参考。
输出 benchmarks/_agsc/gdpo_chain_train.jsonl（kind=mix|clean）。
"""
import json, os, sys, re, warnings, tempfile
warnings.filterwarnings("ignore")
ROOT = OMNI_ROOT
sys.path.insert(0, ROOT + "/code")
import numpy as np, soundfile as sf
import stream_gate as G
from speechbrain.inference.separation import SepformerSeparation
from context_synth_pipeline import MegaASRWrapper
from stream_gate_prep2 import clue_for_span

SR = 16000

def asr_text(asr, wav_arr):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        sf.write(tf.name, wav_arr, SR)
        t = str(asr.transcribe(tf.name))
    os.unlink(tf.name)
    return t.strip()

def main():
    comp_rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/gdpo_train.jsonl") if l.strip()]
    clean = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/SpeakerCounting_LibriTTS-TestClean.jsonl") if json.loads(l)["label"] == "one"]
    comp_man = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/SparseLibriMix2_noisy.jsonl")]
    pool = []
    for r in comp_man:
        pool += re.findall(r"[a-z']+", r["label"].lower())
    pool = [w for w in set(pool) if len(w) >= 3]

    sep = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr",
          savedir=ROOT + "/checkpoints/_sepformer", run_opts={"device": "cuda"})
    asr = MegaASRWrapper()

    outp = ROOT + "/benchmarks/_agsc/gdpo_chain_train.jsonl"
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["id"] for l in open(outp) if l.strip()}
    f = open(outp, "a", encoding="utf-8")

    # 1) composite: add maturity clues + pseudo clean refs
    for i, r in enumerate(comp_rows):
        if r["id"] in done:
            continue
        wav = G.load16(r["wav"])
        cs, ce = r["complex_start"], r["complex_end"]
        try:
            clue2 = clue_for_span(wav, cs, cs + 2.0, sep, asr, pool)
            clue4 = clue_for_span(wav, cs, cs + 4.0, sep, asr, pool)
        except Exception as e:
            print("ERRclue", r["id"], repr(e)[:80]); clue2, clue4 = {}, {}
        ref_a = asr_text(asr, wav[: int(cs * SR)])
        ref_b = asr_text(asr, wav[int(ce * SR):])
        f.write(json.dumps({"id": r["id"], "kind": "mix", "wav": r["wav"],
                            "complex_start": cs, "complex_end": ce,
                            "ref_clean_a": ref_a, "ref_clean_b": ref_b,
                            "ref_spk": r["ref_spk"],
                            "clue_t2": clue2, "clue_t4": clue4, "clue_full": r["clue_gt"]},
                           ensure_ascii=False) + "\n"); f.flush()
        if (i + 1) % 20 == 0:
            print(f"[mix {i+1}/120]", flush=True)

    # 2) 40 clean streams
    outdir = ROOT + "/benchmarks/_wav/_gdpo_clean"; os.makedirs(outdir, exist_ok=True)
    for i in range(40):
        sid = f"gclean_{i:03d}"
        if sid in done:
            continue
        a = G.load16(clean[i % len(clean)]["audio_path"], 4.0)
        b = G.load16(clean[(i + 5) % len(clean)]["audio_path"], 4.0)
        c = G.load16(clean[(i + 7) % len(clean)]["audio_path"], 3.0)
        stream = np.concatenate([a, b, c]).astype("float32")
        wp = f"{outdir}/{sid}.wav"; sf.write(wp, stream, SR)
        refs = [asr_text(asr, x) for x in (a, b, c)]
        f.write(json.dumps({"id": sid, "kind": "clean", "wav": wp,
                            "ref_clean": refs}, ensure_ascii=False) + "\n"); f.flush()
        if (i + 1) % 10 == 0:
            print(f"[clean {i+1}/40]", flush=True)
    f.close(); print("DONE chain prep")

if __name__ == "__main__":
    main()
