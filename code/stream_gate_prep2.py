"""Part C 预处理：门控端到端评测的数据准备。
1) 复用 stream_gate.synth/detect：合成 30 条 [clean|complex|clean] 流 + 逐秒检测器门控时间线。
2) 干净段伪参考：Mega-ASR 转写干净源音频截断段(干净语音上近金标，报告如实标注)。
3) 三种线索变体(SepFormer+ASR+kw_gate)：
   always   : 对整条流盲算线索(含干净段→污染)
   gated    : 仅对检测器标记区间算线索
   gated_gt : 仅对真值复杂区间算线索(oracle 上界)
输出 benchmarks/_agsc/stream_gate_eval.jsonl
"""
import json, os, sys, re, warnings, tempfile
warnings.filterwarnings("ignore")
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
sys.path.insert(0, ROOT + "/code")
import numpy as np, soundfile as sf, torch, torchaudio
import stream_gate as G
from speechbrain.inference.separation import SepformerSeparation
from context_synth_pipeline import MegaASRWrapper, PyannoteDiarizer
from noise_components import SileroVAD
from bench_build_agsc_s2sep import kw_gate

SR = 16000
N = 30


def clue_for_span(wav, a_s, b_s, sep, asr, pool):
    seg = wav[int(a_s * SR): int(b_s * SR)]
    if len(seg) < SR * 0.5:
        return {}
    t = torchaudio.functional.resample(torch.tensor(seg).float().unsqueeze(0), SR, 8000)
    est = sep.separate_batch(t)
    out = {}
    for i in range(min(2, est.shape[-1])):
        s16 = torchaudio.functional.resample(est[0, :, i].cpu().float(), 8000, 16000).numpy()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, s16, 16000)
            draft = str(asr.transcribe(tf.name))
        os.unlink(tf.name)
        out[f"SPEAKER_{i}"] = kw_gate(draft, pool)
    return out


def gate_spans(gate):
    spans, cur = [], None
    for s, on in enumerate(gate):
        if on and cur is None:
            cur = s
        if not on and cur is not None:
            spans.append((cur, s)); cur = None
    if cur is not None:
        spans.append((cur, len(gate)))
    return spans


def main():
    meta = G.synth(N)
    print(f"synth {len(meta)} streams", flush=True)
    diar = PyannoteDiarizer(token=os.environ.get("HF_TOKEN"))
    vad = SileroVAD()
    rows = G.detect(meta, diar, vad)
    # detector metrics on 30 (refresh stream_gate.json too)
    TP = FP = FN = TN = 0; lats = []
    for r in rows:
        for inj, g in zip(r["gate"], r["gt"]):
            if inj and g: TP += 1
            elif inj and not g: FP += 1
            elif not inj and g: FN += 1
            else: TN += 1
        cs = int(r["complex_start"])
        fire = [s for s, inj in enumerate(r["gate"]) if inj and s >= cs]
        if fire: lats.append(fire[0] - cs)
    P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9)
    summary = {"n_streams": len(rows), "precision": round(P, 3), "recall": round(R, 3),
               "f1": round(2 * P * R / (P + R + 1e-9), 3), "TP": TP, "FP": FP, "FN": FN, "TN": TN,
               "mean_trigger_latency_s": round(float(np.mean(lats)), 2) if lats else None,
               "clean_skip_rate": round(TN / (TN + FP + 1e-9), 3),
               "gated_vs_always_saving_pct": round(100 * (1 - (TP + FP) / (TP + FP + FN + TN)), 1)}
    json.dump({"summary": summary, "rows": rows}, open(ROOT + "/results/stream_gate.json", "w"), ensure_ascii=False, indent=2)
    print("DET30:", json.dumps(summary, ensure_ascii=False), flush=True)

    # sources for refs/pseudo-refs
    clean_man = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/SpeakerCounting_LibriTTS-TestClean.jsonl") if json.loads(l)["label"] == "one"]
    comp_man = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/SparseLibriMix2_noisy.jsonl")]
    pool = []
    for r in comp_man:
        pool += re.findall(r"[a-z']+", r["label"].lower())
    pool = [w for w in set(pool) if len(w) >= 3]

    sep = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr",
          savedir=ROOT + "/checkpoints/_sepformer", run_opts={"device": "cuda"})
    asr = MegaASRWrapper()

    # pseudo-refs for the truncated clean segments (transcribe the exact truncated audio)
    def pseudo_ref(path, dur):
        w = G.load16(path, dur)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, w, SR)
            txt = str(asr.transcribe(tf.name))
        os.unlink(tf.name)
        return txt.strip()

    outp = ROOT + "/benchmarks/_agsc/stream_gate_eval.jsonl"
    fout = open(outp, "w", encoding="utf-8")
    for i, r in enumerate(rows):
        wav = G.load16(r["wav"])
        ca = clean_man[i % len(clean_man)]; cb = clean_man[(i + 3) % len(clean_man)]
        ref_clean_a = pseudo_ref(ca["audio_path"], G.CLEAN_S)
        ref_clean_b = pseudo_ref(cb["audio_path"], G.TAIL_S)
        # complex refs: official labels, but stream only contains first COMPLEX_S seconds of comp audio
        # → 同截断曲线一样以完整 label 评(两条件公平), 记录截断时长
        comp = comp_man[i]
        clue_always = clue_for_span(wav, 0, r["dur"], sep, asr, pool)
        spans = gate_spans(r["gate"])
        clue_gated = {}
        if spans:
            a_s = min(s[0] for s in spans); b_s = max(s[1] for s in spans)
            clue_gated = clue_for_span(wav, a_s, b_s, sep, asr, pool)
        clue_gt = clue_for_span(wav, r["complex_start"], r["complex_end"], sep, asr, pool)
        rec = {"id": r["id"], "wav": r["wav"], "dur": r["dur"],
               "complex_start": r["complex_start"], "complex_end": r["complex_end"],
               "gate": r["gate"], "gate_spans": spans,
               "ref_clean_a": ref_clean_a, "ref_clean_b": ref_clean_b,
               "ref_spk": comp["label"].split("\n"),
               "clue_always": clue_always, "clue_gated": clue_gated, "clue_gt": clue_gt}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        print(f"[{i+1}/{len(rows)}] {r['id']} spans={spans}", flush=True)
    fout.close()
    print("DONE ->", outp)


if __name__ == "__main__":
    main()
